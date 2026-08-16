"""Tests for end-to-end orchestration without expensive media operations."""

from dataclasses import replace
import os
from pathlib import Path

import pytest

from src.critic import CriticReasonCode, CriticStatus
from src.pipeline import PipelineConfig, PipelineError, PipelineServices, run_pipeline
from src.video_cutter import CutResult, CutStatus, MediaProbe, MediaVerification
from src.whisper_transcriber import WhisperTranscription, WhisperWord


def input_files(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    transcript = tmp_path / "captions.txt"
    transcript.write_text(
        "00:00\n00:10\nalpha words\n\n"
        "00:10\n00:20\nbeta words\n\n"
        "00:20\n00:30\ngamma words\n\n"
        "00:30\n00:40\ndelta words\n",
        encoding="utf-8",
    )
    return video, transcript


def transcription(words: tuple[WhisperWord, ...] | None = None) -> WhisperTranscription:
    return WhisperTranscription(
        model="tiny.en",
        language="en",
        processing_seconds=0.1,
        transcript="alpha words beta words gamma words delta words",
        words=words or (
            WhisperWord("alpha", 0.1, 0.5, 0.95),
            WhisperWord("words", 0.5, 1.0, 0.95),
            WhisperWord("beta", 10.1, 10.5, 0.95),
            WhisperWord("words", 10.5, 11.0, 0.95),
            WhisperWord("gamma", 20.1, 20.5, 0.95),
            WhisperWord("words", 20.5, 21.0, 0.95),
            WhisperWord("delta", 30.1, 30.5, 0.95),
            WhisperWord("words", 30.5, 31.0, 0.95),
        ),
    )


def fake_probe(*_args, **_kwargs) -> MediaProbe:
    return MediaProbe(40.0, True, True)


def fake_cut(calls: list[dict[str, object]]):
    def cut(input_path, output_path, start_boundary, end_boundary, **kwargs):
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"verified clip")
        start = start_boundary.resolver_decision.selected_timestamp
        end = end_boundary.resolver_decision.selected_timestamp
        assert start is not None and end is not None
        calls.append({"start": start, "end": end, **kwargs})
        verification = MediaVerification(
            True, end - start, end - start, 0.0, True, True, True
        )
        return CutResult(
            start,
            end,
            end - start,
            destination.resolve(),
            CutStatus.SUCCESS,
            verification,
            ("ffmpeg", "-i", str(input_path), str(destination.resolve())),
        )

    return cut


def services(calls: list[dict[str, object]], *, transcribe=None, refine=None) -> PipelineServices:
    return PipelineServices(
        transcribe=transcribe or (lambda *_args, **_kwargs: transcription()),
        probe=fake_probe,
        cut=fake_cut(calls),
        refine=refine or (lambda *_args, **_kwargs: pytest.fail("refinement is off")),
    )


def test_default_pipeline_runs_all_stages_and_exact_semantic_cuts(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    calls: list[dict[str, object]] = []
    logs: list[str] = []
    result = run_pipeline(
        PipelineConfig(video, transcript, tmp_path / "run"),
        services=services(calls),
        logger=logs.append,
    )

    assert result.aligned_boundary_count == 4
    assert len(result.executed_clips) == 3
    assert result.withheld_boundary_ids == ()
    assert result.refinement_enabled is False
    assert len(logs) == 7
    assert logs[0] == "[1/7] Validating input"
    assert logs[-1] == "[7/7] Generating audit report"
    assert all(call["start_refinement"] is None for call in calls)
    assert all(call["end_refinement"] is None for call in calls)
    assert [(call["start"], call["end"]) for call in calls] == [
        (0.0, 10.0), (10.0, 20.0), (20.0, 30.0)
    ]
    assert result.markdown_report_path.is_file()
    assert result.json_report_path.is_file()
    assert all(clip.cut_result.output_path.is_file() for clip in result.executed_clips)
    assert set(result.stage_timings_seconds) == {
        "Validating input", "Running speech-to-text", "Aligning timestamps",
        "Resolving conflicts", "Critic validation", "Executing approved cuts",
        "Generating audit report",
    }


def test_human_review_endpoint_withholds_only_affected_clips(tmp_path: Path, monkeypatch) -> None:
    import src.pipeline as pipeline

    video, transcript = input_files(tmp_path)
    calls: list[dict[str, object]] = []
    real_critic = pipeline.critique_decisions

    def critic_with_review(decisions, **kwargs):
        results = real_critic(decisions, **kwargs)
        results[1] = replace(
            results[1],
            status=CriticStatus.HUMAN_REVIEW,
            risk_score=1.0,
            reason_codes=(CriticReasonCode.NON_MONOTONIC_BOUNDARY,),
            explanation="Send this boundary to human review: test invariant.",
        )
        return results

    monkeypatch.setattr(pipeline, "critique_decisions", critic_with_review)
    result = run_pipeline(
        PipelineConfig(video, transcript, tmp_path / "run"),
        services=services(calls),
        logger=lambda _message: None,
    )

    assert result.withheld_boundary_ids == ("caption-block-2",)
    assert len(calls) == 1
    assert calls[0]["start"] == 20.0
    assert calls[0]["end"] == 30.0
    record = result.report.boundaries[1]
    assert record["critic"]["status"] == "human_review"
    assert record["execution"]["executed"] is False
    assert record["execution"]["final_timestamp_seconds"] is None


def test_refinement_runs_only_with_explicit_opt_in(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    calls: list[dict[str, object]] = []
    refinement_calls: list[float] = []

    def refine(_video, critic, _duration, **_kwargs):
        from src.boundary_refinement import (
            BoundaryRefinementResult, RefinementReason, RefinementValidationStatus,
        )
        original = critic.resolver_decision.selected_timestamp
        assert original is not None
        refinement_calls.append(original)
        return BoundaryRefinementResult(
            original, original, 0.0, 0.15, False, None,
            RefinementReason.NO_LOW_ENERGY_CANDIDATE,
            RefinementValidationStatus.VALID,
        )

    result = run_pipeline(
        PipelineConfig(video, transcript, tmp_path / "run", refinement_enabled=True),
        services=services(calls, refine=refine),
        logger=lambda _message: None,
    )

    assert refinement_calls == [0.0, 10.0, 20.0, 30.0]
    assert result.refinement_enabled is True
    assert all(call["start_refinement"] is not None for call in calls)
    assert result.report.configuration["refinement_enabled"] is True


@pytest.mark.parametrize("missing", ["video", "transcript"])
def test_missing_input_is_clear_and_creates_no_output(tmp_path: Path, missing: str) -> None:
    video, transcript = input_files(tmp_path)
    if missing == "video":
        video.unlink()
    else:
        transcript.unlink()
    output = tmp_path / "run"

    with pytest.raises(PipelineError, match="does not exist"):
        run_pipeline(
            PipelineConfig(video, transcript, output),
            services=services([]),
            logger=lambda _message: None,
        )
    assert not output.exists()


def test_whisper_failure_removes_staging_and_final_artifacts(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    output = tmp_path / "run"

    def fail(*_args, **_kwargs):
        raise RuntimeError("Whisper model failed")

    with pytest.raises(PipelineError, match="Running speech-to-text.*Whisper model failed"):
        run_pipeline(
            PipelineConfig(video, transcript, output),
            services=services([], transcribe=fail),
            logger=lambda _message: None,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".run.staging-*"))


def test_no_usable_alignment_fails_without_final_artifacts(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    output = tmp_path / "run"
    unmatched = transcription((WhisperWord("unrelated", 1.0, 2.0, 0.9),))

    with pytest.raises(PipelineError, match="no usable alignments"):
        run_pipeline(
            PipelineConfig(video, transcript, output),
            services=services([], transcribe=lambda *_a, **_k: unmatched),
            logger=lambda _message: None,
        )
    assert not output.exists()


def test_cut_failure_removes_all_partial_outputs(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    output = tmp_path / "run"
    failing_services = replace(
        services([]),
        cut=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("FFmpeg failed")),
    )

    with pytest.raises(PipelineError, match="Executing approved cuts.*FFmpeg failed"):
        run_pipeline(
            PipelineConfig(video, transcript, output),
            services=failing_services,
            logger=lambda _message: None,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".run.staging-*"))


def test_corrupt_media_probe_failure_is_clear(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    broken = replace(
        services([]),
        probe=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("invalid media")),
    )
    with pytest.raises(PipelineError, match="Validating input.*invalid media"):
        run_pipeline(
            PipelineConfig(video, transcript, tmp_path / "run"),
            services=broken,
            logger=lambda _message: None,
        )


def test_report_failure_removes_clips_and_all_partial_outputs(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    output = tmp_path / "run"
    broken = replace(
        services([]),
        write_reports=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("report write failed")
        ),
    )
    with pytest.raises(PipelineError, match="Generating audit report.*report write failed"):
        run_pipeline(
            PipelineConfig(video, transcript, output),
            services=broken,
            logger=lambda _message: None,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".run.staging-*"))


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    video, transcript = input_files(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(PipelineError, match="already exists"):
        run_pipeline(
            PipelineConfig(video, transcript, output),
            services=services([]),
            logger=lambda _message: None,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_explicit_ffmpeg_directory_is_available_to_whisper_and_restored(
    tmp_path: Path,
) -> None:
    video, transcript = input_files(tmp_path)
    executable = tmp_path / "tools" / "ffmpeg.exe"
    executable.parent.mkdir()
    original_path = os.environ.get("PATH", "")

    def inspect_path(*_args, **_kwargs):
        assert os.environ["PATH"].split(os.pathsep)[0] == str(executable.parent.resolve())
        return transcription()

    run_pipeline(
        PipelineConfig(
            video, transcript, tmp_path / "run", ffmpeg_path=executable
        ),
        services=services([], transcribe=inspect_path),
        logger=lambda _message: None,
    )
    assert os.environ.get("PATH", "") == original_path


def test_missing_path_is_restored_after_whisper_failure(
    tmp_path: Path, monkeypatch
) -> None:
    video, transcript = input_files(tmp_path)
    executable = tmp_path / "tools" / "ffmpeg.exe"
    executable.parent.mkdir()
    monkeypatch.delenv("PATH", raising=False)

    def fail(*_args, **_kwargs):
        assert os.environ["PATH"] == str(executable.parent.resolve())
        raise RuntimeError("Whisper failed")

    with pytest.raises(PipelineError, match="Whisper failed"):
        run_pipeline(
            PipelineConfig(
                video, transcript, tmp_path / "run", ffmpeg_path=executable
            ),
            services=services([], transcribe=fail),
            logger=lambda _message: None,
        )
    assert "PATH" not in os.environ

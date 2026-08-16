"""Tests for validated, Critic-gated FFmpeg cutting."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Callable

import pytest

from src import MetadataTimestamp, STTConfidence, STTTimestamp
from src.critic import CriticResult, critique_decisions
from src.resolver import ResolverEvidence, resolve_timestamp
from src.video_cutter import (
    BoundaryNotApprovedError,
    CutConfig,
    CutStatus,
    CutValidationError,
    CutVerificationError,
    FFmpegExecutionError,
    cut_video,
)


def approved_boundary(timestamp: float) -> CriticResult:
    decision = resolve_timestamp(
        MetadataTimestamp(timestamp), STTTimestamp(timestamp)
    )
    return critique_decisions([decision])[0]


def human_review_boundary(timestamp: float) -> CriticResult:
    decision = resolve_timestamp(
        MetadataTimestamp(timestamp),
        STTTimestamp(timestamp + 0.8),
        ResolverEvidence(stt_confidence=STTConfidence(0.6), alignment_quality=0.6),
    )
    return critique_decisions([decision])[0]


def config_for(tmp_path: Path) -> CutConfig:
    return CutConfig(
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        allowed_output_directory=tmp_path / "output" / "clips",
    )


def fake_runner(
    *,
    source_duration: float = 100.0,
    output_duration: float = 10.0,
    output_streams: tuple[str, ...] = ("video", "audio"),
    ffmpeg_returncode: int = 0,
    create_output: bool = True,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    probe_calls = 0

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal probe_calls
        if Path(command[0]).name == "ffprobe":
            probe_calls += 1
            duration = source_duration if probe_calls == 1 else output_duration
            streams = (
                ("video", "audio") if probe_calls == 1 else output_streams
            )
            payload = {
                "format": {"duration": str(duration)},
                "streams": [{"codec_type": item} for item in streams],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if ffmpeg_returncode == 0 and create_output:
            Path(command[-1]).write_bytes(b"mock media")
        return subprocess.CompletedProcess(
            command,
            ffmpeg_returncode,
            "",
            "encoder failed" if ffmpeg_returncode else "",
        )

    return run


def make_input(tmp_path: Path) -> Path:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    return source


def test_successful_cut_is_verified_and_structured(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    destination = tmp_path / "output" / "clips" / "clip.mp4"

    result = cut_video(
        source,
        destination,
        approved_boundary(10.0),
        approved_boundary(20.0),
        config=config_for(tmp_path),
        runner=fake_runner(),
    )

    assert result.status is CutStatus.SUCCESS
    assert result.actual_output_duration_seconds == 10.0
    assert result.verification.has_video_stream is True
    assert result.verification.has_audio_stream is True
    assert result.verification.duration_within_tolerance is True
    assert result.ffmpeg_command.count("-i") == 1
    assert result.ffmpeg_command[result.ffmpeg_command.index("-i") + 1] == str(
        source.resolve()
    )
    assert result.ffmpeg_command.index("-ss") > result.ffmpeg_command.index("-i")
    assert result.ffmpeg_command[result.ffmpeg_command.index("-c:v") + 1] == "libx264"
    assert result.ffmpeg_command[result.ffmpeg_command.index("-c:a") + 1] == "aac"


def test_human_review_boundary_is_never_executed(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(BoundaryNotApprovedError, match="human review"):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            human_review_boundary(10.0),
            approved_boundary(20.0),
            config=config_for(tmp_path),
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )


def test_human_review_end_boundary_is_never_executed(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(BoundaryNotApprovedError, match="end boundary"):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            approved_boundary(10.0),
            human_review_boundary(20.0),
            config=config_for(tmp_path),
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1.0, 20.0, "non-negative"),
        (20.0, 20.0, "greater than start"),
        (21.0, 20.0, "greater than start"),
        (float("nan"), 20.0, "finite"),
        (10.0, float("inf"), "finite"),
    ],
)
def test_invalid_boundaries_are_rejected(
    tmp_path: Path, start: float, end: float, message: str
) -> None:
    source = make_input(tmp_path)
    start_result = approved_boundary(10.0)
    end_result = approved_boundary(20.0)
    start_result = replace(
        start_result,
        resolver_decision=replace(
            start_result.resolver_decision, selected_timestamp=start
        ),
    )
    end_result = replace(
        end_result,
        resolver_decision=replace(end_result.resolver_decision, selected_timestamp=end),
    )

    with pytest.raises(CutValidationError, match=message):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            start_result,
            end_result,
            config=config_for(tmp_path),
            runner=fake_runner(),
        )


def test_end_cannot_exceed_source_duration(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(CutValidationError, match="exceeds source duration"):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            approved_boundary(90.0),
            approved_boundary(101.0),
            config=config_for(tmp_path),
            runner=fake_runner(),
        )


def test_output_must_stay_inside_allowed_directory(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(CutValidationError, match="must be inside"):
        cut_video(
            source,
            tmp_path / "elsewhere" / "clip.mp4",
            approved_boundary(10.0),
            approved_boundary(20.0),
            config=config_for(tmp_path),
            runner=fake_runner(),
        )


@pytest.mark.parametrize(
    ("streams", "message"),
    [(('video',), "audio stream is missing"), (('audio',), "video stream is missing")],
)
def test_missing_output_stream_fails_verification(
    tmp_path: Path, streams: tuple[str, ...], message: str
) -> None:
    source = make_input(tmp_path)

    with pytest.raises(CutVerificationError, match=message):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            approved_boundary(10.0),
            approved_boundary(20.0),
            config=config_for(tmp_path),
            runner=fake_runner(output_streams=streams),
        )


def test_duration_mismatch_fails_verification(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(CutVerificationError, match="duration differs"):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            approved_boundary(10.0),
            approved_boundary(20.0),
            config=config_for(tmp_path),
            runner=fake_runner(output_duration=9.0),
        )


def test_missing_output_file_fails_verification(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(CutVerificationError, match="did not create"):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            approved_boundary(10.0),
            approved_boundary(20.0),
            config=config_for(tmp_path),
            runner=fake_runner(create_output=False),
        )


def test_ffmpeg_error_includes_stderr(tmp_path: Path) -> None:
    source = make_input(tmp_path)

    with pytest.raises(FFmpegExecutionError, match="encoder failed"):
        cut_video(
            source,
            tmp_path / "output" / "clips" / "clip.mp4",
            approved_boundary(10.0),
            approved_boundary(20.0),
            config=config_for(tmp_path),
            runner=fake_runner(ffmpeg_returncode=1),
        )

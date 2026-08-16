"""End-to-end orchestration of existing TimeSync AI components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import time
from typing import Callable, TypeVar
from uuid import uuid4

from .alignment import AlignedBoundaryCandidate, align_caption_boundaries, resolver_evidence_for
from .boundary_refinement import BoundaryRefinementResult, RefinementConfig, refine_boundary
from .caption_parser import CaptionRecord, parse_caption_file, write_caption_json
from .critic import CriticConfig, CriticResult, CriticStatus, critique_decisions
from .reporting import AuditReport, build_audit_report, write_audit_reports
from .resolver import ResolverConfig, resolve_timestamp
from .video_cutter import CutConfig, CutResult, MediaProbe, cut_video, probe_media
from .whisper_transcriber import WhisperTranscription, transcribe_video, write_whisper_json


LogFunction = Callable[[str], None]
StageResult = TypeVar("StageResult")


class PipelineError(RuntimeError):
    """A stage-aware error raised by the application pipeline."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Explicit inputs and component configuration for one pipeline run."""

    video_path: str | Path
    transcript_path: str | Path
    output_directory: str | Path
    whisper_model: str = "tiny.en"
    whisper_download_root: str | Path | None = None
    refinement_enabled: bool = False
    resolver_config: ResolverConfig = ResolverConfig()
    critic_config: CriticConfig = CriticConfig()
    refinement_config: RefinementConfig = RefinementConfig()
    ffmpeg_path: str | Path = "ffmpeg"
    ffprobe_path: str | Path = "ffprobe"


@dataclass(frozen=True, slots=True)
class ExecutedClip:
    """One published caption clip and its validated boundary audit."""

    clip_id: str
    caption_text: str
    start_boundary_id: str
    end_boundary_id: str
    cut_result: CutResult


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Structured outcome of a successfully published pipeline run."""

    output_directory: Path
    caption_json_path: Path
    whisper_json_path: Path
    markdown_report_path: Path
    json_report_path: Path
    aligned_boundary_count: int
    major_conflict_count: int
    approved_boundary_count: int
    withheld_boundary_ids: tuple[str, ...]
    executed_clips: tuple[ExecutedClip, ...]
    refinement_enabled: bool
    stage_timings_seconds: dict[str, float]
    total_processing_seconds: float
    report: AuditReport


@dataclass(frozen=True, slots=True)
class PipelineServices:
    """Injectable expensive I/O operations used by orchestration tests."""

    transcribe: Callable[..., WhisperTranscription] = transcribe_video
    probe: Callable[..., MediaProbe] = probe_media
    cut: Callable[..., CutResult] = cut_video
    refine: Callable[..., BoundaryRefinementResult] = refine_boundary
    write_reports: Callable[..., None] = write_audit_reports


def _run_stage(
    number: int,
    label: str,
    timings: dict[str, float],
    logger: LogFunction,
    operation: Callable[[], StageResult],
) -> StageResult:
    logger(f"[{number}/7] {label}")
    started = time.perf_counter()
    try:
        return operation()
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(label, str(error) or error.__class__.__name__) from error
    finally:
        timings[label] = round(time.perf_counter() - started, 6)


def _boundary_id(candidate: AlignedBoundaryCandidate) -> str:
    return f"caption-block-{candidate.caption_index + 1}"


@contextmanager
def _configured_ffmpeg_path(ffmpeg_path: str | Path):
    """Temporarily expose an explicitly configured FFmpeg to Whisper."""
    executable = Path(ffmpeg_path)
    if executable.parent == Path("."):
        yield
        return
    previous_path = os.environ.get("PATH")
    suffix = f"{os.pathsep}{previous_path}" if previous_path else ""
    os.environ["PATH"] = str(executable.resolve().parent) + suffix
    try:
        yield
    finally:
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path


def run_pipeline(
    config: PipelineConfig,
    *,
    logger: LogFunction = print,
    services: PipelineServices | None = None,
) -> PipelineResult:
    """Run all existing components and atomically publish successful outputs.

    Consecutive aligned boundaries define a caption clip. A clip is executed
    only when both endpoints are Critic-approved. Refinement is never called
    unless ``refinement_enabled`` is explicitly true.
    """
    services = services or PipelineServices()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    video = Path(config.video_path).resolve()
    transcript = Path(config.transcript_path).resolve()
    output = Path(config.output_directory).resolve()
    staging = output.parent / f".{output.name}.staging-{uuid4().hex}"
    published = False

    try:
        def validate() -> tuple[list[CaptionRecord], MediaProbe]:
            if not video.is_file():
                raise FileNotFoundError(f"video file does not exist: {video}")
            if not transcript.is_file():
                raise FileNotFoundError(f"caption metadata does not exist: {transcript}")
            if output.exists():
                raise FileExistsError(f"output directory already exists: {output}")
            captions = parse_caption_file(transcript)
            if not captions:
                raise ValueError("caption metadata contains no caption blocks")
            media = services.probe(video, ffprobe_path=config.ffprobe_path)
            if (
                not math.isfinite(media.duration_seconds)
                or media.duration_seconds <= 0
                or not media.has_video_stream
                or not media.has_audio_stream
            ):
                raise ValueError("input media must have positive duration, video, and audio")
            staging.mkdir(parents=True, exist_ok=False)
            return captions, media

        captions, media = _run_stage(1, "Validating input", timings, logger, validate)

        def run_stt() -> WhisperTranscription:
            with _configured_ffmpeg_path(config.ffmpeg_path):
                result = services.transcribe(
                    video,
                    model_name=config.whisper_model,
                    download_root=config.whisper_download_root,
                )
            write_caption_json(captions, staging / "caption_metadata.json")
            write_whisper_json(result, staging / "whisper_words.json")
            return result

        transcription = _run_stage(
            2, "Running speech-to-text", timings, logger, run_stt
        )

        def align() -> list[AlignedBoundaryCandidate]:
            result = align_caption_boundaries(captions, transcription.words)
            if not result:
                raise ValueError("caption and STT data produced no usable alignments")
            return result

        candidates = _run_stage(3, "Aligning timestamps", timings, logger, align)

        decisions = _run_stage(
            4,
            "Resolving conflicts",
            timings,
            logger,
            lambda: [
                resolve_timestamp(
                    candidate.metadata_timestamp,
                    candidate.stt_timestamp,
                    resolver_evidence_for(candidate, candidates),
                    config=config.resolver_config,
                )
                for candidate in candidates
            ],
        )
        critics = _run_stage(
            5,
            "Critic validation",
            timings,
            logger,
            lambda: critique_decisions(decisions, config=config.critic_config),
        )

        refinements: dict[int, BoundaryRefinementResult] = {}
        executed: list[ExecutedClip] = []
        executed_boundary_ids: set[str] = set()
        executed_cut_ids: set[str] = set()

        def execute_cuts() -> None:
            approved_positions = [
                index
                for index, critic in enumerate(critics)
                if critic.status is CriticStatus.APPROVED
            ]
            if config.refinement_enabled:
                for approved_index, position in enumerate(approved_positions):
                    previous_position = (
                        approved_positions[approved_index - 1]
                        if approved_index > 0
                        else None
                    )
                    next_position = (
                        approved_positions[approved_index + 1]
                        if approved_index + 1 < len(approved_positions)
                        else None
                    )
                    refinements[candidates[position].caption_index] = services.refine(
                        video,
                        critics[position],
                        media.duration_seconds,
                        previous_approved_timestamp=(
                            critics[previous_position].resolver_decision.selected_timestamp
                            if previous_position is not None
                            else None
                        ),
                        next_approved_timestamp=(
                            critics[next_position].resolver_decision.selected_timestamp
                            if next_position is not None
                            else None
                        ),
                        ffmpeg_path=config.ffmpeg_path,
                        config=config.refinement_config,
                    )

            cut_config = CutConfig(
                ffmpeg_path=config.ffmpeg_path,
                ffprobe_path=config.ffprobe_path,
                allowed_output_directory=staging / "clips",
            )
            for index in range(len(candidates) - 1):
                start_critic = critics[index]
                end_critic = critics[index + 1]
                if (
                    start_critic.status is not CriticStatus.APPROVED
                    or end_critic.status is not CriticStatus.APPROVED
                ):
                    continue
                start_id = _boundary_id(candidates[index])
                end_id = _boundary_id(candidates[index + 1])
                clip_id = f"clip-{candidates[index].caption_index + 1:03d}"
                filename = f"{clip_id}.mp4"
                result = services.cut(
                    video,
                    staging / "clips" / filename,
                    start_critic,
                    end_critic,
                    start_refinement=refinements.get(candidates[index].caption_index),
                    end_refinement=refinements.get(candidates[index + 1].caption_index),
                    config=cut_config,
                )
                executed.append(
                    ExecutedClip(
                        clip_id=clip_id,
                        caption_text=candidates[index].caption_text,
                        start_boundary_id=start_id,
                        end_boundary_id=end_id,
                        cut_result=result,
                    )
                )
                executed_cut_ids.add(clip_id)
                executed_boundary_ids.update((start_id, end_id))

        _run_stage(6, "Executing approved cuts", timings, logger, execute_cuts)

        def generate_reports() -> AuditReport:
            report = build_audit_report(
                candidates,
                critics,
                conflict_threshold_seconds=(
                    config.resolver_config.major_conflict_threshold_seconds
                ),
                resolver_config=config.resolver_config,
                critic_config=config.critic_config,
                refinement_enabled=config.refinement_enabled,
                refinements=refinements,
                executed_boundary_ids=executed_boundary_ids,
                executed_cut_ids=executed_cut_ids,
            )
            services.write_reports(
                report, staging / "report.md", staging / "report.json"
            )
            return report

        report = _run_stage(
            7, "Generating audit report", timings, logger, generate_reports
        )
        try:
            staging.replace(output)
        except OSError as error:
            raise PipelineError("Publishing outputs", str(error)) from error
        published = True

        published_clips = tuple(
            replace(
                item,
                cut_result=replace(
                    item.cut_result,
                    output_path=output / "clips" / item.cut_result.output_path.name,
                    ffmpeg_command=(
                        item.cut_result.ffmpeg_command[:-1]
                        + (str(output / "clips" / item.cut_result.output_path.name),)
                    ),
                ),
            )
            for item in executed
        )
        withheld = tuple(
            _boundary_id(candidate)
            for candidate, critic in zip(candidates, critics)
            if critic.status is CriticStatus.HUMAN_REVIEW
        )
        total = round(time.perf_counter() - started, 6)
        return PipelineResult(
            output_directory=output,
            caption_json_path=output / "caption_metadata.json",
            whisper_json_path=output / "whisper_words.json",
            markdown_report_path=output / "report.md",
            json_report_path=output / "report.json",
            aligned_boundary_count=len(candidates),
            major_conflict_count=sum(
                decision.disagreement_seconds is not None
                and decision.disagreement_seconds
                > config.resolver_config.major_conflict_threshold_seconds
                for decision in decisions
            ),
            approved_boundary_count=sum(
                critic.status is CriticStatus.APPROVED for critic in critics
            ),
            withheld_boundary_ids=withheld,
            executed_clips=published_clips,
            refinement_enabled=config.refinement_enabled,
            stage_timings_seconds=timings,
            total_processing_seconds=total,
            report=report,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

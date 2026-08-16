"""Validated, frame-accurate FFmpeg cutting for Critic-approved boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
import subprocess
from typing import Callable, Sequence

from .boundary_refinement import (
    BoundaryRefinementResult,
    RefinementValidationStatus,
)
from .critic import CriticResult, CriticStatus


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class CutStatus(str, Enum):
    """Execution status of a completed cut."""

    SUCCESS = "success"


class VideoCutError(RuntimeError):
    """Base error for validated cutting failures."""


class BoundaryNotApprovedError(VideoCutError):
    """Raised when a requested boundary is not approved for automation."""


class CutValidationError(VideoCutError):
    """Raised before execution when a cut request is invalid."""


class FFmpegExecutionError(VideoCutError):
    """Raised when FFmpeg or FFprobe exits unsuccessfully."""


class CutVerificationError(VideoCutError):
    """Raised when generated media fails post-execution verification."""


@dataclass(frozen=True, slots=True)
class CutConfig:
    """Executable paths and deterministic encoding/verification settings."""

    ffmpeg_path: str | Path = "ffmpeg"
    ffprobe_path: str | Path = "ffprobe"
    allowed_output_directory: str | Path = "output/clips"
    duration_tolerance_seconds: float = 0.25
    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 18
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.duration_tolerance_seconds)
            or self.duration_tolerance_seconds < 0
        ):
            raise ValueError("duration_tolerance_seconds must be non-negative")
        if not 0 <= self.video_crf <= 51:
            raise ValueError("video_crf must be between 0 and 51")


@dataclass(frozen=True, slots=True)
class MediaVerification:
    """FFprobe evidence collected from a generated clip."""

    output_exists: bool
    expected_duration_seconds: float
    actual_duration_seconds: float
    duration_error_seconds: float
    duration_within_tolerance: bool
    has_video_stream: bool
    has_audio_stream: bool


@dataclass(frozen=True, slots=True)
class CutResult:
    """Structured result of a successful, verified cut."""

    requested_start_seconds: float
    requested_end_seconds: float
    actual_output_duration_seconds: float
    output_path: Path
    status: CutStatus
    verification: MediaVerification
    ffmpeg_command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MediaProbe:
    duration_seconds: float
    has_video_stream: bool
    has_audio_stream: bool


def _run(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    tool_name: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            list(command), capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise FFmpegExecutionError(f"could not start {tool_name}: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no error output").strip()
        raise FFmpegExecutionError(
            f"{tool_name} failed with exit code {completed.returncode}: {detail}"
        )
    return completed


def _probe_media(
    path: Path,
    *,
    ffprobe_path: str | Path,
    runner: CommandRunner,
) -> _MediaProbe:
    command = (
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(path),
    )
    completed = _run(command, runner=runner, tool_name="FFprobe")
    try:
        payload = json.loads(completed.stdout)
        duration = float(payload["format"]["duration"])
        stream_types = {
            stream.get("codec_type") for stream in payload.get("streams", [])
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CutVerificationError(
            f"FFprobe returned invalid media metadata for {path}"
        ) from error
    if not math.isfinite(duration) or duration < 0:
        raise CutVerificationError(f"FFprobe returned an invalid duration for {path}")
    return _MediaProbe(
        duration_seconds=duration,
        has_video_stream="video" in stream_types,
        has_audio_stream="audio" in stream_types,
    )


def _approved_timestamp(boundary: CriticResult, label: str) -> float:
    if boundary.status is not CriticStatus.APPROVED:
        raise BoundaryNotApprovedError(
            f"{label} boundary is {boundary.status.value}; human review is required"
        )
    timestamp = boundary.resolver_decision.selected_timestamp
    if timestamp is None or not math.isfinite(timestamp):
        raise CutValidationError(f"{label} boundary has no finite selected timestamp")
    return timestamp


def _execution_timestamp(
    semantic_timestamp: float,
    refinement: BoundaryRefinementResult | None,
    label: str,
) -> float:
    if refinement is None:
        return semantic_timestamp
    if refinement.validation_status is not RefinementValidationStatus.VALID:
        raise BoundaryNotApprovedError(
            f"{label} refinement is not approved for execution"
        )
    if refinement.original_selected_timestamp is None or not math.isclose(
        refinement.original_selected_timestamp,
        semantic_timestamp,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise CutValidationError(
            f"{label} refinement does not match the Critic-approved timestamp"
        )
    refined = refinement.refined_timestamp
    if refined is None or not math.isfinite(refined):
        raise CutValidationError(f"{label} refinement has no finite timestamp")
    maximum_shift = refinement.maximum_shift_seconds
    if not math.isfinite(maximum_shift) or maximum_shift < 0:
        raise CutValidationError(f"{label} refinement has an invalid shift limit")
    adjustment = refined - semantic_timestamp
    if abs(adjustment) > maximum_shift + 1e-9:
        raise CutValidationError(
            f"{label} refinement exceeds its configured shift limit"
        )
    if not math.isclose(
        refinement.adjustment_milliseconds,
        adjustment * 1000,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise CutValidationError(f"{label} refinement adjustment is inconsistent")
    was_adjusted = not math.isclose(
        refined, semantic_timestamp, rel_tol=0.0, abs_tol=1e-9
    )
    if refinement.refinement_applied is not was_adjusted:
        raise CutValidationError(f"{label} refinement applied flag is inconsistent")
    return refined


def _safe_output_path(
    input_path: Path,
    output_path: Path,
    allowed_directory: Path,
) -> Path:
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    resolved_allowed = allowed_directory.resolve()
    if resolved_output == resolved_input:
        raise CutValidationError("output path cannot overwrite the input video")
    if not resolved_output.is_relative_to(resolved_allowed):
        raise CutValidationError(f"output path must be inside {resolved_allowed}")
    if resolved_output.suffix.casefold() != ".mp4":
        raise CutValidationError("output path must use the .mp4 extension")
    return resolved_output


def cut_video(
    input_path: str | Path,
    output_path: str | Path,
    start_boundary: CriticResult,
    end_boundary: CriticResult,
    *,
    start_refinement: BoundaryRefinementResult | None = None,
    end_refinement: BoundaryRefinementResult | None = None,
    config: CutConfig | None = None,
    runner: CommandRunner | None = None,
) -> CutResult:
    """Cut original media between two independently approved boundaries.

    By default, the exact Resolver-selected timestamps are used. Experimental
    execution refinements apply only when explicitly supplied. Video and the
    original soundtrack are decoded from the source and encoded into the
    output; no Whisper-derived or enhanced audio is introduced.
    """
    config = config or CutConfig()
    runner = runner or subprocess.run
    source = Path(input_path)
    if not source.is_file():
        raise CutValidationError(f"input video does not exist: {source}")

    semantic_start = _approved_timestamp(start_boundary, "start")
    semantic_end = _approved_timestamp(end_boundary, "end")
    start = _execution_timestamp(semantic_start, start_refinement, "start")
    end = _execution_timestamp(semantic_end, end_refinement, "end")
    if start < 0:
        raise CutValidationError("start timestamp must be non-negative")
    if end <= start:
        raise CutValidationError("end timestamp must be greater than start")

    destination = _safe_output_path(
        source, Path(output_path), Path(config.allowed_output_directory)
    )
    source_probe = _probe_media(
        source, ffprobe_path=config.ffprobe_path, runner=runner
    )
    if end > source_probe.duration_seconds:
        raise CutValidationError(
            f"end timestamp {end:.3f}s exceeds source duration "
            f"{source_probe.duration_seconds:.3f}s"
        )
    if not source_probe.has_video_stream or not source_probe.has_audio_stream:
        raise CutValidationError("input must contain both video and audio streams")

    destination.parent.mkdir(parents=True, exist_ok=True)
    requested_duration = end - start
    command = (
        str(config.ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source.resolve()),
        "-ss",
        f"{start:.9f}",
        "-t",
        f"{requested_duration:.9f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        config.video_codec,
        "-preset",
        config.video_preset,
        "-crf",
        str(config.video_crf),
        "-c:a",
        config.audio_codec,
        "-b:a",
        config.audio_bitrate,
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
        str(destination),
    )
    _run(command, runner=runner, tool_name="FFmpeg")

    if not destination.is_file():
        raise CutVerificationError(f"FFmpeg did not create output file: {destination}")
    output_probe = _probe_media(
        destination, ffprobe_path=config.ffprobe_path, runner=runner
    )
    duration_error = abs(output_probe.duration_seconds - requested_duration)
    verification = MediaVerification(
        output_exists=True,
        expected_duration_seconds=requested_duration,
        actual_duration_seconds=output_probe.duration_seconds,
        duration_error_seconds=duration_error,
        duration_within_tolerance=(
            duration_error <= config.duration_tolerance_seconds
        ),
        has_video_stream=output_probe.has_video_stream,
        has_audio_stream=output_probe.has_audio_stream,
    )
    failures: list[str] = []
    if not verification.duration_within_tolerance:
        failures.append(
            f"duration differs by {verification.duration_error_seconds:.3f}s"
        )
    if not verification.has_video_stream:
        failures.append("video stream is missing")
    if not verification.has_audio_stream:
        failures.append("audio stream is missing")
    if failures:
        raise CutVerificationError(
            f"output verification failed for {destination}: " + "; ".join(failures)
        )

    return CutResult(
        requested_start_seconds=start,
        requested_end_seconds=end,
        actual_output_duration_seconds=output_probe.duration_seconds,
        output_path=destination,
        status=CutStatus.SUCCESS,
        verification=verification,
        ffmpeg_command=command,
    )

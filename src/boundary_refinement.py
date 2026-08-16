"""Optional experimental audio-energy refinement of approved boundaries.

This module is never invoked by the default cutting path. Callers must request
refinement explicitly, and its output remains separate from the Resolver's
semantic timestamp.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import subprocess
from typing import Callable, Sequence

from .critic import CriticResult, CriticStatus


AudioCommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


class RefinementValidationStatus(str, Enum):
    """Whether a refinement result is safe for downstream execution."""

    VALID = "valid"
    NOT_APPROVED = "not_approved"


class RefinementReason(str, Enum):
    """Deterministic outcome reasons for boundary refinement."""

    REFINED_TO_LOW_ENERGY = "refined_to_low_energy"
    ORIGINAL_ALREADY_LOW_ENERGY = "original_already_low_energy"
    NO_LOW_ENERGY_CANDIDATE = "no_low_energy_candidate"
    NEIGHBOUR_PROTECTION = "neighbour_protection"
    HUMAN_REVIEW_BOUNDARY = "human_review_boundary"


class BoundaryRefinementError(RuntimeError):
    """Raised when refinement input or audio extraction is invalid."""


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """Explicit deterministic audio-analysis and safety thresholds."""

    maximum_shift_seconds: float = 0.15
    sample_rate_hz: int = 16_000
    frame_duration_seconds: float = 0.02
    hop_duration_seconds: float = 0.01
    silence_rms_threshold: float = 0.02
    minimum_energy_ratio: float = 0.8
    minimum_neighbour_gap_seconds: float = 0.01

    def __post_init__(self) -> None:
        positive = {
            "maximum_shift_seconds": self.maximum_shift_seconds,
            "frame_duration_seconds": self.frame_duration_seconds,
            "hop_duration_seconds": self.hop_duration_seconds,
            "minimum_neighbour_gap_seconds": self.minimum_neighbour_gap_seconds,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in {
            "silence_rms_threshold": self.silence_rms_threshold,
            "minimum_energy_ratio": self.minimum_energy_ratio,
        }.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.frame_duration_seconds <= 0 or self.hop_duration_seconds <= 0:
            raise ValueError("frame and hop durations must be positive")


@dataclass(frozen=True, slots=True)
class AudioEnergyEvidence:
    """Measured PCM evidence supporting or rejecting a small adjustment."""

    extraction_start_seconds: float
    extraction_end_seconds: float
    sample_rate_hz: int
    frame_duration_seconds: float
    hop_duration_seconds: float
    silence_rms_threshold: float
    original_rms: float
    lowest_candidate_rms: float | None
    lowest_candidate_timestamp: float | None
    low_energy_candidate_count: int
    valid_candidate_count: int


@dataclass(frozen=True, slots=True)
class BoundaryRefinementResult:
    """Auditable experimental adjustment preserving the semantic timestamp."""

    original_selected_timestamp: float | None
    refined_timestamp: float | None
    adjustment_milliseconds: float
    maximum_shift_seconds: float
    refinement_applied: bool
    evidence: AudioEnergyEvidence | None
    reason: RefinementReason
    validation_status: RefinementValidationStatus


def _rms(samples: Sequence[int]) -> float:
    if not samples:
        return math.inf
    scale = 32768.0
    return math.sqrt(sum((sample / scale) ** 2 for sample in samples) / len(samples))


def _extract_pcm(
    media_path: Path,
    start: float,
    duration: float,
    *,
    ffmpeg_path: str | Path,
    sample_rate_hz: int,
    runner: AudioCommandRunner,
) -> tuple[int, ...]:
    command = (
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path.resolve()),
        "-ss",
        f"{start:.9f}",
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    )
    try:
        completed = runner(
            list(command), capture_output=True, check=False
        )
    except OSError as error:
        raise BoundaryRefinementError(
            f"could not start FFmpeg audio analysis: {error}"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace") if completed.stderr else ""
        raise BoundaryRefinementError(
            f"FFmpeg audio analysis failed with exit code {completed.returncode}: "
            f"{stderr.strip() or 'no error output'}"
        )
    if len(completed.stdout) % 2:
        raise BoundaryRefinementError("FFmpeg returned malformed 16-bit PCM audio")
    samples = array("h")
    samples.frombytes(completed.stdout)
    return tuple(samples)


def refine_boundary(
    media_path: str | Path,
    boundary: CriticResult,
    media_duration_seconds: float,
    *,
    previous_approved_timestamp: float | None = None,
    next_approved_timestamp: float | None = None,
    ffmpeg_path: str | Path = "ffmpeg",
    config: RefinementConfig | None = None,
    runner: AudioCommandRunner | None = None,
) -> BoundaryRefinementResult:
    """Explicitly request experimental low-energy refinement for one boundary."""
    config = config or RefinementConfig()
    if not math.isfinite(media_duration_seconds) or media_duration_seconds <= 0:
        raise BoundaryRefinementError("media duration must be finite and positive")

    original = boundary.resolver_decision.selected_timestamp
    if boundary.status is not CriticStatus.APPROVED:
        return BoundaryRefinementResult(
            original_selected_timestamp=original,
            refined_timestamp=original,
            adjustment_milliseconds=0.0,
            maximum_shift_seconds=config.maximum_shift_seconds,
            refinement_applied=False,
            evidence=None,
            reason=RefinementReason.HUMAN_REVIEW_BOUNDARY,
            validation_status=RefinementValidationStatus.NOT_APPROVED,
        )
    if original is None or not math.isfinite(original):
        raise BoundaryRefinementError(
            "approved boundary must have a finite selected timestamp"
        )
    if not 0 <= original <= media_duration_seconds:
        raise BoundaryRefinementError("selected timestamp is outside media duration")

    for name, neighbour in (
        ("previous", previous_approved_timestamp),
        ("next", next_approved_timestamp),
    ):
        if neighbour is not None and not math.isfinite(neighbour):
            raise BoundaryRefinementError(f"{name} timestamp must be finite")
        if neighbour is not None and not 0 <= neighbour <= media_duration_seconds:
            raise BoundaryRefinementError(
                f"{name} timestamp is outside media duration"
            )
    if (
        previous_approved_timestamp is not None
        and previous_approved_timestamp >= original
    ):
        raise BoundaryRefinementError("previous boundary must precede this boundary")
    if next_approved_timestamp is not None and next_approved_timestamp <= original:
        raise BoundaryRefinementError("next boundary must follow this boundary")

    media = Path(media_path)
    if not media.is_file():
        raise BoundaryRefinementError(f"media file does not exist: {media}")
    runner = runner or subprocess.run

    half_frame = config.frame_duration_seconds / 2
    extraction_start = max(
        0.0, original - config.maximum_shift_seconds - half_frame
    )
    extraction_end = min(
        media_duration_seconds,
        original + config.maximum_shift_seconds + half_frame,
    )
    samples = _extract_pcm(
        media,
        extraction_start,
        extraction_end - extraction_start,
        ffmpeg_path=ffmpeg_path,
        sample_rate_hz=config.sample_rate_hz,
        runner=runner,
    )
    frame_samples = max(1, round(config.frame_duration_seconds * config.sample_rate_hz))
    hop_samples = max(1, round(config.hop_duration_seconds * config.sample_rate_hz))
    if len(samples) < frame_samples:
        raise BoundaryRefinementError(
            "FFmpeg returned insufficient audio for energy analysis"
        )
    original_center = round((original - extraction_start) * config.sample_rate_hz)
    original_start = max(0, original_center - frame_samples // 2)
    original_rms = _rms(samples[original_start : original_start + frame_samples])

    candidates: list[tuple[float, float]] = []
    for frame_start in range(0, max(0, len(samples) - frame_samples + 1), hop_samples):
        center_sample = frame_start + frame_samples / 2
        timestamp = extraction_start + center_sample / config.sample_rate_hz
        adjustment = timestamp - original
        if abs(adjustment) > config.maximum_shift_seconds + 1e-9:
            continue
        candidates.append((timestamp, _rms(samples[frame_start : frame_start + frame_samples])))

    low_energy = [
        item
        for item in candidates
        if item[1] <= config.silence_rms_threshold
        and item[1] <= original_rms * config.minimum_energy_ratio
    ]
    lower_bound = 0.0
    upper_bound = media_duration_seconds
    if previous_approved_timestamp is not None:
        lower_bound = previous_approved_timestamp + config.minimum_neighbour_gap_seconds
    if next_approved_timestamp is not None:
        upper_bound = next_approved_timestamp - config.minimum_neighbour_gap_seconds
    valid = [item for item in low_energy if lower_bound < item[0] < upper_bound]
    lowest = min(low_energy, key=lambda item: (item[1], abs(item[0] - original), item[0])) if low_energy else None

    evidence = AudioEnergyEvidence(
        extraction_start_seconds=extraction_start,
        extraction_end_seconds=extraction_end,
        sample_rate_hz=config.sample_rate_hz,
        frame_duration_seconds=config.frame_duration_seconds,
        hop_duration_seconds=config.hop_duration_seconds,
        silence_rms_threshold=config.silence_rms_threshold,
        original_rms=original_rms,
        lowest_candidate_rms=lowest[1] if lowest else None,
        lowest_candidate_timestamp=lowest[0] if lowest else None,
        low_energy_candidate_count=len(low_energy),
        valid_candidate_count=len(valid),
    )
    if original_rms <= config.silence_rms_threshold:
        return BoundaryRefinementResult(
            original_selected_timestamp=original,
            refined_timestamp=original,
            adjustment_milliseconds=0.0,
            maximum_shift_seconds=config.maximum_shift_seconds,
            refinement_applied=False,
            evidence=evidence,
            reason=RefinementReason.ORIGINAL_ALREADY_LOW_ENERGY,
            validation_status=RefinementValidationStatus.VALID,
        )
    if not valid:
        reason = (
            RefinementReason.NEIGHBOUR_PROTECTION
            if low_energy
            else RefinementReason.NO_LOW_ENERGY_CANDIDATE
        )
        return BoundaryRefinementResult(
            original_selected_timestamp=original,
            refined_timestamp=original,
            adjustment_milliseconds=0.0,
            maximum_shift_seconds=config.maximum_shift_seconds,
            refinement_applied=False,
            evidence=evidence,
            reason=reason,
            validation_status=RefinementValidationStatus.VALID,
        )

    chosen = min(valid, key=lambda item: (abs(item[0] - original), item[1], item[0]))
    adjustment_ms = (chosen[0] - original) * 1000
    return BoundaryRefinementResult(
        original_selected_timestamp=original,
        refined_timestamp=chosen[0],
        adjustment_milliseconds=round(adjustment_ms, 6),
        maximum_shift_seconds=config.maximum_shift_seconds,
        refinement_applied=True,
        evidence=evidence,
        reason=RefinementReason.REFINED_TO_LOW_ENERGY,
        validation_status=RefinementValidationStatus.VALID,
    )

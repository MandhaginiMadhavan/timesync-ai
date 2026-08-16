"""Validated human boundary decisions and original-media review previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping, Sequence

from .video_cutter import CutVerificationError, FFmpegExecutionError, probe_media


class HumanReviewSource(str, Enum):
    """Auditable source selected by a human reviewer."""

    METADATA = "human_metadata"
    WHISPER = "human_whisper"
    MANUAL = "human_manual"


@dataclass(frozen=True, slots=True)
class BoundaryContext:
    """Small caption context centred on the exact boundary word."""

    before: str
    target_word: str
    after: str


@dataclass(frozen=True, slots=True)
class HumanReviewDecision:
    """Human provenance layered over, never replacing, AI provenance."""

    boundary_id: str
    metadata_timestamp_seconds: float | None
    whisper_timestamp_seconds: float | None
    resolver_selected_source: str
    resolver_selected_timestamp_seconds: float | None
    critic_status: str
    human_selected_source: HumanReviewSource
    human_selected_timestamp_seconds: float
    final_execution_timestamp_seconds: float
    reviewer_note: str | None
    reviewed_at: str


@dataclass(frozen=True, slots=True)
class ReviewPreview:
    """Original-media evidence centred on an absolute candidate timestamp."""

    candidate_timestamp_seconds: float
    preview_start_seconds: float
    preview_end_seconds: float
    candidate_offset_seconds: float
    output_path: Path
    actual_duration_seconds: float


ReviewCommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def build_boundary_context(
    caption_text: str, target_word: str, *, surrounding_words: int = 6
) -> BoundaryContext:
    """Locate the aligned boundary word and return concise surrounding text."""
    if surrounding_words < 0:
        raise ValueError("surrounding_words must be non-negative")
    words = _WORD_PATTERN.findall(caption_text)
    normalized_target = "".join(_WORD_PATTERN.findall(target_word)).casefold()
    if not words or not normalized_target:
        raise ValueError("caption and target word must be non-empty")
    target_index = next(
        (
            index
            for index, word in enumerate(words)
            if word.casefold().replace("’", "'")
            == normalized_target.replace("’", "'")
        ),
        None,
    )
    if target_index is None:
        raise ValueError(f"target word {target_word!r} is not present in caption")
    start = max(0, target_index - surrounding_words)
    stop = min(len(words), target_index + surrounding_words + 1)
    return BoundaryContext(
        before=" ".join(words[start:target_index]),
        target_word=words[target_index],
        after=" ".join(words[target_index + 1 : stop]),
    )


def _numeric_timestamp(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} timestamp is unavailable")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("selected timestamp must be finite and non-negative")
    return timestamp


def validate_human_review_decision(
    boundary: Mapping[str, object],
    source: HumanReviewSource,
    *,
    manual_timestamp_seconds: float | None = None,
    media_duration_seconds: float | None = None,
    previous_accepted_timestamp_seconds: float | None = None,
    next_accepted_timestamp_seconds: float | None = None,
    minimum_gap_seconds: float = 0.01,
    reviewer_note: str | None = None,
    reviewed_at: datetime | None = None,
) -> HumanReviewDecision:
    """Validate an absolute original-video timestamp and preserve provenance."""
    critic = boundary.get("critic")
    resolver = boundary.get("resolver")
    if not isinstance(critic, Mapping) or critic.get("status") != "human_review":
        raise ValueError("only HUMAN_REVIEW boundaries accept a human decision")
    if not isinstance(resolver, Mapping):
        raise ValueError("Resolver provenance is missing")
    boundary_id = boundary.get("boundary_id")
    if not isinstance(boundary_id, str) or not boundary_id:
        raise ValueError("boundary ID is missing")
    if not math.isfinite(minimum_gap_seconds) or minimum_gap_seconds < 0:
        raise ValueError("minimum gap must be finite and non-negative")

    metadata = boundary.get("metadata_timestamp_seconds")
    whisper = boundary.get("stt_timestamp_seconds")
    if source is HumanReviewSource.METADATA:
        selected = _numeric_timestamp(metadata, "metadata")
    elif source is HumanReviewSource.WHISPER:
        selected = _numeric_timestamp(whisper, "Whisper")
    else:
        selected = _numeric_timestamp(manual_timestamp_seconds, "manual")

    if media_duration_seconds is not None:
        if not math.isfinite(media_duration_seconds) or media_duration_seconds <= 0:
            raise ValueError("media duration must be finite and positive")
        if selected > media_duration_seconds:
            raise ValueError("selected timestamp exceeds media duration")
    if previous_accepted_timestamp_seconds is not None:
        previous = _numeric_timestamp(previous_accepted_timestamp_seconds, "previous")
        if selected <= previous + minimum_gap_seconds:
            raise ValueError("selected timestamp must remain after the previous boundary")
    if next_accepted_timestamp_seconds is not None:
        following = _numeric_timestamp(next_accepted_timestamp_seconds, "next")
        if selected >= following - minimum_gap_seconds:
            raise ValueError("selected timestamp must remain before the next boundary")

    timestamp_now = reviewed_at or datetime.now(timezone.utc)
    if timestamp_now.tzinfo is None:
        raise ValueError("review timestamp must be timezone-aware")
    resolver_source = resolver.get("selected_source")
    if not isinstance(resolver_source, str):
        raise ValueError("Resolver selected source is missing")
    resolver_timestamp = resolver.get("semantic_timestamp_seconds")
    if resolver_timestamp is not None:
        resolver_timestamp = _numeric_timestamp(resolver_timestamp, "Resolver")
    return HumanReviewDecision(
        boundary_id=boundary_id,
        metadata_timestamp_seconds=float(metadata) if isinstance(metadata, (int, float)) else None,
        whisper_timestamp_seconds=float(whisper) if isinstance(whisper, (int, float)) else None,
        resolver_selected_source=resolver_source,
        resolver_selected_timestamp_seconds=resolver_timestamp,
        critic_status="human_review",
        human_selected_source=source,
        human_selected_timestamp_seconds=selected,
        final_execution_timestamp_seconds=selected,
        reviewer_note=reviewer_note.strip() if reviewer_note else None,
        reviewed_at=timestamp_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def neighbouring_accepted_timestamps(
    boundaries: Sequence[Mapping[str, object]],
    boundary_index: int,
    human_decisions: Mapping[str, HumanReviewDecision],
) -> tuple[float | None, float | None]:
    """Find nearest accepted absolute timestamps around one review boundary."""
    if not 0 <= boundary_index < len(boundaries):
        raise IndexError("boundary index is outside the report")

    def accepted(index: int) -> float | None:
        item = boundaries[index]
        boundary_id = item.get("boundary_id")
        if isinstance(boundary_id, str) and boundary_id in human_decisions:
            return human_decisions[boundary_id].final_execution_timestamp_seconds
        critic = item.get("critic")
        resolver = item.get("resolver")
        if isinstance(critic, Mapping) and critic.get("status") == "approved" and isinstance(resolver, Mapping):
            value = resolver.get("semantic_timestamp_seconds")
            return float(value) if isinstance(value, (int, float)) else None
        return None

    previous = next(
        (value for index in range(boundary_index - 1, -1, -1) if (value := accepted(index)) is not None),
        None,
    )
    following = next(
        (value for index in range(boundary_index + 1, len(boundaries)) if (value := accepted(index)) is not None),
        None,
    )
    return previous, following


def create_review_preview(
    media_path: str | Path,
    output_path: str | Path,
    candidate_timestamp_seconds: float,
    *,
    allowed_output_directory: str | Path,
    ffmpeg_path: str | Path = "ffmpeg",
    ffprobe_path: str | Path = "ffprobe",
    context_before_seconds: float = 2.0,
    context_after_seconds: float = 2.0,
    runner: ReviewCommandRunner | None = None,
) -> ReviewPreview:
    """Create verified context media from the original video around a boundary."""
    runner = runner or subprocess.run
    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"original media does not exist: {source}")
    candidate = _numeric_timestamp(candidate_timestamp_seconds, "candidate")
    for label, value in (("context before", context_before_seconds), ("context after", context_after_seconds)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")
    media = probe_media(source, ffprobe_path=ffprobe_path, runner=runner)
    if candidate > media.duration_seconds:
        raise ValueError("candidate timestamp exceeds media duration")

    destination = Path(output_path).resolve()
    allowed = Path(allowed_output_directory).resolve()
    if not destination.is_relative_to(allowed):
        raise ValueError(f"preview output must be inside {allowed}")
    if destination == source.resolve() or destination.suffix.casefold() != ".mp4":
        raise ValueError("preview must be a separate .mp4 output")
    start = max(0.0, candidate - context_before_seconds)
    end = min(media.duration_seconds, candidate + context_after_seconds)
    duration = end - start
    if duration <= 0:
        raise ValueError("preview window has no duration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = (
        str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source.resolve()),
        "-ss", f"{start:.9f}", "-t", f"{duration:.9f}", "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(destination),
    )
    completed = runner(list(command), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no error output").strip()
        raise FFmpegExecutionError(f"review preview FFmpeg failed: {detail}")
    if not destination.is_file():
        raise CutVerificationError("FFmpeg did not create the review preview")
    preview_probe = probe_media(destination, ffprobe_path=ffprobe_path, runner=runner)
    if not preview_probe.has_video_stream or not preview_probe.has_audio_stream:
        raise CutVerificationError("review preview must contain original video and audio")
    if abs(preview_probe.duration_seconds - duration) > 0.3:
        raise CutVerificationError("review preview duration failed verification")
    return ReviewPreview(candidate, start, end, candidate - start, destination, preview_probe.duration_seconds)


def write_human_review_ledger(decisions: Sequence[HumanReviewDecision], path: str | Path) -> None:
    """Persist human provenance as deterministic UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(decisions, key=lambda item: item.boundary_id)
    destination.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "decisions": [
                    {**asdict(item), "human_selected_source": item.human_selected_source.value}
                    for item in ordered
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

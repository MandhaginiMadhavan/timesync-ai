"""Typed timestamp domain models for TimeSync AI."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_seconds(
    value: float | None, field_name: str, *, allow_none: bool
) -> None:
    """Validate a timestamp, optionally allowing an explicitly missing value."""
    if value is None:
        if allow_none:
            return
        raise TypeError(f"{field_name} must be a number")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{field_name} must be a number{suffix}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite, non-negative value")


@dataclass(frozen=True, slots=True)
class Boundary:
    """A word or media-cut boundary measured from the start of the media."""

    seconds: float
    word: str | None = None

    def __post_init__(self) -> None:
        _validate_seconds(self.seconds, "seconds", allow_none=False)


@dataclass(frozen=True, slots=True)
class MetadataTimestamp:
    """A timestamp supplied by source metadata, or ``None`` when unavailable."""

    seconds: float | None

    def __post_init__(self) -> None:
        _validate_seconds(self.seconds, "seconds", allow_none=True)


@dataclass(frozen=True, slots=True)
class STTTimestamp:
    """A timestamp inferred by speech-to-text, or ``None`` when unavailable."""

    seconds: float | None

    def __post_init__(self) -> None:
        _validate_seconds(self.seconds, "seconds", allow_none=True)


@dataclass(frozen=True, slots=True)
class STTConfidence:
    """Speech-to-text confidence expressed as an inclusive 0-to-1 score."""

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError("confidence must be a number")
        if not math.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("confidence must be between 0 and 1 inclusive")


@dataclass(frozen=True, slots=True)
class TimestampConflict:
    """The measured disagreement between metadata and STT timestamps."""

    metadata_timestamp: MetadataTimestamp
    stt_timestamp: STTTimestamp
    disagreement_seconds: float | None
    threshold_seconds: float

    def __post_init__(self) -> None:
        _validate_seconds(
            self.disagreement_seconds, "disagreement_seconds", allow_none=True
        )
        _validate_seconds(self.threshold_seconds, "threshold_seconds", allow_none=False)

    @property
    def is_major(self) -> bool:
        """Return whether disagreement strictly exceeds the configured threshold."""
        return (
            self.disagreement_seconds is not None
            and self.disagreement_seconds > self.threshold_seconds
        )

    @property
    def is_comparable(self) -> bool:
        """Return whether both inputs were available for comparison."""
        return self.disagreement_seconds is not None

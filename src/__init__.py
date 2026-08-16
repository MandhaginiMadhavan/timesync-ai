"""Foundational timestamp models and conflict detection for TimeSync AI."""

from .conflicts import (
    DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS,
    detect_timestamp_conflict,
)
from .models import (
    Boundary,
    MetadataTimestamp,
    STTConfidence,
    STTTimestamp,
    TimestampConflict,
)

__all__ = [
    "Boundary",
    "DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS",
    "MetadataTimestamp",
    "STTConfidence",
    "STTTimestamp",
    "TimestampConflict",
    "detect_timestamp_conflict",
]

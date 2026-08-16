"""Timestamp conflict detection."""

from __future__ import annotations

import math

from .models import MetadataTimestamp, STTTimestamp, TimestampConflict


DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS = 0.5


def detect_timestamp_conflict(
    metadata_timestamp: MetadataTimestamp,
    stt_timestamp: STTTimestamp,
    *,
    threshold_seconds: float = DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS,
) -> TimestampConflict:
    """Compare timestamps and classify disagreements above the threshold as major.

    A missing timestamp cannot be compared, so its disagreement is ``None`` and
    it is not classified as a major conflict.
    """
    if (
        not isinstance(threshold_seconds, (int, float))
        or isinstance(threshold_seconds, bool)
    ):
        raise TypeError("threshold_seconds must be a number")
    if not math.isfinite(threshold_seconds) or threshold_seconds < 0:
        raise ValueError("threshold_seconds must be finite and non-negative")

    metadata_seconds = metadata_timestamp.seconds
    stt_seconds = stt_timestamp.seconds
    if metadata_seconds is None or stt_seconds is None:
        disagreement = None
    else:
        disagreement = abs(metadata_seconds - stt_seconds)

    return TimestampConflict(
        metadata_timestamp=metadata_timestamp,
        stt_timestamp=stt_timestamp,
        disagreement_seconds=disagreement,
        threshold_seconds=float(threshold_seconds),
    )

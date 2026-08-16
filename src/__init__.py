"""Foundational timestamp models and conflict detection for TimeSync AI."""

from .alignment import (
    AlignedBoundaryCandidate,
    align_caption_boundaries,
    resolver_evidence_for,
)
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
from .resolver import (
    NeighbourTimestampPair,
    ReasonCode,
    ResolverConfig,
    ResolverDecision,
    ResolverDiagnostics,
    ResolverEvidence,
    TimestampSource,
    resolve_timestamp,
    text_alignment_quality,
)

__all__ = [
    "AlignedBoundaryCandidate",
    "Boundary",
    "DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS",
    "MetadataTimestamp",
    "NeighbourTimestampPair",
    "ReasonCode",
    "ResolverConfig",
    "ResolverDecision",
    "ResolverDiagnostics",
    "ResolverEvidence",
    "STTConfidence",
    "STTTimestamp",
    "TimestampConflict",
    "TimestampSource",
    "align_caption_boundaries",
    "detect_timestamp_conflict",
    "resolve_timestamp",
    "resolver_evidence_for",
    "text_alignment_quality",
]

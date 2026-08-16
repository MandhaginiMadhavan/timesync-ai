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
from .critic import (
    CriticConfig,
    CriticDiagnostics,
    CriticReasonCode,
    CriticResult,
    CriticStatus,
    RiskContribution,
    critique_decisions,
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
    "CriticConfig",
    "CriticDiagnostics",
    "CriticReasonCode",
    "CriticResult",
    "CriticStatus",
    "DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS",
    "MetadataTimestamp",
    "NeighbourTimestampPair",
    "ReasonCode",
    "ResolverConfig",
    "ResolverDecision",
    "ResolverDiagnostics",
    "ResolverEvidence",
    "RiskContribution",
    "STTConfidence",
    "STTTimestamp",
    "TimestampConflict",
    "TimestampSource",
    "align_caption_boundaries",
    "critique_decisions",
    "detect_timestamp_conflict",
    "resolve_timestamp",
    "resolver_evidence_for",
    "text_alignment_quality",
]

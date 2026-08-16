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
from .video_cutter import (
    BoundaryNotApprovedError,
    CutConfig,
    CutResult,
    CutStatus,
    CutValidationError,
    CutVerificationError,
    FFmpegExecutionError,
    MediaVerification,
    VideoCutError,
    cut_video,
)

__all__ = [
    "AlignedBoundaryCandidate",
    "Boundary",
    "BoundaryNotApprovedError",
    "CriticConfig",
    "CriticDiagnostics",
    "CriticReasonCode",
    "CriticResult",
    "CriticStatus",
    "CutConfig",
    "CutResult",
    "CutStatus",
    "CutValidationError",
    "CutVerificationError",
    "DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS",
    "MetadataTimestamp",
    "NeighbourTimestampPair",
    "ReasonCode",
    "ResolverConfig",
    "ResolverDecision",
    "ResolverDiagnostics",
    "ResolverEvidence",
    "RiskContribution",
    "FFmpegExecutionError",
    "MediaVerification",
    "STTConfidence",
    "STTTimestamp",
    "TimestampConflict",
    "TimestampSource",
    "VideoCutError",
    "align_caption_boundaries",
    "critique_decisions",
    "cut_video",
    "detect_timestamp_conflict",
    "resolve_timestamp",
    "resolver_evidence_for",
    "text_alignment_quality",
]

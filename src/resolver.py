"""Deterministic, explainable timestamp resolution.

The resolver ranks metadata and STT evidence. It does not modify media, infer
new timestamps, or hide ambiguity: weak or tied evidence produces an
``unresolved`` decision for a later critic or human reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import math
import re
from statistics import median

from .conflicts import DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS, detect_timestamp_conflict
from .models import MetadataTimestamp, STTConfidence, STTTimestamp


class TimestampSource(str, Enum):
    """A source selected for a resolved timestamp."""

    METADATA = "metadata"
    STT = "stt"
    UNRESOLVED = "unresolved"


class ReasonCode(str, Enum):
    """Stable, machine-readable reasons contributing to a decision."""

    WITHIN_TOLERANCE = "within_tolerance"
    METADATA_MISSING = "metadata_missing"
    STT_MISSING = "stt_missing"
    BOTH_TIMESTAMPS_MISSING = "both_timestamps_missing"
    HIGH_STT_CONFIDENCE = "high_stt_confidence"
    LOW_STT_CONFIDENCE = "low_stt_confidence"
    STT_CONFIDENCE_MISSING = "stt_confidence_missing"
    STRONG_TEXT_ALIGNMENT = "strong_text_alignment"
    WEAK_TEXT_ALIGNMENT = "weak_text_alignment"
    TEXT_ALIGNMENT_MISSING = "text_alignment_missing"
    NEIGHBOUR_CONSISTENT = "neighbour_consistent"
    TEMPORAL_OUTLIER = "temporal_outlier"
    SYSTEMATIC_METADATA_DRIFT = "systematic_metadata_drift"
    INSUFFICIENT_NEIGHBOURS = "insufficient_neighbours"
    LARGE_DISAGREEMENT = "large_disagreement"
    EVIDENCE_AMBIGUOUS = "evidence_ambiguous"


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    """Explicit thresholds and weights used by the deterministic resolver."""

    major_conflict_threshold_seconds: float = (
        DEFAULT_MAJOR_CONFLICT_THRESHOLD_SECONDS
    )
    minimum_decision_margin: float = 0.15
    high_stt_confidence: float = 0.8
    low_stt_confidence: float = 0.45
    strong_alignment: float = 0.75
    weak_alignment: float = 0.35
    neighbour_consistency_seconds: float = 0.2
    temporal_outlier_seconds: float = 0.6
    systematic_drift_seconds: float = 0.5
    maximum_drift_spread_seconds: float = 0.15
    minimum_neighbours_for_drift: int = 3
    large_disagreement_seconds: float = 2.0
    base_source_score: float = 0.5
    high_confidence_stt_bonus: float = 0.15
    low_confidence_stt_penalty: float = 0.2
    low_confidence_metadata_bonus: float = 0.1
    strong_alignment_stt_bonus: float = 0.25
    weak_alignment_stt_penalty: float = 0.2
    weak_alignment_metadata_bonus: float = 0.1
    neighbour_consistency_stt_bonus: float = 0.15
    temporal_outlier_stt_penalty: float = 0.4
    temporal_outlier_metadata_bonus: float = 0.25
    systematic_drift_stt_bonus: float = 0.2
    large_disagreement_score_penalty: float = 0.1

    def __post_init__(self) -> None:
        unit_values = {
            "minimum_decision_margin": self.minimum_decision_margin,
            "high_stt_confidence": self.high_stt_confidence,
            "low_stt_confidence": self.low_stt_confidence,
            "strong_alignment": self.strong_alignment,
            "weak_alignment": self.weak_alignment,
            "base_source_score": self.base_source_score,
        }
        for name, value in unit_values.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        second_values = {
            "major_conflict_threshold_seconds": self.major_conflict_threshold_seconds,
            "neighbour_consistency_seconds": self.neighbour_consistency_seconds,
            "temporal_outlier_seconds": self.temporal_outlier_seconds,
            "systematic_drift_seconds": self.systematic_drift_seconds,
            "maximum_drift_spread_seconds": self.maximum_drift_spread_seconds,
            "large_disagreement_seconds": self.large_disagreement_seconds,
        }
        for name, value in second_values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        weight_values = {
            "high_confidence_stt_bonus": self.high_confidence_stt_bonus,
            "low_confidence_stt_penalty": self.low_confidence_stt_penalty,
            "low_confidence_metadata_bonus": self.low_confidence_metadata_bonus,
            "strong_alignment_stt_bonus": self.strong_alignment_stt_bonus,
            "weak_alignment_stt_penalty": self.weak_alignment_stt_penalty,
            "weak_alignment_metadata_bonus": self.weak_alignment_metadata_bonus,
            "neighbour_consistency_stt_bonus": self.neighbour_consistency_stt_bonus,
            "temporal_outlier_stt_penalty": self.temporal_outlier_stt_penalty,
            "temporal_outlier_metadata_bonus": self.temporal_outlier_metadata_bonus,
            "systematic_drift_stt_bonus": self.systematic_drift_stt_bonus,
            "large_disagreement_score_penalty": self.large_disagreement_score_penalty,
        }
        for name, value in weight_values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.low_stt_confidence > self.high_stt_confidence:
            raise ValueError("low_stt_confidence cannot exceed high_stt_confidence")
        if self.weak_alignment > self.strong_alignment:
            raise ValueError("weak_alignment cannot exceed strong_alignment")
        if self.neighbour_consistency_seconds > self.temporal_outlier_seconds:
            raise ValueError(
                "neighbour consistency tolerance cannot exceed outlier tolerance"
            )
        if self.minimum_neighbours_for_drift < 1:
            raise ValueError("minimum_neighbours_for_drift must be at least 1")


@dataclass(frozen=True, slots=True)
class NeighbourTimestampPair:
    """A nearby aligned metadata/STT pair used only as temporal context."""

    metadata_timestamp: MetadataTimestamp
    stt_timestamp: STTTimestamp

    @property
    def offset_seconds(self) -> float | None:
        """Return signed STT-minus-metadata offset when both values exist."""
        metadata = self.metadata_timestamp.seconds
        stt = self.stt_timestamp.seconds
        return None if metadata is None or stt is None else stt - metadata


@dataclass(frozen=True, slots=True)
class ResolverEvidence:
    """Optional independent evidence associated with one disputed boundary."""

    stt_confidence: STTConfidence | None = None
    caption_text: str | None = None
    stt_text: str | None = None
    alignment_quality: float | None = None
    neighbours: tuple[NeighbourTimestampPair, ...] = ()

    def __post_init__(self) -> None:
        if self.alignment_quality is not None and (
            not math.isfinite(self.alignment_quality)
            or not 0 <= self.alignment_quality <= 1
        ):
            raise ValueError("alignment_quality must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ResolverDiagnostics:
    """Intermediate values exposed for a future independent Critic Agent."""

    metadata_score: float
    stt_score: float
    decision_margin: float
    stt_confidence: float | None
    alignment_quality: float | None
    neighbour_offsets_seconds: tuple[float, ...]
    expected_offset_seconds: float | None
    offset_spread_seconds: float | None
    current_offset_seconds: float | None
    current_offset_residual_seconds: float | None
    config: ResolverConfig


@dataclass(frozen=True, slots=True)
class ResolverDecision:
    """An auditable resolver result for one candidate boundary."""

    metadata_timestamp: float | None
    stt_timestamp: float | None
    disagreement_seconds: float | None
    selected_source: TimestampSource
    selected_timestamp: float | None
    resolver_confidence: float
    reason_codes: tuple[ReasonCode, ...]
    explanation: str
    diagnostics: ResolverDiagnostics


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def text_alignment_quality(caption_text: str, stt_text: str) -> float:
    """Return a deterministic 0-to-1 token-sequence similarity score."""
    caption_tokens = _TOKEN_PATTERN.findall(caption_text.casefold())
    stt_tokens = _TOKEN_PATTERN.findall(stt_text.casefold())
    if not caption_tokens or not stt_tokens:
        return 0.0
    return SequenceMatcher(None, caption_tokens, stt_tokens, autojunk=False).ratio()


def _clamp_score(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def _alignment_from(evidence: ResolverEvidence) -> float | None:
    if evidence.alignment_quality is not None:
        return evidence.alignment_quality
    if evidence.caption_text is None or evidence.stt_text is None:
        return None
    return text_alignment_quality(evidence.caption_text, evidence.stt_text)


def _explain(
    source: TimestampSource,
    reasons: list[ReasonCode],
    disagreement: float | None,
) -> str:
    reason_text = {
        ReasonCode.WITHIN_TOLERANCE: "the timestamps agree within tolerance",
        ReasonCode.METADATA_MISSING: "the metadata timestamp is unavailable",
        ReasonCode.STT_MISSING: "the STT timestamp is unavailable",
        ReasonCode.BOTH_TIMESTAMPS_MISSING: "both timestamps are unavailable",
        ReasonCode.HIGH_STT_CONFIDENCE: "word recognition confidence is high",
        ReasonCode.LOW_STT_CONFIDENCE: "word recognition confidence is low",
        ReasonCode.STT_CONFIDENCE_MISSING: "word confidence is unavailable",
        ReasonCode.STRONG_TEXT_ALIGNMENT: "caption and STT text align strongly",
        ReasonCode.WEAK_TEXT_ALIGNMENT: "caption and STT text align weakly",
        ReasonCode.TEXT_ALIGNMENT_MISSING: "text alignment evidence is unavailable",
        ReasonCode.NEIGHBOUR_CONSISTENT: "the offset matches neighbouring boundaries",
        ReasonCode.TEMPORAL_OUTLIER: "the STT offset is a temporal outlier",
        ReasonCode.SYSTEMATIC_METADATA_DRIFT: "neighbours show consistent metadata drift",
        ReasonCode.INSUFFICIENT_NEIGHBOURS: "neighbour evidence is insufficient",
        ReasonCode.LARGE_DISAGREEMENT: "the timestamp disagreement is unusually large",
        ReasonCode.EVIDENCE_AMBIGUOUS: "the weighted evidence is too close to call",
    }
    prefix = {
        TimestampSource.METADATA: "Selected metadata",
        TimestampSource.STT: "Selected STT",
        TimestampSource.UNRESOLVED: "Left unresolved",
    }[source]
    detail = "; ".join(reason_text[reason] for reason in reasons)
    disagreement_text = (
        "without a measurable disagreement"
        if disagreement is None
        else f"with {disagreement:.3f}s disagreement"
    )
    return f"{prefix} {disagreement_text}: {detail}."


def resolve_timestamp(
    metadata_timestamp: MetadataTimestamp,
    stt_timestamp: STTTimestamp,
    evidence: ResolverEvidence | None = None,
    *,
    config: ResolverConfig | None = None,
) -> ResolverDecision:
    """Resolve one boundary using explicit, reproducible evidence weights."""
    evidence = evidence or ResolverEvidence()
    config = config or ResolverConfig()
    conflict = detect_timestamp_conflict(
        metadata_timestamp,
        stt_timestamp,
        threshold_seconds=config.major_conflict_threshold_seconds,
    )
    metadata = metadata_timestamp.seconds
    stt = stt_timestamp.seconds
    alignment = _alignment_from(evidence)
    confidence = evidence.stt_confidence.value if evidence.stt_confidence else None
    offsets = tuple(
        offset
        for pair in evidence.neighbours
        if (offset := pair.offset_seconds) is not None
    )
    current_offset = None if metadata is None or stt is None else stt - metadata
    expected_offset = float(median(offsets)) if offsets else None
    spread = (
        float(median(abs(offset - expected_offset) for offset in offsets))
        if expected_offset is not None
        else None
    )
    residual = (
        abs(current_offset - expected_offset)
        if current_offset is not None and expected_offset is not None
        else None
    )

    metadata_score = config.base_source_score if metadata is not None else 0.0
    stt_score = config.base_source_score if stt is not None else 0.0
    reasons: list[ReasonCode] = []

    def decision(
        source: TimestampSource,
        selected: float | None,
        resolver_confidence: float,
    ) -> ResolverDecision:
        diagnostics = ResolverDiagnostics(
            metadata_score=_clamp_score(metadata_score),
            stt_score=_clamp_score(stt_score),
            decision_margin=round(abs(metadata_score - stt_score), 6),
            stt_confidence=confidence,
            alignment_quality=alignment,
            neighbour_offsets_seconds=offsets,
            expected_offset_seconds=expected_offset,
            offset_spread_seconds=spread,
            current_offset_seconds=current_offset,
            current_offset_residual_seconds=residual,
            config=config,
        )
        return ResolverDecision(
            metadata_timestamp=metadata,
            stt_timestamp=stt,
            disagreement_seconds=conflict.disagreement_seconds,
            selected_source=source,
            selected_timestamp=selected,
            resolver_confidence=_clamp_score(resolver_confidence),
            reason_codes=tuple(reasons),
            explanation=_explain(source, reasons, conflict.disagreement_seconds),
            diagnostics=diagnostics,
        )

    if metadata is None and stt is None:
        reasons.append(ReasonCode.BOTH_TIMESTAMPS_MISSING)
        return decision(TimestampSource.UNRESOLVED, None, 0.0)
    if stt is None:
        reasons.append(ReasonCode.STT_MISSING)
        return decision(TimestampSource.METADATA, metadata, 0.75)
    if metadata is None:
        reasons.append(ReasonCode.METADATA_MISSING)
        if confidence is None and alignment is None:
            reasons.extend(
                [
                    ReasonCode.STT_CONFIDENCE_MISSING,
                    ReasonCode.TEXT_ALIGNMENT_MISSING,
                    ReasonCode.EVIDENCE_AMBIGUOUS,
                ]
            )
            return decision(TimestampSource.UNRESOLVED, None, 0.0)
        if confidence is not None and confidence < config.low_stt_confidence:
            reasons.extend([ReasonCode.LOW_STT_CONFIDENCE, ReasonCode.EVIDENCE_AMBIGUOUS])
            return decision(TimestampSource.UNRESOLVED, None, 0.1)
        return decision(TimestampSource.STT, stt, 0.6)

    if not conflict.is_major:
        reasons.append(ReasonCode.WITHIN_TOLERANCE)
        agreement_confidence = 1.0 - (
            (conflict.disagreement_seconds or 0.0)
            / (2 * config.major_conflict_threshold_seconds)
            if config.major_conflict_threshold_seconds > 0
            else 0.0
        )
        return decision(TimestampSource.METADATA, metadata, agreement_confidence)

    if confidence is None:
        reasons.append(ReasonCode.STT_CONFIDENCE_MISSING)
    elif confidence >= config.high_stt_confidence:
        stt_score += config.high_confidence_stt_bonus
        reasons.append(ReasonCode.HIGH_STT_CONFIDENCE)
    elif confidence < config.low_stt_confidence:
        stt_score -= config.low_confidence_stt_penalty
        metadata_score += config.low_confidence_metadata_bonus
        reasons.append(ReasonCode.LOW_STT_CONFIDENCE)

    if alignment is None:
        reasons.append(ReasonCode.TEXT_ALIGNMENT_MISSING)
    elif alignment >= config.strong_alignment:
        stt_score += config.strong_alignment_stt_bonus
        reasons.append(ReasonCode.STRONG_TEXT_ALIGNMENT)
    elif alignment < config.weak_alignment:
        stt_score -= config.weak_alignment_stt_penalty
        metadata_score += config.weak_alignment_metadata_bonus
        reasons.append(ReasonCode.WEAK_TEXT_ALIGNMENT)

    if residual is None or len(offsets) < 2:
        reasons.append(ReasonCode.INSUFFICIENT_NEIGHBOURS)
    elif residual <= config.neighbour_consistency_seconds:
        stt_score += config.neighbour_consistency_stt_bonus
        reasons.append(ReasonCode.NEIGHBOUR_CONSISTENT)
    elif residual >= config.temporal_outlier_seconds:
        stt_score -= config.temporal_outlier_stt_penalty
        metadata_score += config.temporal_outlier_metadata_bonus
        reasons.append(ReasonCode.TEMPORAL_OUTLIER)

    if (
        expected_offset is not None
        and spread is not None
        and residual is not None
        and len(offsets) >= config.minimum_neighbours_for_drift
        and abs(expected_offset) >= config.systematic_drift_seconds
        and spread <= config.maximum_drift_spread_seconds
        and residual <= config.neighbour_consistency_seconds
    ):
        stt_score += config.systematic_drift_stt_bonus
        reasons.append(ReasonCode.SYSTEMATIC_METADATA_DRIFT)

    if (
        conflict.disagreement_seconds is not None
        and conflict.disagreement_seconds >= config.large_disagreement_seconds
    ):
        metadata_score -= config.large_disagreement_score_penalty
        stt_score -= config.large_disagreement_score_penalty
        reasons.append(ReasonCode.LARGE_DISAGREEMENT)

    metadata_score = _clamp_score(metadata_score)
    stt_score = _clamp_score(stt_score)
    margin = abs(metadata_score - stt_score)
    if margin < config.minimum_decision_margin:
        reasons.append(ReasonCode.EVIDENCE_AMBIGUOUS)
        return decision(TimestampSource.UNRESOLVED, None, margin)

    selected_source = (
        TimestampSource.STT if stt_score > metadata_score else TimestampSource.METADATA
    )
    selected_timestamp = stt if selected_source is TimestampSource.STT else metadata
    return decision(
        selected_source,
        selected_timestamp,
        0.5 + margin / 2,
    )

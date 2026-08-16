"""Independent invariant and risk checks for resolver decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import median
from typing import Sequence

from .resolver import ReasonCode, ResolverDecision, TimestampSource


class CriticStatus(str, Enum):
    """Outcome of an independent boundary review."""

    APPROVED = "approved"
    HUMAN_REVIEW = "human_review"


class CriticReasonCode(str, Enum):
    """Stable reason codes emitted by Critic checks."""

    VALID_BOUNDARY = "valid_boundary"
    EVIDENCE_CONSISTENT = "evidence_consistent"
    UNRESOLVED_DECISION = "unresolved_decision"
    SELECTED_TIMESTAMP_MISSING = "selected_timestamp_missing"
    SELECTED_TIMESTAMP_MISMATCH = "selected_timestamp_mismatch"
    NON_MONOTONIC_BOUNDARY = "non_monotonic_boundary"
    SUSPICIOUS_LOCAL_JUMP = "suspicious_local_jump"
    SELECTED_SOURCE_UNSUPPORTED = "selected_source_unsupported"
    CONTRADICTORY_REASONS = "contradictory_reasons"
    WEAK_DECISION_MARGIN = "weak_decision_margin"
    LOW_STT_PROBABILITY = "low_stt_probability"
    STT_PROBABILITY_MISSING = "stt_probability_missing"
    POOR_TEXT_ALIGNMENT = "poor_text_alignment"
    TEXT_ALIGNMENT_MISSING = "text_alignment_missing"
    LARGE_SOURCE_DISAGREEMENT = "large_source_disagreement"
    INCONSISTENT_NEIGHBOUR_OFFSET = "inconsistent_neighbour_offset"
    STT_CONFIDENCE_NOT_TIMING_PROOF = "stt_confidence_not_timing_proof"


@dataclass(frozen=True, slots=True)
class CriticConfig:
    """Independent Critic thresholds and risk contributions."""

    human_review_risk_threshold: float = 0.5
    low_stt_probability: float = 0.45
    poor_alignment: float = 0.5
    large_disagreement_seconds: float = 1.0
    maximum_offset_residual_seconds: float = 0.5
    weak_decision_margin: float = 0.15
    minimum_boundary_gap_seconds: float = 0.05
    suspicious_jump_multiplier: float = 2.0
    suspicious_jump_minimum_excess_seconds: float = 3.0
    unresolved_risk: float = 1.0
    invariant_failure_risk: float = 1.0
    source_inconsistency_risk: float = 0.7
    contradiction_risk: float = 0.6
    temporal_inconsistency_risk: float = 0.6
    low_stt_probability_risk: float = 0.5
    poor_alignment_risk: float = 0.5
    missing_stt_evidence_risk: float = 0.3
    large_disagreement_risk: float = 0.2
    weak_margin_risk: float = 0.35

    def __post_init__(self) -> None:
        unit_values = {
            "human_review_risk_threshold": self.human_review_risk_threshold,
            "low_stt_probability": self.low_stt_probability,
            "poor_alignment": self.poor_alignment,
            "weak_decision_margin": self.weak_decision_margin,
            "unresolved_risk": self.unresolved_risk,
            "invariant_failure_risk": self.invariant_failure_risk,
            "source_inconsistency_risk": self.source_inconsistency_risk,
            "contradiction_risk": self.contradiction_risk,
            "temporal_inconsistency_risk": self.temporal_inconsistency_risk,
            "low_stt_probability_risk": self.low_stt_probability_risk,
            "poor_alignment_risk": self.poor_alignment_risk,
            "missing_stt_evidence_risk": self.missing_stt_evidence_risk,
            "large_disagreement_risk": self.large_disagreement_risk,
            "weak_margin_risk": self.weak_margin_risk,
        }
        for name, value in unit_values.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        positive_values = {
            "large_disagreement_seconds": self.large_disagreement_seconds,
            "maximum_offset_residual_seconds": self.maximum_offset_residual_seconds,
            "minimum_boundary_gap_seconds": self.minimum_boundary_gap_seconds,
            "suspicious_jump_multiplier": self.suspicious_jump_multiplier,
            "suspicious_jump_minimum_excess_seconds": (
                self.suspicious_jump_minimum_excess_seconds
            ),
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.suspicious_jump_multiplier < 1:
            raise ValueError("suspicious_jump_multiplier must be at least 1")


@dataclass(frozen=True, slots=True)
class RiskContribution:
    """One auditable contribution to the Critic risk score."""

    reason_code: CriticReasonCode
    risk: float
    evidence: str


@dataclass(frozen=True, slots=True)
class CriticDiagnostics:
    """Evidence used by the Critic, independent of resolver recomputation."""

    previous_selected_timestamp: float | None
    next_selected_timestamp: float | None
    previous_gap_seconds: float | None
    next_gap_seconds: float | None
    typical_gap_seconds: float | None
    selected_source_score: float | None
    alternative_source_score: float | None
    resolver_decision_margin: float
    stt_probability: float | None
    alignment_quality: float | None
    disagreement_seconds: float | None
    offset_residual_seconds: float | None
    contributions: tuple[RiskContribution, ...]
    config: CriticConfig


@dataclass(frozen=True, slots=True)
class CriticResult:
    """Structured independent review of one resolver decision."""

    resolver_decision: ResolverDecision
    status: CriticStatus
    risk_score: float
    reason_codes: tuple[CriticReasonCode, ...]
    explanation: str
    diagnostics: CriticDiagnostics


def _reason_contradictions(reasons: set[ReasonCode]) -> bool:
    contradictory_pairs = (
        (ReasonCode.HIGH_STT_CONFIDENCE, ReasonCode.LOW_STT_CONFIDENCE),
        (ReasonCode.STRONG_TEXT_ALIGNMENT, ReasonCode.WEAK_TEXT_ALIGNMENT),
        (ReasonCode.STT_MISSING, ReasonCode.HIGH_STT_CONFIDENCE),
        (ReasonCode.STT_MISSING, ReasonCode.LOW_STT_CONFIDENCE),
        (ReasonCode.TEXT_ALIGNMENT_MISSING, ReasonCode.STRONG_TEXT_ALIGNMENT),
        (ReasonCode.TEXT_ALIGNMENT_MISSING, ReasonCode.WEAK_TEXT_ALIGNMENT),
    )
    return any(left in reasons and right in reasons for left, right in contradictory_pairs)


def _typical_gap(decisions: Sequence[ResolverDecision]) -> float | None:
    gaps = [
        current.selected_timestamp - previous.selected_timestamp
        for previous, current in zip(decisions, decisions[1:])
        if previous.selected_timestamp is not None
        and current.selected_timestamp is not None
        and current.selected_timestamp > previous.selected_timestamp
    ]
    return float(median(gaps)) if gaps else None


def _explanation(
    status: CriticStatus, contributions: list[RiskContribution]
) -> str:
    prefix = (
        "Approved boundary"
        if status is CriticStatus.APPROVED
        else "Send this boundary to human review"
    )
    if not contributions:
        return f"{prefix}: temporal and recorded-evidence checks passed."
    return f"{prefix}: " + "; ".join(item.evidence for item in contributions) + "."


def critique_decisions(
    decisions: Sequence[ResolverDecision],
    *,
    config: CriticConfig | None = None,
) -> list[CriticResult]:
    """Review an ordered boundary sequence without rerunning resolver scoring."""
    config = config or CriticConfig()
    typical_gap = _typical_gap(decisions)
    results: list[CriticResult] = []

    for index, decision in enumerate(decisions):
        previous_timestamp = (
            decisions[index - 1].selected_timestamp if index > 0 else None
        )
        next_timestamp = (
            decisions[index + 1].selected_timestamp
            if index + 1 < len(decisions)
            else None
        )
        selected = decision.selected_timestamp
        previous_gap = (
            selected - previous_timestamp
            if selected is not None and previous_timestamp is not None
            else None
        )
        next_gap = (
            next_timestamp - selected
            if selected is not None and next_timestamp is not None
            else None
        )
        resolver_diagnostics = decision.diagnostics
        stt_probability = resolver_diagnostics.stt_confidence
        alignment = resolver_diagnostics.alignment_quality
        residual = resolver_diagnostics.current_offset_residual_seconds
        reasons = set(decision.reason_codes)
        contributions: list[RiskContribution] = []
        hard_review = False

        def add(code: CriticReasonCode, risk: float, evidence: str) -> None:
            contributions.append(RiskContribution(code, risk, evidence))

        if decision.selected_source is TimestampSource.UNRESOLVED:
            hard_review = True
            add(
                CriticReasonCode.UNRESOLVED_DECISION,
                config.unresolved_risk,
                "the resolver explicitly left this boundary unresolved",
            )
        elif selected is None:
            hard_review = True
            add(
                CriticReasonCode.SELECTED_TIMESTAMP_MISSING,
                config.invariant_failure_risk,
                "a resolved source has no selected timestamp",
            )
        else:
            expected_selected = (
                decision.metadata_timestamp
                if decision.selected_source is TimestampSource.METADATA
                else decision.stt_timestamp
            )
            if expected_selected is None or not math.isclose(
                selected, expected_selected, rel_tol=0.0, abs_tol=1e-9
            ):
                hard_review = True
                add(
                    CriticReasonCode.SELECTED_TIMESTAMP_MISMATCH,
                    config.invariant_failure_risk,
                    "the selected timestamp does not match the selected source",
                )

        if selected is not None:
            if previous_gap is not None and previous_gap <= config.minimum_boundary_gap_seconds:
                hard_review = True
                add(
                    CriticReasonCode.NON_MONOTONIC_BOUNDARY,
                    config.invariant_failure_risk,
                    f"the boundary is only {previous_gap:.3f}s after its predecessor",
                )
            for label, gap in (("previous", previous_gap), ("next", next_gap)):
                if (
                    gap is not None
                    and typical_gap is not None
                    and gap > typical_gap * config.suspicious_jump_multiplier
                    and gap - typical_gap
                    >= config.suspicious_jump_minimum_excess_seconds
                ):
                    add(
                        CriticReasonCode.SUSPICIOUS_LOCAL_JUMP,
                        config.temporal_inconsistency_risk,
                        f"the {label} gap of {gap:.3f}s is abnormal versus the "
                        f"{typical_gap:.3f}s typical gap",
                    )

        metadata_score = resolver_diagnostics.metadata_score
        stt_score = resolver_diagnostics.stt_score
        selected_score: float | None
        alternative_score: float | None
        if decision.selected_source is TimestampSource.STT:
            selected_score, alternative_score = stt_score, metadata_score
        elif decision.selected_source is TimestampSource.METADATA:
            selected_score, alternative_score = metadata_score, stt_score
        else:
            selected_score = alternative_score = None

        within_tolerance = ReasonCode.WITHIN_TOLERANCE in reasons
        if (
            selected_score is not None
            and alternative_score is not None
            and selected_score <= alternative_score
            and not within_tolerance
        ):
            hard_review = True
            add(
                CriticReasonCode.SELECTED_SOURCE_UNSUPPORTED,
                config.source_inconsistency_risk,
                "the recorded source scores do not support the selected source",
            )

        source_reason_conflict = (
            decision.selected_source is TimestampSource.STT
            and ReasonCode.TEMPORAL_OUTLIER in reasons
        ) or (
            decision.selected_source is TimestampSource.METADATA
            and ReasonCode.SYSTEMATIC_METADATA_DRIFT in reasons
        ) or (
            decision.selected_source is not TimestampSource.UNRESOLVED
            and ReasonCode.EVIDENCE_AMBIGUOUS in reasons
        )
        if _reason_contradictions(reasons) or source_reason_conflict:
            hard_review = True
            add(
                CriticReasonCode.CONTRADICTORY_REASONS,
                config.contradiction_risk,
                "recorded reason codes contradict each other or the selected source",
            )

        if (
            decision.selected_source is not TimestampSource.UNRESOLVED
            and not within_tolerance
            and resolver_diagnostics.decision_margin < config.weak_decision_margin
        ):
            add(
                CriticReasonCode.WEAK_DECISION_MARGIN,
                config.weak_margin_risk,
                f"the recorded source-score margin is only "
                f"{resolver_diagnostics.decision_margin:.3f}",
            )

        stt_evidence_is_relevant = (
            decision.selected_source is TimestampSource.STT
            or (
                decision.selected_source is TimestampSource.UNRESOLVED
                and decision.stt_timestamp is not None
            )
        )
        if stt_evidence_is_relevant:
            stt_label = (
                "selected STT"
                if decision.selected_source is TimestampSource.STT
                else "STT candidate"
            )
            if stt_probability is None:
                add(
                    CriticReasonCode.STT_PROBABILITY_MISSING,
                    config.missing_stt_evidence_risk,
                    f"the {stt_label} has no word probability evidence",
                )
            elif stt_probability < config.low_stt_probability:
                add(
                    CriticReasonCode.LOW_STT_PROBABILITY,
                    config.low_stt_probability_risk,
                    f"the {stt_label} word probability is only {stt_probability:.3f}",
                )

            if alignment is None:
                add(
                    CriticReasonCode.TEXT_ALIGNMENT_MISSING,
                    config.missing_stt_evidence_risk,
                    f"the {stt_label} has no text-alignment evidence",
                )
            elif alignment < config.poor_alignment:
                add(
                    CriticReasonCode.POOR_TEXT_ALIGNMENT,
                    config.poor_alignment_risk,
                    f"the {stt_label} text alignment is only {alignment:.3f}",
                )

            if (
                decision.selected_source is TimestampSource.STT
                and residual is not None
                and residual >= config.maximum_offset_residual_seconds
            ):
                add(
                    CriticReasonCode.INCONSISTENT_NEIGHBOUR_OFFSET,
                    config.temporal_inconsistency_risk,
                    f"the selected STT offset differs from local expectation by "
                    f"{residual:.3f}s",
                )
                if stt_probability is not None and stt_probability >= 0.8:
                    add(
                        CriticReasonCode.STT_CONFIDENCE_NOT_TIMING_PROOF,
                        0.0,
                        "high recognition probability does not validate this timing outlier",
                    )

        if (
            decision.disagreement_seconds is not None
            and decision.disagreement_seconds >= config.large_disagreement_seconds
        ):
            add(
                CriticReasonCode.LARGE_SOURCE_DISAGREEMENT,
                config.large_disagreement_risk,
                f"source timestamps disagree by {decision.disagreement_seconds:.3f}s",
            )

        risk_score = min(1.0, sum(item.risk for item in contributions))
        status = (
            CriticStatus.HUMAN_REVIEW
            if hard_review or risk_score >= config.human_review_risk_threshold
            else CriticStatus.APPROVED
        )
        if status is CriticStatus.APPROVED:
            approval_codes = (
                CriticReasonCode.VALID_BOUNDARY,
                CriticReasonCode.EVIDENCE_CONSISTENT,
            )
            reason_codes = approval_codes + tuple(
                item.reason_code for item in contributions
            )
        else:
            reason_codes = tuple(item.reason_code for item in contributions)

        diagnostics = CriticDiagnostics(
            previous_selected_timestamp=previous_timestamp,
            next_selected_timestamp=next_timestamp,
            previous_gap_seconds=previous_gap,
            next_gap_seconds=next_gap,
            typical_gap_seconds=typical_gap,
            selected_source_score=selected_score,
            alternative_source_score=alternative_score,
            resolver_decision_margin=resolver_diagnostics.decision_margin,
            stt_probability=stt_probability,
            alignment_quality=alignment,
            disagreement_seconds=decision.disagreement_seconds,
            offset_residual_seconds=residual,
            contributions=tuple(contributions),
            config=config,
        )
        results.append(
            CriticResult(
                resolver_decision=decision,
                status=status,
                risk_score=round(risk_score, 6),
                reason_codes=reason_codes,
                explanation=_explanation(status, contributions),
                diagnostics=diagnostics,
            )
        )

    return results

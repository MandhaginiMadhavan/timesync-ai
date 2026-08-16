"""Tests for independent Critic checks."""

from dataclasses import replace

import pytest

from src import MetadataTimestamp, STTConfidence, STTTimestamp
from src.critic import CriticReasonCode, CriticStatus, critique_decisions
from src.resolver import (
    NeighbourTimestampPair,
    ReasonCode,
    ResolverEvidence,
    TimestampSource,
    resolve_timestamp,
)


def pair(metadata: float, stt: float) -> NeighbourTimestampPair:
    return NeighbourTimestampPair(MetadataTimestamp(metadata), STTTimestamp(stt))


def stt_decision(
    metadata: float,
    stt: float,
    *,
    probability: float = 0.95,
    alignment: float = 0.95,
    neighbours: tuple[NeighbourTimestampPair, ...] = (),
):
    return resolve_timestamp(
        MetadataTimestamp(metadata),
        STTTimestamp(stt),
        ResolverEvidence(
            stt_confidence=STTConfidence(probability),
            alignment_quality=alignment,
            neighbours=neighbours,
        ),
    )


def test_clearly_valid_resolver_decision_is_approved() -> None:
    decisions = [
        stt_decision(10.0, 10.7),
        stt_decision(20.0, 20.7),
        stt_decision(30.0, 30.7),
    ]

    result = critique_decisions(decisions)[1]

    assert result.status is CriticStatus.APPROVED
    assert result.risk_score == 0.0
    assert CriticReasonCode.VALID_BOUNDARY in result.reason_codes


def test_high_confidence_stt_temporal_inconsistency_requires_review() -> None:
    decision = stt_decision(
        20.0,
        22.0,
        neighbours=(pair(18.0, 18.1), pair(19.0, 19.1), pair(21.0, 21.1)),
    )
    # Isolate the Critic check from whether the Resolver itself selected STT.
    forced_stt = replace(
        decision,
        selected_source=TimestampSource.STT,
        selected_timestamp=22.0,
    )

    result = critique_decisions([forced_stt])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.INCONSISTENT_NEIGHBOUR_OFFSET in result.reason_codes
    assert CriticReasonCode.STT_CONFIDENCE_NOT_TIMING_PROOF in result.reason_codes


def test_low_stt_probability_and_large_disagreement_requires_review() -> None:
    decision = stt_decision(10.0, 11.2, probability=0.2)
    forced_stt = replace(
        decision,
        selected_source=TimestampSource.STT,
        selected_timestamp=11.2,
    )

    result = critique_decisions([forced_stt])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.LOW_STT_PROBABILITY in result.reason_codes
    assert CriticReasonCode.LARGE_SOURCE_DISAGREEMENT in result.reason_codes


def test_unresolved_decision_requires_review() -> None:
    decision = resolve_timestamp(
        MetadataTimestamp(10.0),
        STTTimestamp(10.8),
        ResolverEvidence(stt_confidence=STTConfidence(0.6), alignment_quality=0.6),
    )

    result = critique_decisions([decision])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert result.reason_codes == (CriticReasonCode.UNRESOLVED_DECISION,)


def test_unresolved_decision_preserves_recognition_risk() -> None:
    decision = resolve_timestamp(
        MetadataTimestamp(10.0),
        STTTimestamp(10.8),
        ResolverEvidence(stt_confidence=STTConfidence(0.2), alignment_quality=0.9),
    )

    result = critique_decisions([decision])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.UNRESOLVED_DECISION in result.reason_codes
    assert CriticReasonCode.LOW_STT_PROBABILITY in result.reason_codes


def test_non_monotonic_boundary_requires_only_that_boundary_review() -> None:
    decisions = [
        stt_decision(10.0, 10.7),
        stt_decision(20.0, 9.7),
        stt_decision(30.0, 30.7),
    ]

    results = critique_decisions(decisions)

    assert results[0].status is CriticStatus.APPROVED
    assert results[1].status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.NON_MONOTONIC_BOUNDARY in results[1].reason_codes
    assert len(results) == 3


def test_poor_alignment_requires_review_when_stt_selected() -> None:
    decision = stt_decision(10.0, 10.8, alignment=0.2)
    forced_stt = replace(
        decision,
        selected_source=TimestampSource.STT,
        selected_timestamp=10.8,
    )

    result = critique_decisions([forced_stt])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.POOR_TEXT_ALIGNMENT in result.reason_codes


def test_safe_metadata_decision_is_approved() -> None:
    decision = resolve_timestamp(MetadataTimestamp(10.0), STTTimestamp(10.3))

    result = critique_decisions([decision])[0]

    assert result.status is CriticStatus.APPROVED
    assert result.resolver_decision.selected_source is TimestampSource.METADATA


def test_missing_stt_evidence_accumulates_to_review() -> None:
    decision = resolve_timestamp(MetadataTimestamp(None), STTTimestamp(10.0))
    forced_stt = replace(
        decision,
        selected_source=TimestampSource.STT,
        selected_timestamp=10.0,
    )

    result = critique_decisions([forced_stt])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.STT_PROBABILITY_MISSING in result.reason_codes
    assert CriticReasonCode.TEXT_ALIGNMENT_MISSING in result.reason_codes


def test_selected_timestamp_source_mismatch_requires_review() -> None:
    decision = resolve_timestamp(MetadataTimestamp(10.0), STTTimestamp(10.3))
    invalid = replace(decision, selected_timestamp=10.3)

    result = critique_decisions([invalid])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.SELECTED_TIMESTAMP_MISMATCH in result.reason_codes


def test_contradictory_reason_codes_require_review() -> None:
    decision = stt_decision(10.0, 10.8)
    invalid = replace(
        decision,
        reason_codes=decision.reason_codes + (ReasonCode.LOW_STT_CONFIDENCE,),
    )

    result = critique_decisions([invalid])[0]

    assert result.status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.CONTRADICTORY_REASONS in result.reason_codes


def test_suspicious_local_jump_requires_review() -> None:
    decisions = [
        stt_decision(0.0, 0.8),
        stt_decision(10.0, 10.8),
        stt_decision(50.0, 50.8),
        stt_decision(60.0, 60.8),
    ]

    results = critique_decisions(decisions)

    assert results[1].status is CriticStatus.HUMAN_REVIEW
    assert CriticReasonCode.SUSPICIOUS_LOCAL_JUMP in results[1].reason_codes


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_review_threshold_is_rejected(value: float) -> None:
    from src.critic import CriticConfig

    with pytest.raises(ValueError):
        CriticConfig(human_review_risk_threshold=value)


def test_invalid_jump_multiplier_is_rejected() -> None:
    from src.critic import CriticConfig

    with pytest.raises(ValueError):
        CriticConfig(suspicious_jump_multiplier=0.9)

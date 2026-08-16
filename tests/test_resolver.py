"""Tests for deterministic, explainable timestamp resolution."""

import pytest

from src import MetadataTimestamp, STTConfidence, STTTimestamp
from src.resolver import (
    NeighbourTimestampPair,
    ReasonCode,
    ResolverConfig,
    ResolverEvidence,
    TimestampSource,
    resolve_timestamp,
    text_alignment_quality,
)


def pair(metadata: float, stt: float) -> NeighbourTimestampPair:
    return NeighbourTimestampPair(MetadataTimestamp(metadata), STTTimestamp(stt))


def test_strong_stt_evidence_selects_stt() -> None:
    result = resolve_timestamp(
        MetadataTimestamp(10.0),
        STTTimestamp(10.8),
        ResolverEvidence(
            stt_confidence=STTConfidence(0.95),
            caption_text="keep your head",
            stt_text="keep your head",
        ),
    )

    assert result.selected_source is TimestampSource.STT
    assert result.selected_timestamp == 10.8
    assert result.disagreement_seconds == pytest.approx(0.8)
    assert ReasonCode.HIGH_STT_CONFIDENCE in result.reason_codes
    assert ReasonCode.STRONG_TEXT_ALIGNMENT in result.reason_codes


def test_low_confidence_and_weak_alignment_select_metadata() -> None:
    result = resolve_timestamp(
        MetadataTimestamp(20.0),
        STTTimestamp(20.8),
        ResolverEvidence(
            stt_confidence=STTConfidence(0.2),
            caption_text="keep your head",
            stt_text="unrelated words here",
        ),
    )

    assert result.selected_source is TimestampSource.METADATA
    assert result.selected_timestamp == 20.0
    assert ReasonCode.LOW_STT_CONFIDENCE in result.reason_codes
    assert ReasonCode.WEAK_TEXT_ALIGNMENT in result.reason_codes


def test_high_word_probability_does_not_override_temporal_outlier() -> None:
    result = resolve_timestamp(
        MetadataTimestamp(30.0),
        STTTimestamp(32.0),
        ResolverEvidence(
            stt_confidence=STTConfidence(0.99),
            caption_text="hold on",
            stt_text="hold on",
            neighbours=(pair(29.0, 29.1), pair(31.0, 31.1), pair(33.0, 33.1)),
        ),
    )

    assert result.selected_source is TimestampSource.METADATA
    assert ReasonCode.HIGH_STT_CONFIDENCE in result.reason_codes
    assert ReasonCode.TEMPORAL_OUTLIER in result.reason_codes


def test_systematic_metadata_drift_selects_consistent_stt() -> None:
    result = resolve_timestamp(
        MetadataTimestamp(40.0),
        STTTimestamp(40.8),
        ResolverEvidence(
            stt_confidence=STTConfidence(0.6),
            alignment_quality=0.6,
            neighbours=(pair(35.0, 35.8), pair(38.0, 38.82), pair(42.0, 42.79)),
        ),
    )

    assert result.selected_source is TimestampSource.STT
    assert ReasonCode.NEIGHBOUR_CONSISTENT in result.reason_codes
    assert ReasonCode.SYSTEMATIC_METADATA_DRIFT in result.reason_codes
    assert result.diagnostics.expected_offset_seconds == pytest.approx(0.8)


def test_missing_stt_uses_available_metadata() -> None:
    result = resolve_timestamp(MetadataTimestamp(12.0), STTTimestamp(None))

    assert result.selected_source is TimestampSource.METADATA
    assert result.selected_timestamp == 12.0
    assert result.reason_codes == (ReasonCode.STT_MISSING,)


def test_missing_metadata_and_unsupported_stt_is_unresolved() -> None:
    result = resolve_timestamp(MetadataTimestamp(None), STTTimestamp(12.0))

    assert result.selected_source is TimestampSource.UNRESOLVED
    assert result.selected_timestamp is None
    assert result.resolver_confidence == 0.0


def test_both_missing_is_unresolved() -> None:
    result = resolve_timestamp(MetadataTimestamp(None), STTTimestamp(None))

    assert result.selected_source is TimestampSource.UNRESOLVED
    assert ReasonCode.BOTH_TIMESTAMPS_MISSING in result.reason_codes


def test_tied_evidence_is_unresolved() -> None:
    result = resolve_timestamp(
        MetadataTimestamp(10.0),
        STTTimestamp(10.8),
        ResolverEvidence(stt_confidence=STTConfidence(0.6), alignment_quality=0.6),
    )

    assert result.selected_source is TimestampSource.UNRESOLVED
    assert result.selected_timestamp is None
    assert ReasonCode.EVIDENCE_AMBIGUOUS in result.reason_codes


def test_non_major_conflict_preserves_metadata_boundary() -> None:
    result = resolve_timestamp(MetadataTimestamp(10.0), STTTimestamp(10.5))

    assert result.selected_source is TimestampSource.METADATA
    assert ReasonCode.WITHIN_TOLERANCE in result.reason_codes


def test_custom_conflict_and_decision_thresholds_are_used() -> None:
    config = ResolverConfig(
        major_conflict_threshold_seconds=0.2,
        minimum_decision_margin=0.1,
    )
    result = resolve_timestamp(
        MetadataTimestamp(10.0),
        STTTimestamp(10.3),
        ResolverEvidence(stt_confidence=STTConfidence(0.9)),
        config=config,
    )

    assert result.selected_source is TimestampSource.STT


def test_alignment_is_deterministic_and_token_based() -> None:
    assert text_alignment_quality("Hold on!", "hold on") == 1.0
    assert text_alignment_quality("", "hold on") == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_decision_margin": -0.1},
        {"high_stt_confidence": 1.1},
        {"minimum_neighbours_for_drift": 0},
        {"neighbour_consistency_seconds": 1.0, "temporal_outlier_seconds": 0.5},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        ResolverConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("quality", [-0.1, 1.1, float("nan")])
def test_invalid_alignment_quality_is_rejected(quality: float) -> None:
    with pytest.raises(ValueError):
        ResolverEvidence(alignment_quality=quality)


def test_resolution_is_reproducible() -> None:
    evidence = ResolverEvidence(
        stt_confidence=STTConfidence(0.9),
        alignment_quality=0.8,
        neighbours=(pair(8.0, 8.7), pair(12.0, 12.7), pair(14.0, 14.7)),
    )

    first = resolve_timestamp(MetadataTimestamp(10.0), STTTimestamp(10.7), evidence)
    second = resolve_timestamp(MetadataTimestamp(10.0), STTTimestamp(10.7), evidence)

    assert first == second
    assert first.explanation
    assert first.diagnostics.config == ResolverConfig()
    assert first.diagnostics.neighbour_offsets_seconds == pytest.approx((0.7, 0.7, 0.7))


def test_score_weights_are_configurable() -> None:
    config = ResolverConfig(high_confidence_stt_bonus=0.0)
    result = resolve_timestamp(
        MetadataTimestamp(10.0),
        STTTimestamp(10.8),
        ResolverEvidence(stt_confidence=STTConfidence(0.95)),
        config=config,
    )

    assert result.selected_source is TimestampSource.UNRESOLVED
    assert result.diagnostics.config.high_confidence_stt_bonus == 0.0

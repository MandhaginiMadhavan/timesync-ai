"""Tests for timestamp disagreement classification."""

import pytest

from src import MetadataTimestamp, STTTimestamp, detect_timestamp_conflict


@pytest.mark.parametrize(
    ("difference", "expected_major"),
    [
        (0.4, False),
        (0.5, False),
        (0.51, True),
        (0.8, True),
    ],
)
def test_major_conflict_boundary(difference: float, expected_major: bool) -> None:
    conflict = detect_timestamp_conflict(
        MetadataTimestamp(10.0),
        STTTimestamp(10.0 + difference),
    )

    assert conflict.disagreement_seconds == pytest.approx(difference)
    assert conflict.is_major is expected_major


def test_threshold_is_configurable() -> None:
    conflict = detect_timestamp_conflict(
        MetadataTimestamp(10.0),
        STTTimestamp(10.4),
        threshold_seconds=0.3,
    )

    assert conflict.is_major is True
    assert conflict.threshold_seconds == 0.3


@pytest.mark.parametrize(
    ("metadata", "stt"),
    [
        (MetadataTimestamp(None), STTTimestamp(1.0)),
        (MetadataTimestamp(1.0), STTTimestamp(None)),
        (MetadataTimestamp(None), STTTimestamp(None)),
    ],
)
def test_missing_timestamp_is_not_comparable(
    metadata: MetadataTimestamp, stt: STTTimestamp
) -> None:
    conflict = detect_timestamp_conflict(metadata, stt)

    assert conflict.disagreement_seconds is None
    assert conflict.is_comparable is False
    assert conflict.is_major is False


@pytest.mark.parametrize("threshold", [-0.1, float("inf"), float("nan")])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        detect_timestamp_conflict(
            MetadataTimestamp(1.0),
            STTTimestamp(1.0),
            threshold_seconds=threshold,
        )

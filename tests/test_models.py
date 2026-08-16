"""Tests for foundational timestamp models."""

import pytest

from src import Boundary, MetadataTimestamp, STTConfidence, STTTimestamp


@pytest.mark.parametrize(
    "model",
    [Boundary, MetadataTimestamp, STTTimestamp],
)
def test_negative_timestamp_is_rejected(model: type[object]) -> None:
    with pytest.raises(ValueError):
        model(-0.01)  # type: ignore[call-arg]


def test_boundary_requires_a_timestamp() -> None:
    with pytest.raises(TypeError):
        Boundary(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("inf"), float("nan")])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        STTConfidence(confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_valid_confidence_is_accepted(confidence: float) -> None:
    assert STTConfidence(confidence).value == confidence

"""Tests for conservative low-energy boundary refinement."""

from array import array
from dataclasses import replace
from pathlib import Path
import subprocess
from typing import Callable

import pytest

from src import MetadataTimestamp, STTConfidence, STTTimestamp
from src.boundary_refinement import (
    BoundaryRefinementError,
    RefinementConfig,
    RefinementReason,
    RefinementValidationStatus,
    refine_boundary,
)
from src.critic import CriticResult, CriticStatus, critique_decisions
from src.resolver import ResolverEvidence, resolve_timestamp


def approved_boundary(timestamp: float) -> CriticResult:
    decision = resolve_timestamp(
        MetadataTimestamp(timestamp), STTTimestamp(timestamp)
    )
    return critique_decisions([decision])[0]


def human_review_boundary(timestamp: float) -> CriticResult:
    decision = resolve_timestamp(
        MetadataTimestamp(timestamp),
        STTTimestamp(timestamp + 0.8),
        ResolverEvidence(stt_confidence=STTConfidence(0.6), alignment_quality=0.6),
    )
    result = critique_decisions([decision])[0]
    assert result.status is CriticStatus.HUMAN_REVIEW
    return result


def analysis_config() -> RefinementConfig:
    return RefinementConfig(sample_rate_hz=1_000)


def pcm_runner(samples: list[int]) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    payload = array("h", samples).tobytes()

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, payload, b"")

    return run


def media_file(tmp_path: Path) -> Path:
    path = tmp_path / "media.mp4"
    path.write_bytes(b"media")
    return path


def samples_with_silence(
    *, length: int = 320, silence_start: int = 230, silence_end: int = 250
) -> list[int]:
    samples = [10_000] * length
    samples[silence_start:silence_end] = [0] * (silence_end - silence_start)
    return samples


def test_nearby_silence_produces_small_refinement(tmp_path: Path) -> None:
    result = refine_boundary(
        media_file(tmp_path),
        approved_boundary(1.0),
        2.0,
        config=analysis_config(),
        runner=pcm_runner(samples_with_silence()),
    )

    assert result.refinement_applied is True
    assert result.refined_timestamp == pytest.approx(1.08)
    assert result.adjustment_milliseconds == pytest.approx(80.0)
    assert abs(result.refined_timestamp - result.original_selected_timestamp) <= 0.15
    assert result.reason is RefinementReason.REFINED_TO_LOW_ENERGY


def test_no_nearby_silence_keeps_original_timestamp(tmp_path: Path) -> None:
    result = refine_boundary(
        media_file(tmp_path),
        approved_boundary(1.0),
        2.0,
        config=analysis_config(),
        runner=pcm_runner([10_000] * 320),
    )

    assert result.refinement_applied is False
    assert result.refined_timestamp == 1.0
    assert result.reason is RefinementReason.NO_LOW_ENERGY_CANDIDATE


def test_silence_beyond_maximum_shift_is_rejected(tmp_path: Path) -> None:
    result = refine_boundary(
        media_file(tmp_path),
        approved_boundary(1.0),
        2.0,
        config=analysis_config(),
        runner=pcm_runner(
            samples_with_silence(length=400, silence_start=330, silence_end=350)
        ),
    )

    assert result.refinement_applied is False
    assert result.refined_timestamp == 1.0


def test_human_review_boundary_is_not_analysed(tmp_path: Path) -> None:
    result = refine_boundary(
        media_file(tmp_path),
        human_review_boundary(1.0),
        2.0,
        config=analysis_config(),
        runner=lambda *_args, **_kwargs: pytest.fail("FFmpeg must not execute"),
    )

    assert result.validation_status is RefinementValidationStatus.NOT_APPROVED
    assert result.reason is RefinementReason.HUMAN_REVIEW_BOUNDARY
    assert result.evidence is None


def test_neighbour_protection_rejects_crossing_candidate(tmp_path: Path) -> None:
    result = refine_boundary(
        media_file(tmp_path),
        approved_boundary(1.0),
        2.0,
        next_approved_timestamp=1.085,
        config=analysis_config(),
        runner=pcm_runner(samples_with_silence()),
    )

    assert result.refinement_applied is False
    assert result.reason is RefinementReason.NEIGHBOUR_PROTECTION


@pytest.mark.parametrize(
    ("timestamp", "duration", "silence_start", "silence_end"),
    [(0.05, 2.0, 90, 110), (1.95, 2.0, 90, 110)],
)
def test_media_edge_refinement_stays_in_bounds(
    tmp_path: Path,
    timestamp: float,
    duration: float,
    silence_start: int,
    silence_end: int,
) -> None:
    result = refine_boundary(
        media_file(tmp_path),
        approved_boundary(timestamp),
        duration,
        config=analysis_config(),
        runner=pcm_runner(
            samples_with_silence(
                length=210, silence_start=silence_start, silence_end=silence_end
            )
        ),
    )

    assert result.refined_timestamp is not None
    assert 0 <= result.refined_timestamp <= duration


@pytest.mark.parametrize("duration", [-1.0, float("inf"), float("nan")])
def test_invalid_media_duration_is_rejected(tmp_path: Path, duration: float) -> None:
    with pytest.raises(BoundaryRefinementError, match="media duration"):
        refine_boundary(
            media_file(tmp_path),
            approved_boundary(1.0),
            duration,
            config=analysis_config(),
            runner=pcm_runner([0] * 320),
        )


def test_non_finite_selected_timestamp_is_rejected(tmp_path: Path) -> None:
    boundary = approved_boundary(1.0)
    boundary = replace(
        boundary,
        resolver_decision=replace(
            boundary.resolver_decision, selected_timestamp=float("nan")
        ),
    )

    with pytest.raises(BoundaryRefinementError, match="finite"):
        refine_boundary(
            media_file(tmp_path),
            boundary,
            2.0,
            config=analysis_config(),
            runner=pcm_runner([0] * 320),
        )


def test_invalid_neighbour_timestamp_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BoundaryRefinementError, match="outside media duration"):
        refine_boundary(
            media_file(tmp_path),
            approved_boundary(1.0),
            2.0,
            previous_approved_timestamp=-0.1,
            config=analysis_config(),
            runner=pcm_runner([0] * 320),
        )


def test_insufficient_audio_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BoundaryRefinementError, match="insufficient audio"):
        refine_boundary(
            media_file(tmp_path),
            approved_boundary(1.0),
            2.0,
            config=analysis_config(),
            runner=pcm_runner([]),
        )


def test_refinement_is_deterministic(tmp_path: Path) -> None:
    kwargs = {
        "media_path": media_file(tmp_path),
        "boundary": approved_boundary(1.0),
        "media_duration_seconds": 2.0,
        "config": analysis_config(),
        "runner": pcm_runner(samples_with_silence()),
    }

    assert refine_boundary(**kwargs) == refine_boundary(**kwargs)

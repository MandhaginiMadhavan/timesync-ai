"""Tests for focused human-review validation, previews, and provenance."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest

from src.human_review import (
    HumanReviewSource,
    build_boundary_context,
    create_review_preview,
    neighbouring_accepted_timestamps,
    validate_human_review_decision,
    write_human_review_ledger,
)


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def boundary(status: str = "human_review", boundary_id: str = "caption-block-9") -> dict[str, object]:
    return {
        "boundary_id": boundary_id,
        "caption_text": "turn of pitch-and-toss and lose and start again",
        "target_stt_word": "turn",
        "metadata_timestamp_seconds": 69.0,
        "stt_timestamp_seconds": 70.24,
        "critic": {"status": status},
        "resolver": {
            "selected_source": "stt",
            "semantic_timestamp_seconds": 70.24,
        },
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [(HumanReviewSource.METADATA, 69.0), (HumanReviewSource.WHISPER, 70.24)],
)
def test_source_choice_uses_real_absolute_candidate(source, expected) -> None:
    result = validate_human_review_decision(
        boundary(), source, media_duration_seconds=100.0, reviewed_at=NOW
    )
    assert result.human_selected_timestamp_seconds == expected
    assert result.final_execution_timestamp_seconds == expected
    assert result.reviewed_at == "2026-08-16T12:00:00Z"


def test_manual_timestamp_is_absolute_and_validated() -> None:
    result = validate_human_review_decision(
        boundary(), HumanReviewSource.MANUAL,
        manual_timestamp_seconds=69.75, media_duration_seconds=100.0,
        previous_accepted_timestamp_seconds=60.7,
        next_accepted_timestamp_seconds=78.0,
        reviewer_note="Checked speech onset", reviewed_at=NOW,
    )
    assert result.human_selected_timestamp_seconds == 69.75
    assert result.human_selected_source is HumanReviewSource.MANUAL
    assert result.reviewer_note == "Checked speech onset"


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), 101.0])
def test_invalid_manual_timestamp_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        validate_human_review_decision(
            boundary(), HumanReviewSource.MANUAL,
            manual_timestamp_seconds=value, media_duration_seconds=100.0,
        )


@pytest.mark.parametrize(
    ("value", "previous", "following", "message"),
    [(60.7, 60.7, 80.0, "after"), (80.0, 60.0, 80.0, "before")],
)
def test_neighbour_ordering_is_enforced(value, previous, following, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_human_review_decision(
            boundary(), HumanReviewSource.MANUAL,
            manual_timestamp_seconds=value,
            previous_accepted_timestamp_seconds=previous,
            next_accepted_timestamp_seconds=following,
        )


def test_ai_provenance_is_preserved_separately_from_human_choice() -> None:
    result = validate_human_review_decision(
        boundary(), HumanReviewSource.METADATA, reviewed_at=NOW
    )
    assert result.metadata_timestamp_seconds == 69.0
    assert result.whisper_timestamp_seconds == 70.24
    assert result.resolver_selected_source == "stt"
    assert result.resolver_selected_timestamp_seconds == 70.24
    assert result.critic_status == "human_review"
    assert result.human_selected_source is HumanReviewSource.METADATA
    assert result.final_execution_timestamp_seconds == 69.0


def test_approved_boundary_cannot_be_human_overridden() -> None:
    with pytest.raises(ValueError, match="only HUMAN_REVIEW"):
        validate_human_review_decision(boundary("approved"), HumanReviewSource.METADATA)


def test_target_boundary_context_is_small_and_exact() -> None:
    context = build_boundary_context(
        "all your winnings and risk it on one turn of pitch-and-toss and lose",
        "turn",
        surrounding_words=3,
    )
    assert context.before == "it on one"
    assert context.target_word == "turn"
    assert context.after == "of pitch-and-toss and"


def test_nearest_accepted_boundaries_include_prior_human_decisions() -> None:
    approved_before = boundary("approved", "caption-block-8")
    approved_before["resolver"] = {"semantic_timestamp_seconds": 60.7}
    first_review = boundary(boundary_id="caption-block-9")
    second_review = boundary(boundary_id="caption-block-10")
    approved_after = boundary("approved", "caption-block-11")
    approved_after["resolver"] = {"semantic_timestamp_seconds": 86.7}
    first_decision = validate_human_review_decision(
        first_review, HumanReviewSource.METADATA, reviewed_at=NOW
    )
    assert neighbouring_accepted_timestamps(
        [approved_before, first_review, second_review, approved_after],
        2,
        {"caption-block-9": first_decision},
    ) == (69.0, 86.7)


def preview_runner(source_duration: float = 100.0):
    probe_count = 0

    def run(command: list[str], **_kwargs):
        nonlocal probe_count
        if Path(command[0]).name.casefold().startswith("ffprobe"):
            probe_count += 1
            duration = source_duration if probe_count == 1 else 4.0
            payload = json.dumps({
                "format": {"duration": str(duration)},
                "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
            })
            return subprocess.CompletedProcess(command, 0, payload, "")
        Path(command[-1]).write_bytes(b"original media preview")
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def test_candidate_preview_uses_original_source_and_absolute_timestamp(tmp_path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original")
    allowed = tmp_path / "previews"
    result = create_review_preview(
        source,
        allowed / "metadata.mp4",
        69.0,
        allowed_output_directory=allowed,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        runner=preview_runner(),
    )
    assert result.candidate_timestamp_seconds == 69.0
    assert result.preview_start_seconds == 67.0
    assert result.preview_end_seconds == 71.0
    assert result.candidate_offset_seconds == 2.0
    assert result.output_path.read_bytes() == b"original media preview"


def test_ledger_is_sorted_and_retains_full_provenance(tmp_path) -> None:
    first = validate_human_review_decision(
        boundary(), HumanReviewSource.WHISPER, reviewed_at=NOW
    )
    second = validate_human_review_decision(
        boundary(boundary_id="caption-block-2"), HumanReviewSource.METADATA, reviewed_at=NOW
    )
    destination = tmp_path / "reviews.json"
    write_human_review_ledger((first, second), destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert [item["boundary_id"] for item in payload["decisions"]] == [
        "caption-block-2", "caption-block-9"
    ]
    assert payload["decisions"][1]["human_selected_source"] == "human_whisper"
    assert payload["decisions"][1]["resolver_selected_source"] == "stt"

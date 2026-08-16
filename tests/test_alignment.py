"""Tests for caption-block to Whisper-word boundary alignment."""

import pytest

from src.alignment import (
    BOUNDARY_DERIVATION,
    align_caption_boundaries,
    resolver_evidence_for,
)
from src.caption_parser import CaptionRecord
from src.whisper_transcriber import WhisperWord


def word(text: str, start: float, probability: float = 0.9) -> WhisperWord:
    return WhisperWord(text, start, start + 0.2, probability)


def test_aligns_only_real_first_word_boundaries_in_order() -> None:
    captions = [
        CaptionRecord(5.0, "keep your head"),
        CaptionRecord(8.0, "when all doubt"),
    ]
    words = [
        word("keep", 5.2),
        word("your", 5.4),
        word("head", 5.6),
        word("when", 8.3),
        word("all", 8.5),
        word("doubt", 8.7),
    ]

    aligned = align_caption_boundaries(captions, words)

    assert [item.caption_index for item in aligned] == [0, 1]
    assert aligned[0].metadata_timestamp.seconds == 5.0
    assert aligned[0].stt_timestamp.seconds == 5.2
    assert aligned[0].alignment_quality == 1.0
    assert aligned[0].derivation == BOUNDARY_DERIVATION


def test_does_not_fabricate_boundary_when_first_token_is_unmatched() -> None:
    captions = [CaptionRecord(5.0, "missing your head")]
    words = [word("your", 5.4), word("head", 5.6)]

    assert align_caption_boundaries(captions, words) == []


def test_preserves_missing_word_probability() -> None:
    captions = [CaptionRecord(5.0, "keep going")]
    words = [WhisperWord("keep", 5.2, 5.4, None), word("going", 5.4)]

    aligned = align_caption_boundaries(captions, words)

    assert aligned[0].stt_confidence is None


def test_builds_evidence_from_nearest_aligned_neighbours() -> None:
    captions = [
        CaptionRecord(5.0, "one"),
        CaptionRecord(8.0, "two"),
        CaptionRecord(11.0, "three"),
        CaptionRecord(14.0, "four"),
    ]
    words = [
        word("one", 5.1),
        word("two", 8.2),
        word("three", 11.3),
        word("four", 14.4),
    ]
    aligned = align_caption_boundaries(captions, words)

    evidence = resolver_evidence_for(aligned[2], aligned, neighbour_radius=1)

    assert len(evidence.neighbours) == 2
    assert [pair.offset_seconds for pair in evidence.neighbours] == pytest.approx(
        [0.2, 0.4]
    )


def test_rejects_candidate_outside_alignment_set() -> None:
    first = align_caption_boundaries(
        [CaptionRecord(5.0, "one")], [word("one", 5.1)]
    )
    other = align_caption_boundaries(
        [CaptionRecord(8.0, "two")], [word("two", 8.1)]
    )

    with pytest.raises(ValueError):
        resolver_evidence_for(first[0], other)

"""Honest alignment of caption-block starts to Whisper word boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Sequence

from .caption_parser import CaptionRecord
from .models import MetadataTimestamp, STTConfidence, STTTimestamp
from .resolver import NeighbourTimestampPair, ResolverEvidence, text_alignment_quality
from .whisper_transcriber import WhisperWord


BOUNDARY_DERIVATION = "exact_ordered_match_of_first_caption_token"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True, slots=True)
class AlignedBoundaryCandidate:
    """A real caption-block timestamp paired with a real Whisper word start.

    No word-level metadata timestamp is inferred. ``metadata_timestamp`` is the
    original block start, while ``stt_timestamp`` is the start of the exactly
    matched first caption token in Whisper's output.
    """

    caption_index: int
    caption_text: str
    metadata_timestamp: MetadataTimestamp
    stt_timestamp: STTTimestamp
    stt_word_index: int
    stt_word: str
    stt_confidence: STTConfidence | None
    aligned_stt_text: str
    alignment_quality: float
    matched_caption_tokens: int
    caption_token_count: int
    derivation: str = BOUNDARY_DERIVATION


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def align_caption_boundaries(
    captions: Sequence[CaptionRecord],
    whisper_words: Sequence[WhisperWord],
) -> list[AlignedBoundaryCandidate]:
    """Align block starts through a global, order-preserving exact token match.

    A candidate is emitted only if the first token of that caption block is in
    a matching block produced by the global sequence alignment. This avoids
    assigning the timestamp of a later word to an unmatched caption boundary.
    """
    caption_tokens: list[str] = []
    caption_token_locations: list[tuple[int, int]] = []
    first_token_positions: dict[int, int] = {}
    token_counts: dict[int, int] = {}
    for caption_index, caption in enumerate(captions):
        tokens = _tokens(caption.text)
        token_counts[caption_index] = len(tokens)
        if tokens:
            first_token_positions[caption_index] = len(caption_tokens)
        for local_index, token in enumerate(tokens):
            caption_tokens.append(token)
            caption_token_locations.append((caption_index, local_index))

    stt_tokens: list[str] = []
    stt_token_word_indices: list[int] = []
    for word_index, word in enumerate(whisper_words):
        for token in _tokens(word.text):
            stt_tokens.append(token)
            stt_token_word_indices.append(word_index)

    matcher = SequenceMatcher(None, caption_tokens, stt_tokens, autojunk=False)
    caption_to_stt_token: dict[int, int] = {}
    matched_counts = {index: 0 for index in range(len(captions))}
    for match in matcher.get_matching_blocks():
        for offset in range(match.size):
            caption_position = match.a + offset
            stt_position = match.b + offset
            caption_to_stt_token[caption_position] = stt_position
            caption_index, _ = caption_token_locations[caption_position]
            matched_counts[caption_index] += 1

    provisional: list[tuple[int, int]] = []
    for caption_index, first_position in first_token_positions.items():
        stt_token_position = caption_to_stt_token.get(first_position)
        if stt_token_position is None:
            continue
        provisional.append(
            (caption_index, stt_token_word_indices[stt_token_position])
        )

    candidates: list[AlignedBoundaryCandidate] = []
    for position, (caption_index, word_index) in enumerate(provisional):
        next_word_index = (
            provisional[position + 1][1]
            if position + 1 < len(provisional)
            else len(whisper_words)
        )
        aligned_words = whisper_words[word_index:next_word_index]
        aligned_text = " ".join(word.text for word in aligned_words)
        word = whisper_words[word_index]
        confidence = (
            STTConfidence(word.probability) if word.probability is not None else None
        )
        candidates.append(
            AlignedBoundaryCandidate(
                caption_index=caption_index,
                caption_text=captions[caption_index].text,
                metadata_timestamp=MetadataTimestamp(
                    captions[caption_index].start_seconds
                ),
                stt_timestamp=STTTimestamp(word.start_seconds),
                stt_word_index=word_index,
                stt_word=word.text,
                stt_confidence=confidence,
                aligned_stt_text=aligned_text,
                alignment_quality=text_alignment_quality(
                    captions[caption_index].text, aligned_text
                ),
                matched_caption_tokens=matched_counts[caption_index],
                caption_token_count=token_counts[caption_index],
            )
        )

    return candidates


def resolver_evidence_for(
    candidate: AlignedBoundaryCandidate,
    aligned_candidates: Sequence[AlignedBoundaryCandidate],
    *,
    neighbour_radius: int = 2,
) -> ResolverEvidence:
    """Build resolver evidence from nearby independently aligned boundaries."""
    if neighbour_radius < 0:
        raise ValueError("neighbour_radius must be non-negative")
    try:
        candidate_position = aligned_candidates.index(candidate)
    except ValueError as error:
        raise ValueError("candidate is not present in aligned_candidates") from error

    start = max(0, candidate_position - neighbour_radius)
    stop = min(len(aligned_candidates), candidate_position + neighbour_radius + 1)
    neighbours = tuple(
        NeighbourTimestampPair(item.metadata_timestamp, item.stt_timestamp)
        for index, item in enumerate(aligned_candidates[start:stop], start=start)
        if index != candidate_position
    )
    return ResolverEvidence(
        stt_confidence=candidate.stt_confidence,
        caption_text=candidate.caption_text,
        stt_text=candidate.aligned_stt_text,
        alignment_quality=candidate.alignment_quality,
        neighbours=neighbours,
    )

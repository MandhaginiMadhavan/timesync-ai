"""Tests for explainable audit report construction and serialization."""

from datetime import datetime, timezone
import json

import pytest

from src.alignment import AlignedBoundaryCandidate
from src.boundary_refinement import (
    BoundaryRefinementResult,
    RefinementReason,
    RefinementValidationStatus,
)
from src.critic import critique_decisions
from src.models import MetadataTimestamp, STTConfidence, STTTimestamp
from src.reporting import (
    build_audit_report,
    render_json_report,
    render_markdown_report,
    report_to_dict,
)
from src.resolver import ResolverEvidence, resolve_timestamp


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def candidate(index: int, metadata: float, stt: float, probability: float | None = 0.95, alignment: float = 0.95) -> AlignedBoundaryCandidate:
    return AlignedBoundaryCandidate(
        caption_index=index,
        caption_text=f"caption {index}",
        metadata_timestamp=MetadataTimestamp(metadata),
        stt_timestamp=STTTimestamp(stt),
        stt_word_index=index,
        stt_word="caption",
        stt_confidence=STTConfidence(probability) if probability is not None else None,
        aligned_stt_text=f"caption {index}",
        alignment_quality=alignment,
        matched_caption_tokens=2,
        caption_token_count=2,
    )


def decision(item: AlignedBoundaryCandidate):
    return resolve_timestamp(
        item.metadata_timestamp,
        item.stt_timestamp,
        ResolverEvidence(
            stt_confidence=item.stt_confidence,
            alignment_quality=item.alignment_quality,
        ),
    )


def report_for(candidates, decisions=None, **kwargs):
    decisions = decisions or [decision(item) for item in candidates]
    critics = critique_decisions(decisions)
    return build_audit_report(candidates, critics, generated_at=NOW, **kwargs)


def test_approved_stt_and_metadata_decisions_are_reported() -> None:
    stt = candidate(0, 10.0, 10.8)
    metadata = candidate(1, 20.0, 20.8, probability=0.2, alignment=0.1)
    report = report_for([stt, metadata])

    assert report.boundaries[0]["resolver"]["selected_source"] == "stt"
    assert report.boundaries[0]["critic"]["status"] == "approved"
    assert report.boundaries[1]["resolver"]["selected_source"] == "metadata"
    assert report.boundaries[1]["critic"]["status"] == "approved"


def test_unresolved_decision_is_withheld_for_human_review() -> None:
    item = candidate(0, 10.0, 10.8, probability=0.6, alignment=0.6)
    report = report_for([item])
    boundary = report.boundaries[0]

    assert boundary["resolver"]["selected_source"] == "unresolved"
    assert boundary["critic"]["status"] == "human_review"
    assert boundary["execution"]["final_timestamp_seconds"] is None
    assert boundary["execution"]["withheld"] is True


def test_summary_calculations() -> None:
    items = [
        candidate(0, 10.0, 10.4),
        candidate(1, 20.0, 20.8),
        candidate(2, 30.0, 30.8, probability=0.2, alignment=0.1),
        candidate(3, 40.0, 40.8, probability=0.6, alignment=0.6),
    ]
    report = report_for(
        items,
        executed_boundary_ids={"caption-block-1"},
        executed_cut_ids={"clip-1"},
    )
    summary = report.summary

    assert summary.total_boundaries == 4
    assert summary.major_conflicts == 3
    assert summary.metadata_wins == 2
    assert summary.stt_wins == 1
    assert summary.unresolved_decisions == 1
    assert summary.critic_approved == 3
    assert summary.human_review == 1
    assert summary.executed_cuts == 1
    assert summary.executed_boundaries == 1
    assert summary.average_disagreement_seconds == pytest.approx(0.7)
    assert summary.maximum_disagreement_seconds == pytest.approx(0.8)


def test_optional_refinement_has_separate_semantic_and_execution_fields() -> None:
    item = candidate(0, 10.0, 10.8)
    refinement = BoundaryRefinementResult(
        original_selected_timestamp=10.8,
        refined_timestamp=10.75,
        adjustment_milliseconds=-50.0,
        maximum_shift_seconds=0.15,
        refinement_applied=True,
        evidence=None,
        reason=RefinementReason.REFINED_TO_LOW_ENERGY,
        validation_status=RefinementValidationStatus.VALID,
    )
    report = report_for(
        [item],
        refinement_enabled=True,
        refinements={0: refinement},
        executed_boundary_ids={"caption-block-1"},
    )
    boundary = report.boundaries[0]

    assert boundary["resolver"]["semantic_timestamp_seconds"] == 10.8
    assert boundary["execution"]["final_timestamp_seconds"] == 10.75
    assert boundary["execution"]["refinement_used"] is True
    assert boundary["execution"]["refinement_adjustment_milliseconds"] == -50.0


def test_refinement_is_ignored_when_feature_is_disabled() -> None:
    item = candidate(0, 10.0, 10.8)
    refinement = BoundaryRefinementResult(
        10.8, 10.75, -50.0, 0.15, True, None,
        RefinementReason.REFINED_TO_LOW_ENERGY,
        RefinementValidationStatus.VALID,
    )
    boundary = report_for([item], refinements={0: refinement}).boundaries[0]

    assert boundary["execution"]["final_timestamp_seconds"] == 10.8
    assert boundary["execution"]["refinement_enabled"] is False
    assert boundary["execution"]["refinement_used"] is False
    assert boundary["execution"]["refinement_adjustment_milliseconds"] is None


def test_json_serialization_and_stable_top_level_structure() -> None:
    report = report_for([candidate(0, 10.0, 10.4)])
    payload = json.loads(render_json_report(report))

    assert list(payload) == [
        "schema_version", "generated_at", "configuration", "summary", "boundaries"
    ]
    assert set(payload["boundaries"][0]) == {
        "boundary_id", "caption_index", "caption_text", "target_stt_word",
        "alignment_derivation", "metadata_timestamp_seconds", "stt_timestamp_seconds",
        "absolute_disagreement_seconds", "exceeds_conflict_threshold",
        "stt_probability", "text_alignment_quality", "resolver", "critic", "execution",
    }
    assert set(payload["configuration"]) == {
        "conflict_threshold_seconds", "resolver", "critic", "refinement_enabled"
    }
    assert set(payload["summary"]) == {
        "total_boundaries", "major_conflicts", "metadata_wins", "stt_wins",
        "unresolved_decisions", "critic_approved", "human_review", "executed_cuts",
        "executed_boundaries", "average_disagreement_seconds",
        "maximum_disagreement_seconds",
    }
    assert set(payload["boundaries"][0]["resolver"]) == {
        "selected_source", "semantic_timestamp_seconds", "confidence",
        "source_scores", "reason_codes", "explanation",
    }
    assert set(payload["boundaries"][0]["critic"]) == {
        "status", "risk_score", "reason_codes", "explanation",
    }
    assert set(payload["boundaries"][0]["execution"]) == {
        "final_timestamp_seconds", "refinement_enabled", "refinement_used",
        "refinement_adjustment_milliseconds", "refinement_reason", "executed",
        "withheld",
    }
    assert payload == report_to_dict(report)
    assert payload["generated_at"] == "2026-08-16T12:00:00Z"


def test_markdown_contains_semantic_execution_and_review_language() -> None:
    item = candidate(0, 10.0, 10.8, probability=0.6, alignment=0.6)
    markdown = render_markdown_report(report_for([item]))

    assert "Resolver: unresolved" in markdown
    assert "Final execution timestamp: N/A" in markdown
    assert "HUMAN_REVIEW / WITHHELD" in markdown
    assert "### Resolver configuration\n\n```json" in markdown


def test_missing_optional_stt_evidence_serializes_as_null() -> None:
    item = candidate(0, 10.0, 10.4, probability=None)
    report = report_for([item])
    boundary = report.boundaries[0]

    assert boundary["stt_probability"] is None
    assert json.loads(render_json_report(report))["boundaries"][0]["stt_probability"] is None


def test_human_review_boundary_cannot_be_marked_executed() -> None:
    item = candidate(0, 10.0, 10.8, probability=0.6, alignment=0.6)
    with pytest.raises(ValueError, match="human-review boundary"):
        report_for([item], executed_boundary_ids={"caption-block-1"})


def test_report_rejects_threshold_configuration_mismatch() -> None:
    item = candidate(0, 10.0, 10.4)
    from src.resolver import ResolverConfig

    with pytest.raises(ValueError, match="must match"):
        report_for([item], conflict_threshold_seconds=0.6, resolver_config=ResolverConfig())


def test_report_rejects_unknown_execution_boundary() -> None:
    with pytest.raises(ValueError, match="unknown boundaries"):
        report_for(
            [candidate(0, 10.0, 10.4)],
            executed_boundary_ids={"caption-block-99"},
        )

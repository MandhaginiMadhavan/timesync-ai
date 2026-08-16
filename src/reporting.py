"""Deterministic human and machine-readable timestamp audit reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .alignment import AlignedBoundaryCandidate
from .boundary_refinement import BoundaryRefinementResult, RefinementValidationStatus
from .critic import CriticConfig, CriticResult, CriticStatus
from .resolver import ResolverConfig, TimestampSource


REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """Aggregate counts and disagreement statistics for an audit report."""

    total_boundaries: int
    major_conflicts: int
    metadata_wins: int
    stt_wins: int
    unresolved_decisions: int
    critic_approved: int
    human_review: int
    executed_cuts: int
    executed_boundaries: int
    average_disagreement_seconds: float | None
    maximum_disagreement_seconds: float | None


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Stable report envelope shared by JSON and Markdown renderers."""

    schema_version: str
    generated_at: str
    configuration: dict[str, object]
    summary: ReportSummary
    boundaries: tuple[dict[str, object], ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}s"


def _score_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _validate_parallel_inputs(
    candidates: Sequence[AlignedBoundaryCandidate],
    critic_results: Sequence[CriticResult],
) -> None:
    if len(candidates) != len(critic_results):
        raise ValueError("aligned candidates and Critic results must have equal lengths")
    for candidate, critic in zip(candidates, critic_results):
        decision = critic.resolver_decision
        if (
            decision.metadata_timestamp != candidate.metadata_timestamp.seconds
            or decision.stt_timestamp != candidate.stt_timestamp.seconds
        ):
            raise ValueError(
                f"Critic result does not match caption block {candidate.caption_index + 1}"
            )


def build_audit_report(
    candidates: Sequence[AlignedBoundaryCandidate],
    critic_results: Sequence[CriticResult],
    *,
    conflict_threshold_seconds: float = 0.5,
    resolver_config: ResolverConfig | None = None,
    critic_config: CriticConfig | None = None,
    refinement_enabled: bool = False,
    refinements: Mapping[int, BoundaryRefinementResult] | None = None,
    executed_boundary_ids: set[str] | frozenset[str] | None = None,
    executed_cut_ids: set[str] | frozenset[str] | None = None,
    generated_at: datetime | None = None,
) -> AuditReport:
    """Build an audit report without rerunning any upstream decision logic.

    ``refinements`` is keyed by zero-based ``caption_index``. A refinement is
    used only when the report explicitly declares refinement enabled, it is
    valid, and it records a real adjustment. Execution status must be supplied
    explicitly; the report never infers that a media operation occurred.
    """
    if not math.isfinite(conflict_threshold_seconds) or conflict_threshold_seconds < 0:
        raise ValueError("conflict threshold must be finite and non-negative")
    _validate_parallel_inputs(candidates, critic_results)
    resolver_config = resolver_config or ResolverConfig(
        major_conflict_threshold_seconds=conflict_threshold_seconds
    )
    if not math.isclose(
        resolver_config.major_conflict_threshold_seconds,
        conflict_threshold_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("report threshold must match the Resolver configuration")
    critic_config = critic_config or CriticConfig()
    refinements = refinements or {}
    executed_boundary_ids = executed_boundary_ids or frozenset()
    executed_cut_ids = executed_cut_ids or frozenset()
    known_caption_indices = {candidate.caption_index for candidate in candidates}
    unknown_refinements = set(refinements) - known_caption_indices
    if unknown_refinements:
        raise ValueError(f"refinements reference unknown caption indices: {sorted(unknown_refinements)}")
    known_boundary_ids = {
        f"caption-block-{candidate.caption_index + 1}" for candidate in candidates
    }
    unknown_executions = set(executed_boundary_ids) - known_boundary_ids
    if unknown_executions:
        raise ValueError(f"execution references unknown boundaries: {sorted(unknown_executions)}")

    boundaries: list[dict[str, object]] = []
    for candidate, critic in zip(candidates, critic_results):
        decision = critic.resolver_decision
        boundary_id = f"caption-block-{candidate.caption_index + 1}"
        refinement = refinements.get(candidate.caption_index)
        refinement_used = bool(
            refinement_enabled
            and refinement is not None
            and refinement.validation_status is RefinementValidationStatus.VALID
            and refinement.refinement_applied
        )
        if refinement is not None and (
            refinement.original_selected_timestamp != decision.selected_timestamp
        ):
            raise ValueError(f"refinement does not match {boundary_id}")

        if critic.status is CriticStatus.HUMAN_REVIEW:
            execution_timestamp = None
        elif refinement_used:
            execution_timestamp = refinement.refined_timestamp
        else:
            execution_timestamp = decision.selected_timestamp

        executed = boundary_id in executed_boundary_ids
        if executed and critic.status is not CriticStatus.APPROVED:
            raise ValueError(f"human-review boundary cannot be executed: {boundary_id}")
        if executed and execution_timestamp is None:
            raise ValueError(f"executed boundary has no execution timestamp: {boundary_id}")

        diagnostics = decision.diagnostics
        boundaries.append(
            {
                "boundary_id": boundary_id,
                "caption_index": candidate.caption_index,
                "caption_text": candidate.caption_text,
                "target_stt_word": candidate.stt_word,
                "alignment_derivation": candidate.derivation,
                "metadata_timestamp_seconds": decision.metadata_timestamp,
                "stt_timestamp_seconds": decision.stt_timestamp,
                "absolute_disagreement_seconds": decision.disagreement_seconds,
                "exceeds_conflict_threshold": (
                    decision.disagreement_seconds is not None
                    and decision.disagreement_seconds > conflict_threshold_seconds
                ),
                "stt_probability": diagnostics.stt_confidence,
                "text_alignment_quality": diagnostics.alignment_quality,
                "resolver": {
                    "selected_source": decision.selected_source.value,
                    "semantic_timestamp_seconds": decision.selected_timestamp,
                    "confidence": decision.resolver_confidence,
                    "source_scores": {
                        "metadata": diagnostics.metadata_score,
                        "stt": diagnostics.stt_score,
                        "decision_margin": diagnostics.decision_margin,
                    },
                    "reason_codes": [reason.value for reason in decision.reason_codes],
                    "explanation": decision.explanation,
                },
                "critic": {
                    "status": critic.status.value,
                    "risk_score": critic.risk_score,
                    "reason_codes": [reason.value for reason in critic.reason_codes],
                    "explanation": critic.explanation,
                },
                "execution": {
                    "final_timestamp_seconds": execution_timestamp,
                    "refinement_enabled": refinement_enabled,
                    "refinement_used": refinement_used,
                    "refinement_adjustment_milliseconds": (
                        refinement.adjustment_milliseconds if refinement_used else None
                    ),
                    "refinement_reason": (
                        refinement.reason.value if refinement is not None else None
                    ),
                    "executed": executed,
                    "withheld": not executed,
                },
            }
        )

    disagreements = [
        float(item["absolute_disagreement_seconds"])
        for item in boundaries
        if item["absolute_disagreement_seconds"] is not None
    ]
    summary = ReportSummary(
        total_boundaries=len(boundaries),
        major_conflicts=sum(bool(item["exceeds_conflict_threshold"]) for item in boundaries),
        metadata_wins=sum(
            item["resolver"]["selected_source"] == TimestampSource.METADATA.value
            for item in boundaries
        ),
        stt_wins=sum(
            item["resolver"]["selected_source"] == TimestampSource.STT.value
            for item in boundaries
        ),
        unresolved_decisions=sum(
            item["resolver"]["selected_source"] == TimestampSource.UNRESOLVED.value
            for item in boundaries
        ),
        critic_approved=sum(
            item["critic"]["status"] == CriticStatus.APPROVED.value
            for item in boundaries
        ),
        human_review=sum(
            item["critic"]["status"] == CriticStatus.HUMAN_REVIEW.value
            for item in boundaries
        ),
        executed_cuts=len(executed_cut_ids),
        executed_boundaries=sum(bool(item["execution"]["executed"]) for item in boundaries),
        average_disagreement_seconds=(
            round(sum(disagreements) / len(disagreements), 6) if disagreements else None
        ),
        maximum_disagreement_seconds=(round(max(disagreements), 6) if disagreements else None),
    )
    timestamp = generated_at or _utc_now()
    if timestamp.tzinfo is None:
        raise ValueError("generation timestamp must be timezone-aware")
    return AuditReport(
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at=timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        configuration={
            "conflict_threshold_seconds": conflict_threshold_seconds,
            "resolver": asdict(resolver_config),
            "critic": asdict(critic_config),
            "refinement_enabled": refinement_enabled,
        },
        summary=summary,
        boundaries=tuple(boundaries),
    )


def report_to_dict(report: AuditReport) -> dict[str, object]:
    """Convert a report to its stable JSON-compatible field structure."""
    payload = asdict(report)
    payload["boundaries"] = list(payload["boundaries"])
    return payload


def render_json_report(report: AuditReport) -> str:
    """Serialize a report as readable, deterministic UTF-8 JSON text."""
    return json.dumps(report_to_dict(report), indent=2, ensure_ascii=False) + "\n"


def render_markdown_report(report: AuditReport) -> str:
    """Render a concise human-review report with full per-boundary details."""
    summary = report.summary
    config = report.configuration
    lines = [
        "# TimeSync AI Timestamp Audit",
        "",
        f"Generated: {report.generated_at}",
        f"Schema: {report.schema_version}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total boundaries | {summary.total_boundaries} |",
        f"| Conflicts > {config['conflict_threshold_seconds']:.3f}s | {summary.major_conflicts} |",
        f"| Metadata wins | {summary.metadata_wins} |",
        f"| STT wins | {summary.stt_wins} |",
        f"| Unresolved | {summary.unresolved_decisions} |",
        f"| Critic approved | {summary.critic_approved} |",
        f"| Human review | {summary.human_review} |",
        f"| Executed cuts | {summary.executed_cuts} |",
        f"| Executed boundary endpoints | {summary.executed_boundaries} |",
        f"| Average disagreement | {_timestamp_text(summary.average_disagreement_seconds)} |",
        f"| Maximum disagreement | {_timestamp_text(summary.maximum_disagreement_seconds)} |",
        "",
        "## Configuration",
        "",
        f"- Conflict threshold: {config['conflict_threshold_seconds']:.3f}s",
        f"- Optional refinement enabled: {str(config['refinement_enabled']).lower()}",
        "",
        "### Resolver configuration",
        "",
        "```json",
        json.dumps(config["resolver"], indent=2, sort_keys=True),
        "```",
        "",
        "### Critic configuration",
        "",
        "```json",
        json.dumps(config["critic"], indent=2, sort_keys=True),
        "```",
        "",
        "## Boundary audit",
        "",
    ]
    for item in report.boundaries:
        resolver = item["resolver"]
        critic = item["critic"]
        execution = item["execution"]
        disposition = "EXECUTED" if execution["executed"] else "WITHHELD"
        lines.extend(
            [
                f"### {item['boundary_id']} -- {critic['status'].upper()} / {disposition}",
                "",
                f"Caption: {item['caption_text']}",
                "",
                f"- Target STT word: `{item['target_stt_word']}`",
                f"- Metadata / STT: {_timestamp_text(item['metadata_timestamp_seconds'])} / {_timestamp_text(item['stt_timestamp_seconds'])}",
                f"- Disagreement: {_timestamp_text(item['absolute_disagreement_seconds'])} (major: {str(item['exceeds_conflict_threshold']).lower()})",
                f"- STT probability / text alignment: {_score_text(item['stt_probability'])} / {_score_text(item['text_alignment_quality'])}",
                f"- Resolver: {resolver['selected_source']} at {_timestamp_text(resolver['semantic_timestamp_seconds'])}, confidence {_score_text(resolver['confidence'])}",
                f"- Resolver scores: metadata {_score_text(resolver['source_scores']['metadata'])}, STT {_score_text(resolver['source_scores']['stt'])}, margin {_score_text(resolver['source_scores']['decision_margin'])}",
                f"- Resolver reasons: {', '.join(resolver['reason_codes']) or 'none'}",
                f"- Resolver explanation: {resolver['explanation']}",
                f"- Critic: {critic['status']}, risk {_score_text(critic['risk_score'])}",
                f"- Critic reasons: {', '.join(critic['reason_codes']) or 'none'}",
                f"- Critic explanation: {critic['explanation']}",
                f"- Final execution timestamp: {_timestamp_text(execution['final_timestamp_seconds'])}",
                f"- Optional refinement used: {str(execution['refinement_used']).lower()} ({_score_text(execution['refinement_adjustment_milliseconds'])}ms)",
                f"- Execution: {disposition}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_audit_reports(
    report: AuditReport,
    markdown_path: str | Path,
    json_path: str | Path,
) -> None:
    """Write both report formats as UTF-8 files."""
    markdown_destination = Path(markdown_path)
    json_destination = Path(json_path)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.write_text(render_markdown_report(report), encoding="utf-8")
    json_destination.write_text(render_json_report(report), encoding="utf-8")

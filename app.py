"""Polished single-page Streamlit interface for TimeSync AI."""

from __future__ import annotations

from html import escape
from pathlib import Path
import shutil
from uuid import uuid4

import streamlit as st

from src.human_review import (
    BoundaryContext,
    HumanReviewDecision,
    HumanReviewSource,
    ReviewPreview,
    build_boundary_context,
    create_review_preview,
    neighbouring_accepted_timestamps,
    validate_human_review_decision,
    write_human_review_ledger,
)
from src.pipeline import PipelineConfig, PipelineError, PipelineResult, run_pipeline
from src.video_cutter import probe_media


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_FFMPEG = PROJECT_ROOT / ".tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
LOCAL_FFPROBE = PROJECT_ROOT / ".tools" / "ffmpeg" / "bin" / "ffprobe.exe"
LOCAL_MODELS = PROJECT_ROOT / ".tools" / "whisper"


def _tool_path(local_path: Path, fallback: str) -> str:
    return str(local_path) if local_path.is_file() else fallback


def _format_seconds(value: object) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.3f}s"


def _format_score(value: object) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.3f}"


def _format_probability(value: object) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value * 100:.1f}%"


def _friendly_explanation(value: object) -> str:
    return str(value).replace("STT", "Whisper").replace("stt", "Whisper")


def _save_upload(upload, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(upload.getvalue())
    return destination


def _cleanup_upload_directory(path_value: object) -> None:
    if not isinstance(path_value, (str, Path)):
        return
    upload_root = (PROJECT_ROOT / ".tools" / "ui_uploads").resolve()
    target = Path(path_value).resolve()
    if target.is_relative_to(upload_root) and target != upload_root and target.is_dir():
        shutil.rmtree(target)


def _initial_manual_timestamp(boundary: dict[str, object]) -> float:
    candidates = [
        float(value)
        for value in (
            boundary.get("metadata_timestamp_seconds"),
            boundary.get("stt_timestamp_seconds"),
        )
        if isinstance(value, (int, float))
    ]
    if candidates:
        return sum(candidates) / len(candidates)
    resolver = boundary.get("resolver")
    if isinstance(resolver, dict):
        selected = resolver.get("semantic_timestamp_seconds")
        if isinstance(selected, (int, float)):
            return float(selected)
    return 0.0


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #07101f; }
        .block-container { max-width: 1120px; padding: 3.5rem 2.25rem 7rem; }
        h1, h2, h3 { letter-spacing: -0.035em; }
        h1 { font-size: clamp(2.8rem, 7vw, 5.3rem) !important; line-height: .98 !important; }
        h2 { margin-top: 3.5rem !important; }
        .eyebrow { color: #9aa9c3; font-size: .76rem; letter-spacing: .16em;
                   text-transform: uppercase; font-weight: 700; }
        .hero-copy { color: #aab7cd; font-size: 1.15rem; max-width: 690px;
                     line-height: 1.65; margin: 1rem 0 2.2rem; }
        .accent { background: linear-gradient(90deg, #65a5ff, #9c7cff);
                  -webkit-background-clip: text; color: transparent; }
        div[data-testid="stMetric"] { background: #0b1526; border: 1px solid #1b2940;
                  border-radius: 13px; padding: .7rem .85rem; }
        div[data-testid="stMetricLabel"] { color: #91a0b8; }
        div[data-testid="stExpander"] { background: #0b1526; border: 1px solid #1d2a40;
                  border-radius: 14px; overflow: hidden; }
        div[data-testid="stFileUploader"] section { background: #0b1526;
                  border: 1px dashed #344563; border-radius: 15px; }
        .review-card { background: #10192b; border: 1px solid #493f75;
                  border-radius: 18px; padding: 1.25rem 1.4rem .35rem;
                  box-shadow: 0 14px 36px #03071255; margin: .75rem 0; }
        .review-label { color: #b9aefc; font-size: .74rem; font-weight: 700;
                  letter-spacing: .12em; text-transform: uppercase; }
        .word-context { font-size: 1.35rem; line-height: 1.75; color: #aebbd0;
                  text-align: center; padding: 1.2rem .5rem 1.8rem; }
        .target-word { color: #f5f2ff; background: #6f5ee844; border: 1px solid #7666de;
                  border-radius: 8px; padding: .18rem .48rem; font-weight: 800; }
        .compact-complete { color: #aebbd0; padding: .65rem .85rem; border: 1px solid #20304a;
                  border-radius: 11px; background: #0b1526; }
        .muted { color: #91a0b8; }
        .stage-line { padding: .48rem .72rem; border-left: 2px solid #30415d;
                  color: #aab7cd; margin: .2rem 0; }
        .stage-current { border-left-color: #8878ff; color: #f1efff; background: #151b35; }
        .stButton > button[kind="primary"] { border: 0; border-radius: 12px;
                  background: linear-gradient(90deg, #377ee9, #7964e8); font-weight: 700; }
        .stButton > button { border-radius: 11px; }
        hr { border-color: #18253a !important; margin: 4rem 0 !important; }
        @media (max-width: 760px) { .block-container { padding: 2rem 1rem 5rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_stage_status(message: str, completed: list[str], placeholder) -> None:
    completed.append(message)
    rows = []
    for index, stage in enumerate(completed):
        css = "stage-current" if index == len(completed) - 1 else ""
        rows.append(f'<div class="stage-line {css}">{stage}</div>')
    placeholder.markdown("".join(rows), unsafe_allow_html=True)


def _render_summary(result: PipelineResult) -> None:
    st.header("Analysis summary")
    first = st.columns(3)
    second = st.columns(3)
    values = (
        ("Aligned boundaries", result.aligned_boundary_count),
        ("Conflicts", result.major_conflict_count),
        ("Approved", result.approved_boundary_count),
        ("Human review", len(result.withheld_boundary_ids)),
        ("Clips generated", len(result.executed_clips)),
        ("Processing time", f"{result.total_processing_seconds:.1f}s"),
    )
    for column, (label, value) in zip((*first, *second), values):
        column.metric(label, value)
    st.markdown(
        f"**{result.approved_boundary_count} of {result.aligned_boundary_count} boundaries "
        f"were resolved automatically. {len(result.withheld_boundary_ids)} need human review.**"
    )


def _render_conflicts(result: PipelineResult) -> None:
    conflicts = [
        item for item in result.report.boundaries if item["exceeds_conflict_threshold"]
    ]
    st.header("Timestamp Decisions")
    st.caption("Open a disputed boundary to see what TimeSync AI selected and why.")
    for item in conflicts:
        resolver = item["resolver"]
        critic = item["critic"]
        label = (
            f"Decision #{item['caption_index'] + 1} · {_format_seconds(item['absolute_disagreement_seconds'])} difference "
            f"· {resolver['selected_source'].replace('stt', 'Whisper').upper()} "
            f"· {critic['status'].replace('_', ' ').upper()}"
        )
        with st.expander(label, expanded=False):
            st.markdown(
                f"#### Timestamp Decision #{item['caption_index'] + 1} · "
                f"{critic['status'].replace('_', ' ').upper()}"
            )
            cols = st.columns(3)
            cols[0].metric("Metadata", _format_seconds(item["metadata_timestamp_seconds"]))
            cols[1].metric("Whisper", _format_seconds(item["stt_timestamp_seconds"]))
            cols[2].metric("Disagreement", _format_seconds(item["absolute_disagreement_seconds"]))
            selected_label = resolver["selected_source"].replace("stt", "Whisper").upper()
            st.markdown(
                f"**AI decision**  \n{selected_label} → "
                f"{_format_seconds(resolver['semantic_timestamp_seconds'])}"
            )
            st.caption(f"Whisper recognition confidence: {_format_probability(item['stt_probability'])}")
            st.markdown(f"**Why?**  \n{_friendly_explanation(resolver['explanation'])}")
            st.markdown(
                f"**Independent review — {critic['status'].replace('_', ' ').upper()}**  \n"
                f"{_friendly_explanation(critic['explanation'])}"
            )


def _record_review(
    result: PipelineResult,
    boundary: dict[str, object],
    boundary_index: int,
    source: HumanReviewSource,
    *,
    manual_timestamp: float | None = None,
) -> HumanReviewDecision:
    media_path = Path(st.session_state["source_video_path"])
    duration = probe_media(
        media_path,
        ffprobe_path=st.session_state["ffprobe_path"],
    ).duration_seconds
    decisions: dict[str, HumanReviewDecision] = st.session_state.setdefault(
        "review_decisions", {}
    )
    previous, following = neighbouring_accepted_timestamps(
        result.report.boundaries, boundary_index, decisions
    )
    decision = validate_human_review_decision(
        boundary,
        source,
        manual_timestamp_seconds=manual_timestamp,
        media_duration_seconds=duration,
        previous_accepted_timestamp_seconds=previous,
        next_accepted_timestamp_seconds=following,
    )
    decisions[decision.boundary_id] = decision
    write_human_review_ledger(
        tuple(decisions.values()), result.output_directory / "human_review_decisions.json"
    )
    return decision


def _human_review_cases(result: PipelineResult) -> list[tuple[int, dict[str, object]]]:
    return [
        (index, item)
        for index, item in enumerate(result.report.boundaries)
        if item["critic"]["status"] == "human_review"
    ]


def _render_human_review_cta(result: PipelineResult) -> None:
    cases = _human_review_cases(result)
    if not cases:
        st.success("All boundaries passed independent validation. No human review is needed.")
        return
    st.header("Human review")
    st.markdown(
        f"**{len(cases)} boundaries need your review.**  \n"
        "TimeSync AI found timing evidence that was not reliable enough to make "
        "these cuts automatically."
    )
    if st.button("Open Human Review", type="primary"):
        st.session_state["view"] = "human_review"
        st.session_state["review_index"] = 0
        st.rerun()


def _preview_for(
    result: PipelineResult,
    boundary: dict[str, object],
    label: str,
    timestamp: float,
) -> ReviewPreview:
    cache = st.session_state.setdefault("review_previews", {})
    cache_key = f"{boundary['boundary_id']}:{label}:{timestamp:.6f}"
    if cache_key not in cache:
        preview_directory = result.output_directory / "human-review" / "previews"
        safe_label = label.casefold().replace(" ", "-")
        cache[cache_key] = create_review_preview(
            st.session_state["source_video_path"],
            preview_directory / f"{boundary['boundary_id']}-{safe_label}.mp4",
            timestamp,
            allowed_output_directory=preview_directory,
            ffmpeg_path=st.session_state["ffmpeg_path"],
            ffprobe_path=st.session_state["ffprobe_path"],
        )
    return cache[cache_key]


def _context_html(context: BoundaryContext) -> str:
    before = f"…{escape(context.before)} " if context.before else ""
    after = f" {escape(context.after)}…" if context.after else ""
    return (
        f'<div class="word-context">{before}'
        f'<span class="target-word">{escape(context.target_word.upper())}</span>{after}</div>'
    )


def _render_focused_human_review(result: PipelineResult) -> None:
    cases = _human_review_cases(result)
    if not cases:
        st.session_state["view"] = "results"
        st.rerun()
    if st.button("← Back to results"):
        st.session_state["view"] = "results"
        st.rerun()
    review_index = min(max(int(st.session_state.get("review_index", 0)), 0), len(cases) - 1)
    boundary_index, item = cases[review_index]
    resolver = item["resolver"]
    st.markdown('<div class="eyebrow">Focused human validation</div>', unsafe_allow_html=True)
    st.title("Human Review")
    st.caption(f"Boundary {review_index + 1} of {len(cases)}")
    st.subheader("Which timestamp correctly starts this word?")
    context = build_boundary_context(str(item["caption_text"]), str(item["target_stt_word"]))
    st.markdown(_context_html(context), unsafe_allow_html=True)

    candidates = st.columns(2)
    candidate_specs = (
        (candidates[0], "METADATA", item["metadata_timestamp_seconds"]),
        (candidates[1], "WHISPER", item["stt_timestamp_seconds"]),
    )
    for column, label, value in candidate_specs:
        with column:
            st.markdown(f"### {label}")
            st.metric("Timestamp in original video", _format_seconds(value))
            if isinstance(value, (int, float)):
                with st.spinner(f"Preparing {label.title()} preview…"):
                    preview = _preview_for(result, item, label, float(value))
                st.caption(
                    f"Candidate boundary: {float(value):.3f}s from the start of the original video. "
                    "The candidate is centred in this preview."
                )
                st.video(str(preview.output_path))
    st.caption(f"Difference: {_format_seconds(item['absolute_disagreement_seconds'])}")
    recommendation = resolver["selected_source"].replace("stt", "Whisper").upper()
    st.markdown(
        f"**AI recommendation:** {recommendation} → "
        f"{_format_seconds(resolver['semantic_timestamp_seconds'])}"
    )
    st.markdown("**Why does this need review?**")
    st.write(_friendly_explanation(item["critic"]["explanation"]))

    decisions = st.session_state.setdefault("review_decisions", {})
    key = str(item["boundary_id"])
    try:
        actions = st.columns(2)
        if actions[0].button("Use Metadata", key=f"focused-metadata-{key}", use_container_width=True):
            _record_review(result, item, boundary_index, HumanReviewSource.METADATA)
            st.success("Metadata timestamp validated and recorded.")
        if actions[1].button("Use Whisper", key=f"focused-whisper-{key}", use_container_width=True):
            _record_review(result, item, boundary_index, HumanReviewSource.WHISPER)
            st.success("Whisper timestamp validated and recorded.")

        manual = st.number_input(
            "Timestamp in original video (seconds)", min_value=0.0,
            value=_initial_manual_timestamp(item),
            step=0.001, format="%.3f", key=f"focused-manual-value-{key}",
        )
        manual_actions = st.columns(2)
        if manual_actions[0].button("Preview Manual Time", key=f"preview-manual-{key}", use_container_width=True):
            media = probe_media(
                st.session_state["source_video_path"], ffprobe_path=st.session_state["ffprobe_path"]
            )
            previous, following = neighbouring_accepted_timestamps(
                result.report.boundaries, boundary_index, decisions
            )
            validate_human_review_decision(
                item, HumanReviewSource.MANUAL, manual_timestamp_seconds=float(manual),
                media_duration_seconds=media.duration_seconds,
                previous_accepted_timestamp_seconds=previous,
                next_accepted_timestamp_seconds=following,
            )
            st.session_state[f"manual-preview-{key}"] = _preview_for(
                result, item, f"manual-{float(manual):.3f}", float(manual)
            )
        if manual_actions[1].button("Use Manual Time", key=f"use-manual-{key}", use_container_width=True):
            _record_review(
                result, item, boundary_index, HumanReviewSource.MANUAL,
                manual_timestamp=float(manual),
            )
            st.success("Manual timestamp validated and recorded.")
    except (OSError, ValueError, RuntimeError) as error:
        st.error(str(error))

    manual_preview = st.session_state.get(f"manual-preview-{key}")
    if manual_preview is not None:
        st.caption(
            f"Manual candidate: {manual_preview.candidate_timestamp_seconds:.3f}s from "
            "the start of the original video, centred in this preview."
        )
        st.video(str(manual_preview.output_path))
    saved = decisions.get(key)
    if saved:
        st.info(
            f"Recorded {saved.human_selected_source.value.upper()} at "
            f"{saved.human_selected_timestamp_seconds:.3f}s. AI provenance is unchanged; "
            "automatic media execution remains withheld."
        )
    navigation = st.columns(2)
    if navigation[0].button("Previous", disabled=review_index == 0, use_container_width=True):
        st.session_state["review_index"] = review_index - 1
        st.rerun()
    if navigation[1].button("Next", disabled=review_index == len(cases) - 1, use_container_width=True):
        st.session_state["review_index"] = review_index + 1
        st.rerun()


def _render_outputs(result: PipelineResult) -> None:
    st.header("Generated clips")
    st.caption("Frame-accurate clips retain the original video soundtrack. Preview only when needed.")
    if not result.executed_clips:
        st.info("No intervals had two approved boundaries, so no clips were generated.")
        return
    for clip in result.executed_clips:
        columns = st.columns([1.6, 1, 1])
        columns[0].markdown(f"**{clip.clip_id.replace('-', ' ').title()}**  \n{clip.caption_text[:72]}{'…' if len(clip.caption_text) > 72 else ''}")
        with columns[1].expander("Preview"):
            st.video(str(clip.cut_result.output_path))
        with columns[2]:
                st.download_button(
                    "Download clip",
                    data=clip.cut_result.output_path.read_bytes(),
                    file_name=clip.cut_result.output_path.name,
                    mime="video/mp4",
                    key=f"download-{clip.clip_id}",
                    use_container_width=True,
                )


def _render_reports(result: PipelineResult) -> None:
    st.header("Audit Trail")
    st.caption("Every timestamp, AI decision, review and final action is recorded.")
    columns = st.columns(2)
    columns[0].download_button(
        "Download Markdown Report",
        result.markdown_report_path.read_bytes(),
        file_name="timesync-report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    columns[1].download_button(
        "Download JSON Audit",
        result.json_report_path.read_bytes(),
        file_name="timesync-report.json",
        mime="application/json",
        use_container_width=True,
    )
    ledger = result.output_directory / "human_review_decisions.json"
    if ledger.is_file():
        st.download_button(
            "Download Human Review Ledger",
            ledger.read_bytes(),
            file_name="human-review-decisions.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="TimeSync AI", page_icon="⏱", layout="wide")
    _inject_styles()
    result = st.session_state.get("pipeline_result")
    if st.session_state.get("view") == "human_review" and result is not None:
        _render_focused_human_review(result)
        return
    st.markdown('# TimeSync <span class="accent">AI</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Intelligent timestamp reconciliation for precise, reliable video clipping.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploads = st.columns(2)
        video_upload = uploads[0].file_uploader(
            "Video", type=["mp4", "mov", "mkv", "webm"], help="Original media with audio"
        )
        transcript_upload = uploads[1].file_uploader(
            "Timestamp / caption metadata", type=["txt"], help="TimeSync caption metadata format"
        )
        model = st.selectbox("Whisper model", ("tiny.en", "base.en", "small.en"), index=2)
        with st.expander("Advanced options", expanded=False):
            refinement = st.toggle(
                "Experimental low-energy boundary refinement",
                value=False,
                help="Optional execution-only adjustment up to ±150 ms. Resolver semantics remain unchanged.",
            )
        analyse = st.button("Analyse Video", type="primary", use_container_width=True)

    if analyse:
        _cleanup_upload_directory(st.session_state.pop("upload_root", None))
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("review_decisions", None)
        if video_upload is None or transcript_upload is None:
            st.error("Upload both a video and timestamp/caption file to continue.")
        else:
            run_id = uuid4().hex[:12]
            upload_root = PROJECT_ROOT / ".tools" / "ui_uploads" / run_id
            video_suffix = Path(video_upload.name).suffix.lower() or ".mp4"
            source_video = _save_upload(video_upload, upload_root / f"source{video_suffix}")
            source_transcript = _save_upload(transcript_upload, upload_root / "captions.txt")
            output = PROJECT_ROOT / "output" / "ui-runs" / run_id
            ffmpeg = _tool_path(LOCAL_FFMPEG, "ffmpeg")
            ffprobe = _tool_path(LOCAL_FFPROBE, "ffprobe")
            st.subheader("Processing")
            stage_placeholder = st.empty()
            completed: list[str] = []
            try:
                with st.spinner("Analysing video and validating timestamp evidence…", show_time=True):
                    result = run_pipeline(
                        PipelineConfig(
                            video_path=source_video,
                            transcript_path=source_transcript,
                            output_directory=output,
                            whisper_model=model,
                            whisper_download_root=LOCAL_MODELS,
                            refinement_enabled=refinement,
                            ffmpeg_path=ffmpeg,
                            ffprobe_path=ffprobe,
                        ),
                        logger=lambda message: _render_stage_status(
                            message, completed, stage_placeholder
                        ),
                    )
                st.session_state["pipeline_result"] = result
                st.session_state["upload_root"] = str(upload_root)
                st.session_state["source_video_path"] = str(source_video)
                st.session_state["ffmpeg_path"] = ffmpeg
                st.session_state["ffprobe_path"] = ffprobe
                st.session_state["review_decisions"] = {}
                st.session_state["review_previews"] = {}
                st.session_state["view"] = "results"
                stage_placeholder.empty()
            except PipelineError as error:
                _cleanup_upload_directory(upload_root)
                st.error(f"Analysis failed during {error.stage}: {error}")

    result = st.session_state.get("pipeline_result")
    if result is not None:
        st.divider()
        st.markdown(
            f'<div class="compact-complete">✓ Analysis complete · '
            f'{result.total_processing_seconds:.1f}s</div>',
            unsafe_allow_html=True,
        )
        _render_summary(result)
        st.divider()
        _render_conflicts(result)
        _render_human_review_cta(result)
        st.divider()
        _render_outputs(result)
        st.divider()
        _render_reports(result)


if __name__ == "__main__":
    main()

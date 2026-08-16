"""Headless smoke tests for the progressive Streamlit interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.human_review import ReviewPreview
from src.pipeline import ExecutedClip, PipelineResult
from src.reporting import AuditReport, ReportSummary
from src.video_cutter import CutResult, CutStatus, MediaVerification


def completed_result(tmp_path: Path) -> PipelineResult:
    output = tmp_path / "run"
    clips = output / "clips"
    clips.mkdir(parents=True)
    clip_path = clips / "clip-001.mp4"
    clip_path.write_bytes(b"clip")
    markdown = output / "report.md"
    json_report = output / "report.json"
    markdown.write_text("# report", encoding="utf-8")
    json_report.write_text("{}", encoding="utf-8")
    boundaries = (
        {
            "boundary_id": "caption-block-1", "caption_index": 0,
            "caption_text": "alpha words", "target_stt_word": "alpha",
            "metadata_timestamp_seconds": 0.0, "stt_timestamp_seconds": 0.1,
            "absolute_disagreement_seconds": 0.1, "exceeds_conflict_threshold": False,
            "stt_probability": 0.9, "text_alignment_quality": 1.0,
            "resolver": {"selected_source": "metadata", "semantic_timestamp_seconds": 0.0,
                         "explanation": "Selected metadata.", "source_scores": {}},
            "critic": {"status": "approved", "explanation": "Checks passed."},
            "execution": {"executed": True},
        },
        {
            "boundary_id": "caption-block-2", "caption_index": 1,
            "caption_text": "one turn of pitch-and-toss and lose", "target_stt_word": "turn",
            "metadata_timestamp_seconds": 10.0, "stt_timestamp_seconds": 10.8,
            "absolute_disagreement_seconds": 0.8, "exceeds_conflict_threshold": True,
            "stt_probability": 0.95, "text_alignment_quality": 0.8,
            "resolver": {"selected_source": "stt", "semantic_timestamp_seconds": 10.8,
                         "explanation": "Selected STT because alignment was strong.", "source_scores": {}},
            "critic": {"status": "human_review",
                       "explanation": "The selected STT offset differs from nearby boundaries."},
            "execution": {"executed": False},
        },
    )
    summary = ReportSummary(2, 1, 1, 1, 0, 1, 1, 1, 1, 0.45, 0.8)
    report = AuditReport("1.0", "2026-08-16T12:00:00Z", {"refinement_enabled": False}, summary, boundaries)
    verification = MediaVerification(True, 10.0, 10.0, 0.0, True, True, True)
    cut = CutResult(0.0, 10.0, 10.0, clip_path, CutStatus.SUCCESS, verification, ())
    return PipelineResult(
        output, output / "caption_metadata.json", output / "whisper_words.json",
        markdown, json_report, 2, 1, 1, ("caption-block-2",),
        (ExecutedClip("clip-001", "alpha words", "caption-block-1", "caption-block-2", cut),),
        False, {}, 4.2, report,
    )


def completed_page(tmp_path: Path) -> AppTest:
    app_path = Path(__file__).parents[1] / "app.py"
    page = AppTest.from_file(str(app_path), default_timeout=10).run()
    page.session_state["pipeline_result"] = completed_result(tmp_path)
    page.session_state["view"] = "results"
    page.session_state["review_decisions"] = {}
    return page.run()


def test_initial_page_has_focused_inputs_and_no_result_sections() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    page = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not page.exception
    assert [button.label for button in page.button] == ["Analyse Video"]
    assert [selector.label for selector in page.selectbox] == ["Whisper model"]
    assert page.selectbox[0].value == "small.en"
    assert page.selectbox[0].options == ["tiny.en", "base.en", "small.en"]
    assert len(page.get("file_uploader")) == 2
    assert len(page.toggle) == 1
    assert page.toggle[0].label == "Experimental low-energy boundary refinement"
    assert page.toggle[0].value is False
    assert any(
        "Intelligent timestamp reconciliation for precise, reliable video clipping."
        in markdown.value
        for markdown in page.markdown
    )
    assert "Analysis summary" not in [header.value for header in page.header]


def test_completed_page_uses_decisions_cta_and_compact_downloads(tmp_path: Path) -> None:
    page = completed_page(tmp_path)

    assert not page.exception
    headers = [header.value for header in page.header]
    assert headers == [
        "Analysis summary", "Timestamp Decisions", "Human review",
        "Generated clips", "Audit Trail",
    ]
    assert [item.label for item in page.expander] == [
        "Advanced options",
        "Decision #2 · 0.800s difference · WHISPER · HUMAN REVIEW",
        "Preview",
    ]
    rendered_text = "\n".join(markdown.value for markdown in page.markdown)
    assert "Technical evidence" not in rendered_text
    assert not page.get("json")
    assert "Open Human Review" in [button.label for button in page.button]
    assert {button.label for button in page.get("download_button")} == {
        "Download clip", "Download Markdown Report", "Download JSON Audit"
    }


def test_focused_human_review_navigation_and_target_word(tmp_path: Path) -> None:
    page = completed_page(tmp_path)
    preview_file = tmp_path / "preview.mp4"
    preview_file.write_bytes(b"preview")
    preview = ReviewPreview(10.0, 8.0, 12.0, 2.0, preview_file, 4.0)
    page.session_state["review_previews"] = {
        "caption-block-2:METADATA:10.000000": preview,
        "caption-block-2:WHISPER:10.800000": ReviewPreview(
            10.8, 8.8, 12.8, 2.0, preview_file, 4.0
        ),
    }
    page.session_state["source_video_path"] = str(tmp_path / "original.mp4")
    page.session_state["ffmpeg_path"] = "ffmpeg"
    page.session_state["ffprobe_path"] = "ffprobe"
    open_button = next(button for button in page.button if button.label == "Open Human Review")
    page = open_button.click().run()

    assert not page.exception
    assert [title.value for title in page.title] == ["Human Review"]
    assert "← Back to results" in [button.label for button in page.button]
    assert "Use Metadata" in [button.label for button in page.button]
    assert "Use Whisper" in [button.label for button in page.button]
    assert "Preview Manual Time" in [button.label for button in page.button]
    assert "Use Manual Time" in [button.label for button in page.button]
    assert any("TURN" in markdown.value for markdown in page.markdown)
    assert page.number_input[0].label == "Timestamp in original video (seconds)"

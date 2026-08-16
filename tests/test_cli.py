"""Tests for the TimeSync AI command-line entry point."""

from pathlib import Path
from types import SimpleNamespace

import main
from src.pipeline import PipelineError


def test_cli_builds_default_pipeline_config(monkeypatch, capsys, tmp_path: Path) -> None:
    captured = {}

    def run(config):
        captured["config"] = config
        return SimpleNamespace(
            aligned_boundary_count=14,
            executed_clips=(1, 2, 3),
            withheld_boundary_ids=("caption-block-9",),
            total_processing_seconds=12.3456,
            output_directory=tmp_path / "run",
        )

    monkeypatch.setattr(main, "run_pipeline", run)
    code = main.main([
        "--video", "input.mp4",
        "--transcript", "captions.txt",
        "--output", str(tmp_path / "run"),
    ])

    assert code == 0
    assert captured["config"].refinement_enabled is False
    assert captured["config"].whisper_model == "tiny.en"
    output = capsys.readouterr().out
    assert "14 boundaries, 3 clips, 1 withheld" in output
    assert "12.346s total" in output


def test_cli_experimental_refinement_requires_explicit_flag(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def run(config):
        captured["config"] = config
        return SimpleNamespace(
            aligned_boundary_count=0,
            executed_clips=(),
            withheld_boundary_ids=(),
            total_processing_seconds=0.1,
            output_directory=tmp_path / "run",
        )

    monkeypatch.setattr(main, "run_pipeline", run)
    assert main.main([
        "--video", "input.mp4", "--transcript", "captions.txt",
        "--output", str(tmp_path / "run"), "--experimental-refinement",
    ]) == 0
    assert captured["config"].refinement_enabled is True


def test_cli_returns_nonzero_and_clear_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        main,
        "run_pipeline",
        lambda _config: (_ for _ in ()).throw(PipelineError("Validating input", "bad media")),
    )
    code = main.main([
        "--video", "input.mp4", "--transcript", "captions.txt", "--output", "run"
    ])

    assert code == 1
    assert "TimeSync AI failed: Validating input: bad media" in capsys.readouterr().err

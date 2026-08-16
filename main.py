"""Command-line entry point for the TimeSync AI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.pipeline import PipelineConfig, PipelineError, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the stable TimeSync AI command-line interface."""
    parser = argparse.ArgumentParser(description="Run the TimeSync AI pipeline")
    parser.add_argument("--video", required=True, type=Path, help="original input video")
    parser.add_argument(
        "--transcript", required=True, type=Path, help="caption metadata text file"
    )
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--whisper-model", default="small.en", help="local Whisper model")
    parser.add_argument(
        "--whisper-download-root", type=Path, help="local Whisper model directory"
    )
    parser.add_argument(
        "--experimental-refinement",
        action="store_true",
        help="opt in to experimental low-energy execution-boundary refinement",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable path")
    parser.add_argument("--ffprobe", default="ffprobe", help="FFprobe executable path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the pipeline, and return a process exit code."""
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        video_path=args.video,
        transcript_path=args.transcript,
        output_directory=args.output,
        whisper_model=args.whisper_model,
        whisper_download_root=args.whisper_download_root,
        refinement_enabled=args.experimental_refinement,
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe,
    )
    try:
        result = run_pipeline(config)
    except PipelineError as error:
        print(f"TimeSync AI failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Completed: {result.aligned_boundary_count} boundaries, "
        f"{len(result.executed_clips)} clips, "
        f"{len(result.withheld_boundary_ids)} withheld; "
        f"{result.total_processing_seconds:.3f}s total; "
        f"outputs: {result.output_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

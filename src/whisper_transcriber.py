"""Local OpenAI Whisper transcription with word-level timestamps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class WhisperWord:
    """One recognized word with its timing and optional Whisper probability."""

    text: str
    start_seconds: float
    end_seconds: float
    probability: float | None


@dataclass(frozen=True, slots=True)
class WhisperTranscription:
    """Structured output from a local Whisper transcription run."""

    model: str
    language: str | None
    processing_seconds: float
    transcript: str
    words: tuple[WhisperWord, ...]


def transcribe_video(
    video_path: str | Path,
    *,
    model_name: str = "tiny.en",
    download_root: str | Path | None = None,
) -> WhisperTranscription:
    """Transcribe a media file locally and retain word timing probabilities."""
    import torch
    import whisper

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video file not found: {source}")

    load_options: dict[str, Any] = {}
    if download_root is not None:
        load_options["download_root"] = str(download_root)

    model = whisper.load_model(model_name, **load_options)
    started = time.perf_counter()
    result = model.transcribe(
        str(source),
        language="en",
        word_timestamps=True,
        fp16=torch.cuda.is_available(),
        verbose=False,
    )
    processing_seconds = time.perf_counter() - started

    words = tuple(
        WhisperWord(
            text=str(word["word"]).strip(),
            start_seconds=float(word["start"]),
            end_seconds=float(word["end"]),
            probability=(
                float(word["probability"])
                if word.get("probability") is not None
                else None
            ),
        )
        for segment in result.get("segments", [])
        for word in segment.get("words", [])
    )

    return WhisperTranscription(
        model=model_name,
        language=result.get("language"),
        processing_seconds=processing_seconds,
        transcript=str(result.get("text", "")).strip(),
        words=words,
    )


def write_whisper_json(result: WhisperTranscription, path: str | Path) -> None:
    """Write a structured Whisper result as readable UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

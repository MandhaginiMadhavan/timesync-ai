"""Parsing and serialization for timestamped caption metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable


_TIMESTAMP_PATTERN = re.compile(r"^(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d{2})$")


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    """A caption and its start time relative to the beginning of the media."""

    start_seconds: float
    text: str


def parse_timestamp(value: str) -> float:
    """Convert an ``MM:SS`` or ``HH:MM:SS`` caption timestamp to seconds."""
    match = _TIMESTAMP_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid caption timestamp: {value!r}")

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if seconds >= 60 or (match.group("hours") is not None and minutes >= 60):
        raise ValueError(f"invalid caption timestamp: {value!r}")
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_caption_metadata(content: str) -> list[CaptionRecord]:
    """Parse blank-line-separated timestamp, display-time, and caption blocks."""
    records: list[CaptionRecord] = []
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip()) if content.strip() else []

    for block_number, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"caption block {block_number} must contain at least 3 lines")

        start_seconds = parse_timestamp(lines[0])
        caption_text = " ".join(lines[2:]).strip()
        if not caption_text:
            raise ValueError(f"caption block {block_number} has no caption text")

        records.append(CaptionRecord(start_seconds=start_seconds, text=caption_text))

    return records


def parse_caption_file(path: str | Path) -> list[CaptionRecord]:
    """Read and parse a UTF-8 caption metadata file."""
    return parse_caption_metadata(Path(path).read_text(encoding="utf-8-sig"))


def write_caption_json(records: Iterable[CaptionRecord], path: str | Path) -> None:
    """Write caption records as readable UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

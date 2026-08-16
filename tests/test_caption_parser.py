"""Tests for caption metadata parsing."""

from pathlib import Path

import pytest

from src.caption_parser import (
    CaptionRecord,
    parse_caption_file,
    parse_caption_metadata,
    parse_timestamp,
    write_caption_json,
)


SAMPLE_METADATA = """0:05
5 seconds
first caption

1:09
1 minute, 9 seconds
second caption continues
onto another line
"""


def test_parse_caption_blocks() -> None:
    assert parse_caption_metadata(SAMPLE_METADATA) == [
        CaptionRecord(start_seconds=5.0, text="first caption"),
        CaptionRecord(
            start_seconds=69.0,
            text="second caption continues onto another line",
        ),
    ]


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [("0:05", 5.0), ("1:09", 69.0), ("1:02:03", 3723.0)],
)
def test_parse_timestamp(timestamp: str, expected: float) -> None:
    assert parse_timestamp(timestamp) == expected


@pytest.mark.parametrize("timestamp", ["", "five", "1:60", "1:60:00"])
def test_invalid_timestamp_is_rejected(timestamp: str) -> None:
    with pytest.raises(ValueError):
        parse_timestamp(timestamp)


def test_incomplete_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="block 1"):
        parse_caption_metadata("0:05\ncaption without display time")


def test_file_parse_and_json_write(tmp_path: Path) -> None:
    source = tmp_path / "captions.txt"
    destination = tmp_path / "captions.json"
    source.write_text(SAMPLE_METADATA, encoding="utf-8")

    records = parse_caption_file(source)
    write_caption_json(records, destination)

    assert records[0].start_seconds == 5.0
    assert '"start_seconds": 5.0' in destination.read_text(encoding="utf-8")

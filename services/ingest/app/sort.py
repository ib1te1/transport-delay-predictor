"""Bounded-memory sort of traffic.csv by dataset event time."""

import csv
import heapq
import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO
from zoneinfo import ZoneInfo

from app.normalize import parse_time

REQUIRED_COLUMNS = {
    "unit_id",
    "tr_id",
    "event_time",
    "location_valid",
    "lat",
    "lon",
    "speed",
    "heading",
}


@dataclass(frozen=True)
class SortedCsvRow:
    event_time: datetime
    line_number: int
    values: dict[str, str | None]

    @property
    def key(self) -> tuple[datetime, int]:
        return self.event_time, self.line_number


def _write_chunk(rows: list[SortedCsvRow], directory: Path, number: int) -> Path:
    rows.sort(key=lambda row: row.key)
    path = directory / f"chunk-{number:05d}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps([row.event_time.isoformat(), row.line_number, row.values]))
            file.write("\n")
    return path


def _read_chunk(file: TextIO) -> Iterator[SortedCsvRow]:
    for line in file:
        event_time, line_number, values = json.loads(line)
        yield SortedCsvRow(datetime.fromisoformat(event_time), line_number, values)


def sorted_csv_rows(
    path: Path,
    source_zone: ZoneInfo,
    chunk_rows: int,
    *,
    on_bad_time: Callable[[int], None] | None = None,
    on_row: Callable[[], None] | None = None,
) -> Iterator[SortedCsvRow]:
    """Yield rows globally ordered by (event_time, physical CSV line).

    Rows without a parseable time cannot be ordered and are reported to
    ``on_bad_time``. Temporary chunks are removed when iteration ends.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    with (
        TemporaryDirectory(prefix="ingest-sort-") as tmp,
        path.open(encoding="utf-8", newline="") as source,
    ):
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing traffic.csv columns: {', '.join(sorted(missing))}")
        directory = Path(tmp)
        chunks: list[Path] = []
        rows: list[SortedCsvRow] = []
        for values in reader:
            if on_row is not None:
                on_row()
            line_number = reader.line_num
            event_time = parse_time(values.get("event_time") or "", source_zone)
            if event_time is None:
                if on_bad_time is not None:
                    on_bad_time(line_number)
                continue
            rows.append(SortedCsvRow(event_time, line_number, values))
            if len(rows) == chunk_rows:
                chunks.append(_write_chunk(rows, directory, len(chunks)))
                rows = []
        if not chunks:
            yield from sorted(rows, key=lambda row: row.key)
            return
        if rows:
            chunks.append(_write_chunk(rows, directory, len(chunks)))
        with ExitStack() as stack:
            files = [stack.enter_context(chunk.open(encoding="utf-8")) for chunk in chunks]
            yield from heapq.merge(*(_read_chunk(file) for file in files), key=lambda row: row.key)

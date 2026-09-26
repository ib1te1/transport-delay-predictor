"""Bounded-memory sort of traffic.csv by dataset event time."""

import csv
import heapq
import json
import logging
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

from app.normalize import parse_time

log = logging.getLogger(__name__)

SORTED_NAME = "sorted.jsonl"
BAD_TIME_LOG_LIMIT = 20

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


class SortCancelled(Exception):
    """The caller asked the sort to stop before it finished."""


@dataclass(frozen=True)
class SortResult:
    path: Path
    read: int
    bad_time: int
    selected: int
    to_play: int


def _line(row: SortedCsvRow) -> str:
    return json.dumps([row.event_time.isoformat(), row.line_number, row.values]) + "\n"


def _write_chunk(rows: list[SortedCsvRow], directory: Path, number: int) -> Path:
    rows.sort(key=lambda row: row.key)
    path = directory / f"chunk-{number:05d}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(_line(row))
    return path


def _check_stop(stop: threading.Event | None) -> None:
    if stop is not None and stop.is_set():
        raise SortCancelled


def _read_chunk(file: TextIO) -> Iterator[SortedCsvRow]:
    for line in file:
        event_time, line_number, values = json.loads(line)
        yield SortedCsvRow(datetime.fromisoformat(event_time), line_number, values)


def sort_to_file(
    path: Path,
    source_zone: ZoneInfo,
    chunk_rows: int,
    out_dir: Path,
    *,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
    after: tuple[datetime, int] | None = None,
    stop: threading.Event | None = None,
) -> SortResult:
    """Write the rows still to be played to ``out_dir/sorted.jsonl``.

    Rows are ordered by (event_time, physical CSV line). Only rows inside the
    inclusive [start_at, end_at] range and after the ``after`` cursor are
    written; chunk files are removed after the merge. Meant to run in a
    worker thread: once ``stop`` is set the next row raises SortCancelled.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    read = bad_time = selected = to_play = 0
    chunks: list[Path] = []
    pending: list[SortedCsvRow] = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing traffic.csv columns: {', '.join(sorted(missing))}")
        for values in reader:
            _check_stop(stop)
            read += 1
            line_number = reader.line_num
            event_time = parse_time(values.get("event_time") or "", source_zone)
            if event_time is None:
                bad_time += 1
                if bad_time <= BAD_TIME_LOG_LIMIT:
                    log.warning("skipping CSV line %d: invalid event_time", line_number)
                continue
            if start_at is not None and event_time < start_at:
                continue
            if end_at is not None and event_time > end_at:
                continue
            selected += 1
            row = SortedCsvRow(event_time, line_number, values)
            if after is not None and row.key <= after:
                continue
            to_play += 1
            pending.append(row)
            if len(pending) == chunk_rows:
                chunks.append(_write_chunk(pending, out_dir, len(chunks)))
                pending = []
    if pending:
        chunks.append(_write_chunk(pending, out_dir, len(chunks)))
    target = out_dir / SORTED_NAME
    with ExitStack() as stack:
        files = [stack.enter_context(chunk.open(encoding="utf-8")) for chunk in chunks]
        with target.open("w", encoding="utf-8", newline="\n") as out:
            merged = heapq.merge(*(_read_chunk(file) for file in files), key=lambda row: row.key)
            for row in merged:
                _check_stop(stop)
                out.write(_line(row))
    for chunk in chunks:
        chunk.unlink()
    return SortResult(target, read, bad_time, selected, to_play)


def read_sorted(path: Path) -> Iterator[SortedCsvRow]:
    """Yield rows written by sort_to_file; close the generator to release the file."""
    with path.open(encoding="utf-8") as file:
        yield from _read_chunk(file)

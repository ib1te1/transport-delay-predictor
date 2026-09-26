import threading
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.sort import SortCancelled, read_sorted, sort_to_file

ZONE = ZoneInfo("UTC")
HEADER = "unit_id,tr_id,event_time,location_valid,lat,lon,speed,heading\n"
UNSORTED = (
    HEADER
    + "1,2,2026-01-06 00:00:03,False,,,,\n"
    + "1,2,2026-01-06 00:00:01,False,,,,\n"
    + "1,2,2026-01-06 00:00:02,False,,,,\n"
    + "1,2,2026-01-06 00:00:02,False,,,,\n"
)


def at(second: int) -> datetime:
    return datetime(2026, 1, 6, 0, 0, second, tzinfo=UTC)


def write_traffic(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "traffic.csv"
    path.write_text(text, encoding="utf-8")
    return path


def work_dir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


class StopAfter(threading.Event):
    """Reports itself set once it has been checked more than ``calls`` times."""

    def __init__(self, calls: int) -> None:
        super().__init__()
        self.calls = calls

    def is_set(self) -> bool:
        self.calls -= 1
        return self.calls < 0


def test_chunks_merge_into_one_sorted_file_and_preserve_ties(tmp_path: Path) -> None:
    out = work_dir(tmp_path)
    result = sort_to_file(write_traffic(tmp_path, UNSORTED), ZONE, 2, out)
    rows = list(read_sorted(result.path))
    assert [row.line_number for row in rows] == [3, 4, 5, 2]
    assert result.path == out / "sorted.jsonl"
    assert sorted(path.name for path in out.iterdir()) == ["sorted.jsonl"]
    assert (result.read, result.bad_time, result.selected, result.to_play) == (4, 0, 4, 4)


def test_single_partial_chunk_goes_through_the_same_file(tmp_path: Path) -> None:
    out = work_dir(tmp_path)
    result = sort_to_file(write_traffic(tmp_path, UNSORTED), ZONE, 100, out)
    assert [row.line_number for row in read_sorted(result.path)] == [3, 4, 5, 2]
    assert sorted(path.name for path in out.iterdir()) == ["sorted.jsonl"]


def test_range_and_cursor_are_applied_before_sorting(tmp_path: Path) -> None:
    traffic = write_traffic(
        tmp_path,
        HEADER
        + "1,2,2026-01-06 00:00:04,False,,,,\n"
        + "1,2,2026-01-06 00:00:01,False,,,,\n"
        + "1,2,2026-01-06 00:00:03,False,,,,\n"
        + "1,2,2026-01-06 00:00:02,False,,,,\n",
    )
    result = sort_to_file(
        traffic, ZONE, 2, work_dir(tmp_path), start_at=at(2), end_at=at(3), after=(at(2), 5)
    )
    assert [row.line_number for row in read_sorted(result.path)] == [4]
    assert (result.read, result.bad_time, result.selected, result.to_play) == (4, 0, 2, 1)


def test_bad_time_is_counted_without_stopping_the_sort(tmp_path: Path) -> None:
    traffic = write_traffic(
        tmp_path, HEADER + "1,2,no-date,False,,,,\n" + "1,2,2026-01-06 00:00:01,False,,,,\n"
    )
    result = sort_to_file(traffic, ZONE, 1, work_dir(tmp_path))
    assert (result.read, result.bad_time, result.selected) == (2, 1, 1)
    assert [row.line_number for row in read_sorted(result.path)] == [3]


def test_missing_header_fails_before_anything_is_written(tmp_path: Path) -> None:
    traffic = write_traffic(tmp_path, "unit_id,event_time\n1,2026-01-06 00:00:01\n")
    out = work_dir(tmp_path)
    with pytest.raises(ValueError, match="missing traffic.csv columns"):
        sort_to_file(traffic, ZONE, 1, out)
    assert not any(out.iterdir())


def test_stop_request_cancels_the_sort(tmp_path: Path) -> None:
    with pytest.raises(SortCancelled):
        sort_to_file(
            write_traffic(tmp_path, UNSORTED), ZONE, 10, work_dir(tmp_path), stop=StopAfter(2)
        )

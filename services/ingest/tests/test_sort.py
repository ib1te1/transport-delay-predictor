from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.sort import sorted_csv_rows

ZONE = ZoneInfo("UTC")
HEADER = "unit_id,tr_id,event_time,location_valid,lat,lon,speed,heading\n"


def test_external_sort_orders_unsorted_rows_and_preserves_ties(tmp_path: Path) -> None:
    traffic = tmp_path / "traffic.csv"
    traffic.write_text(
        HEADER
        + "1,2,2026-01-06 00:00:03,False,,,,\n"
        + "1,2,2026-01-06 00:00:01,False,,,,\n"
        + "1,2,2026-01-06 00:00:02,False,,,,\n"
        + "1,2,2026-01-06 00:00:02,False,,,,\n",
        encoding="utf-8",
    )
    rows = list(sorted_csv_rows(traffic, ZONE, chunk_rows=2))
    assert [row.line_number for row in rows] == [3, 4, 5, 2]
    assert [row.event_time.timestamp() for row in rows] == sorted(
        row.event_time.timestamp() for row in rows
    )


def test_bad_time_is_reported_without_stopping_the_sort(tmp_path: Path) -> None:
    traffic = tmp_path / "traffic.csv"
    traffic.write_text(
        HEADER + "1,2,no-date,False,,,,\n" + "1,2,2026-01-06 00:00:01,False,,,,\n",
        encoding="utf-8",
    )
    bad: list[int] = []
    rows = list(sorted_csv_rows(traffic, ZONE, chunk_rows=1, on_bad_time=bad.append))
    assert bad == [2]
    assert [row.line_number for row in rows] == [3]


def test_missing_header_fails_before_any_row_is_yielded(tmp_path: Path) -> None:
    traffic = tmp_path / "traffic.csv"
    traffic.write_text("unit_id,event_time\n1,2026-01-06 00:00:01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing traffic.csv columns"):
        list(sorted_csv_rows(traffic, ZONE, chunk_rows=1))

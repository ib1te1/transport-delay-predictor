from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.config import ReplayConfig
from app.replay import play_sorted_rows
from app.store import RunState
from contracts import TelemetryRecord

HEADER = "unit_id,tr_id,event_time,location_valid,lat,lon,speed,heading\n"
ZONE = ZoneInfo("UTC")


def traffic_file(tmp_path: Path) -> Path:
    path = tmp_path / "traffic.csv"
    path.write_text(
        HEADER
        + "1,2,2026-01-06 00:00:03,False,,,,\n"
        + "1,2,2026-01-06 00:00:01,False,,,,\n"
        + "1,2,2026-01-06 00:00:02,False,,,,\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.anyio
async def test_replay_paces_sorted_points_without_changing_dataset_time(tmp_path: Path) -> None:
    path = traffic_file(tmp_path)
    saved: list[tuple[int, TelemetryRecord, tuple[datetime, int]]] = []
    delays: list[float] = []

    async def save(line: int, record: TelemetryRecord, cursor: tuple[datetime, int]) -> int:
        saved.append((line, record, cursor))
        return len(saved)

    async def skip(_cursor: tuple[datetime, int]) -> None:
        raise AssertionError("no row should be skipped")

    async def delivered(_row_id: int) -> None:
        pass

    async def sleep(delay: float) -> None:
        delays.append(delay)

    stats = await play_sorted_rows(
        path,
        ReplayConfig(speedup=2, sort_chunk_rows=2),
        ZONE,
        {1: 2},
        RunState(uuid4(), "active", None, None, None),
        save=save,
        skip=skip,
        delivered=delivered,
        sleep=sleep,
    )
    assert [line for line, _, _ in saved] == [3, 4, 2]
    assert [record.event_time.second for _, record, _ in saved] == [1, 2, 3]
    assert delays == [0.5, 0.5]
    assert stats.read == 3
    assert stats.published == 3


@pytest.mark.anyio
async def test_resume_starts_after_the_last_stored_sorted_key(tmp_path: Path) -> None:
    path = traffic_file(tmp_path)
    saved: list[int] = []

    async def save(line: int, _record: TelemetryRecord, _cursor: tuple[datetime, int]) -> int:
        saved.append(line)
        return line

    async def skip(_cursor: tuple[datetime, int]) -> None:
        pass

    async def delivered(_row_id: int) -> None:
        pass

    async def sleep(_delay: float) -> None:
        pass

    run = RunState(uuid4(), "active", datetime(2026, 1, 6, 0, 0, 2, tzinfo=UTC), 4, None)
    stats = await play_sorted_rows(
        path,
        ReplayConfig(sort_chunk_rows=2),
        ZONE,
        {1: 2},
        run,
        save=save,
        skip=skip,
        delivered=delivered,
        sleep=sleep,
    )
    assert saved == [2]
    assert stats.published == 1


@pytest.mark.anyio
async def test_read_count_covers_rows_after_the_selected_range(tmp_path: Path) -> None:
    path = traffic_file(tmp_path)
    saved: list[int] = []

    async def save(line: int, _record: TelemetryRecord, _cursor: tuple[datetime, int]) -> int:
        saved.append(line)
        return line

    async def skip(_cursor: tuple[datetime, int]) -> None:
        pass

    async def delivered(_row_id: int) -> None:
        pass

    stats = await play_sorted_rows(
        path,
        ReplayConfig(end_at=datetime(2026, 1, 6, 0, 0, 1, tzinfo=UTC)),
        ZONE,
        {1: 2},
        RunState(uuid4(), "active", None, None, None),
        save=save,
        skip=skip,
        delivered=delivered,
    )
    assert saved == [3]
    assert stats.read == 3

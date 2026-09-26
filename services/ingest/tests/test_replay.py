import asyncio
import logging
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.config import ReplayConfig
from app.replay import (
    ReplayStats,
    file_sha256,
    main,
    play_sorted_rows,
    remove_stale_sort_dirs,
    run_replay,
    sort_and_play,
    sort_in_thread,
)
from app.sort import SortCancelled, SortedCsvRow, read_sorted, sort_to_file
from app.store import ReplayCompleted
from common.config import ServiceSettings
from common.db import connect
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


def work_dir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


class Sink:
    """Stands in for Postgres and the outbox: records what playback stores."""

    def __init__(self) -> None:
        self.saved: list[tuple[int, TelemetryRecord, tuple[datetime, int]]] = []
        self.skipped: list[tuple[datetime, int]] = []

    async def save(self, line: int, record: TelemetryRecord, cursor: tuple[datetime, int]) -> int:
        self.saved.append((line, record, cursor))
        return len(self.saved)

    async def skip(self, cursor: tuple[datetime, int]) -> None:
        self.skipped.append(cursor)

    async def delivered(self, _row_id: int) -> None:
        pass

    @property
    def lines(self) -> list[int]:
        return [line for line, _, _ in self.saved]


async def no_sleep(_delay: float) -> None:
    pass


class FakeClock:
    """Monotonic time that moves only when playback sleeps or saves."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        assert delay > 0
        self.sleeps.append(delay)
        self.now += delay


def point(second: int, line: int, unit_id: str = "1") -> SortedCsvRow:
    return SortedCsvRow(
        datetime(2026, 1, 6, 0, 0, second, tzinfo=UTC),
        line,
        {
            "unit_id": unit_id,
            "tr_id": "2",
            "event_time": f"2026-01-06 00:00:{second:02d}",
            "location_valid": "False",
            "lat": "",
            "lon": "",
            "speed": "",
            "heading": "",
        },
    )


async def play_timed(
    rows: list[SortedCsvRow], clock: FakeClock, costs: list[float]
) -> tuple[list[float], ReplayStats]:
    """Play at speedup 1; each save advances the clock by the next cost."""
    published: list[float] = []
    cost = iter(costs)

    async def save(line: int, _record: TelemetryRecord, _cursor: tuple[datetime, int]) -> int:
        published.append(clock.now)
        clock.now += next(cost)
        return line

    async def skip(_cursor: tuple[datetime, int]) -> None:
        pass

    async def delivered(_row_id: int) -> None:
        pass

    stats = await play_sorted_rows(
        rows,
        ReplayConfig(speedup=1),
        ZONE,
        {1: 2},
        save=save,
        skip=skip,
        delivered=delivered,
        sleep=clock.sleep,
        clock=clock,
    )
    return published, stats


@pytest.mark.anyio
async def test_replay_paces_sorted_points_without_changing_dataset_time(tmp_path: Path) -> None:
    result = sort_to_file(traffic_file(tmp_path), ZONE, 2, work_dir(tmp_path))
    sink = Sink()
    clock = FakeClock()
    stats = await play_sorted_rows(
        read_sorted(result.path),
        ReplayConfig(speedup=2, sort_chunk_rows=2),
        ZONE,
        {1: 2},
        save=sink.save,
        skip=sink.skip,
        delivered=sink.delivered,
        sleep=clock.sleep,
        clock=clock,
    )
    assert sink.lines == [3, 4, 2]
    assert [record.event_time.second for _, record, _ in sink.saved] == [1, 2, 3]
    assert clock.sleeps == [0.5, 0.5]
    assert sink.skipped == []
    assert stats.published == 3


@pytest.mark.anyio
async def test_resume_starts_after_the_last_stored_sorted_key(tmp_path: Path) -> None:
    sink = Sink()
    stats = await sort_and_play(
        traffic_file(tmp_path),
        ReplayConfig(sort_chunk_rows=2),
        ZONE,
        {1: 2},
        work_dir(tmp_path),
        after=(datetime(2026, 1, 6, 0, 0, 2, tzinfo=UTC), 4),
        save=sink.save,
        skip=sink.skip,
        delivered=sink.delivered,
        sleep=no_sleep,
    )
    assert sink.lines == [2]
    assert (stats.read, stats.published) == (3, 1)


@pytest.mark.anyio
async def test_read_count_covers_rows_after_the_selected_range(tmp_path: Path) -> None:
    sink = Sink()
    stats = await sort_and_play(
        traffic_file(tmp_path),
        ReplayConfig(end_at=datetime(2026, 1, 6, 0, 0, 1, tzinfo=UTC)),
        ZONE,
        {1: 2},
        work_dir(tmp_path),
        after=None,
        save=sink.save,
        skip=sink.skip,
        delivered=sink.delivered,
        sleep=no_sleep,
    )
    assert sink.lines == [3]
    assert stats.read == 3


@pytest.mark.anyio
async def test_empty_range_fails_before_publishing(tmp_path: Path) -> None:
    sink = Sink()
    with pytest.raises(ValueError, match="contains no telemetry"):
        await sort_and_play(
            traffic_file(tmp_path),
            ReplayConfig(start_at=datetime(2026, 1, 7, tzinfo=UTC)),
            ZONE,
            {1: 2},
            work_dir(tmp_path),
            after=None,
            save=sink.save,
            skip=sink.skip,
            delivered=sink.delivered,
            sleep=no_sleep,
        )
    assert sink.saved == []


@pytest.mark.anyio
async def test_nothing_left_after_the_cursor_finishes_without_error(tmp_path: Path) -> None:
    sink = Sink()
    stats = await sort_and_play(
        traffic_file(tmp_path),
        ReplayConfig(),
        ZONE,
        {1: 2},
        work_dir(tmp_path),
        after=(datetime(2026, 1, 6, 0, 0, 3, tzinfo=UTC), 2),
        save=sink.save,
        skip=sink.skip,
        delivered=sink.delivered,
        sleep=no_sleep,
    )
    assert sink.saved == []
    assert stats.published == 0


@pytest.mark.anyio
async def test_cancelling_the_sort_stops_its_thread_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def blocking_sort(path, source_zone, chunk_rows, out_dir, *, start_at, end_at, after, stop):
        started.set()
        if not stop.wait(5):
            raise AssertionError("the sort was never asked to stop")
        stopped.set()
        raise SortCancelled

    monkeypatch.setattr("app.replay.sort_to_file", blocking_sort)
    task = asyncio.create_task(
        sort_in_thread(
            tmp_path / "traffic.csv", ZONE, 10, tmp_path, start_at=None, end_at=None, after=None
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    begun = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - begun < 1
    assert stopped.is_set()


def test_stale_sort_dirs_are_removed(tmp_path: Path) -> None:
    stale = tmp_path / "ingest-replay-abc123"
    stale.mkdir()
    (stale / "chunk-00000.jsonl").write_text("[]\n", encoding="utf-8")
    other = tmp_path / "keep-me"
    other.mkdir()
    remove_stale_sort_dirs(tmp_path)
    assert not stale.exists()
    assert other.exists()


@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_cancel_during_sort_marks_the_run_interrupted_and_removes_sort_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database_url: str, redis_url: str
) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "test").mkdir(parents=True)
    traffic = data_dir / "test" / "traffic.csv"
    traffic.write_text(HEADER + "1,2,2026-01-06 00:00:01,False,,,,\n", encoding="utf-8")
    config = tmp_path / "system.yaml"
    config.write_text(
        "dataset:\n  source_timezone: UTC\n"
        "ingest:\n  mode: replay\n"
        "seed:\n  period: test\n"
        "replay:\n  period: test\n",
        encoding="utf-8",
    )
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    started = threading.Event()

    def blocking_sort(path, source_zone, chunk_rows, out_dir, *, start_at, end_at, after, stop):
        (out_dir / "chunk-00000.jsonl").write_text("", encoding="utf-8")
        started.set()
        if not stop.wait(5):
            raise AssertionError("the sort was never asked to stop")
        raise SortCancelled

    monkeypatch.setattr("app.replay.sort_to_file", blocking_sort)
    sha = file_sha256(traffic)
    with connect(database_url) as conn:
        existing = conn.execute("SELECT count(*) FROM ingest_runs WHERE mode = 'replay'").fetchone()
    assert existing == (0,), "this test needs a database without replay runs"
    settings = ServiceSettings(
        database_url=database_url, redis_url=redis_url, config_path=config, data_dir=data_dir
    )
    task = asyncio.create_task(run_replay(settings))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        begun = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - begun < 1
        with connect(database_url) as conn:
            status = conn.execute(
                "SELECT status FROM ingest_runs WHERE mode = 'replay' AND file_sha256 = %s",
                (sha,),
            ).fetchone()
        assert status == ("interrupted",)
        assert not list(temp_root.glob("ingest-replay-*"))
    finally:
        with connect(database_url) as conn:
            conn.execute(
                "DELETE FROM telemetry WHERE run_id IN"
                " (SELECT run_id FROM ingest_runs WHERE file_sha256 = %s)",
                (sha,),
            )
            conn.execute("DELETE FROM ingest_runs WHERE file_sha256 = %s", (sha,))


@pytest.mark.anyio
async def test_processing_time_is_absorbed_by_the_next_pause() -> None:
    clock = FakeClock()
    published, _ = await play_timed([point(0, 2), point(1, 3), point(2, 4)], clock, [0.3] * 3)
    assert published == pytest.approx([100.0, 101.0, 102.0])
    assert clock.sleeps == pytest.approx([0.7, 0.7])
    assert clock.now == pytest.approx(102.3)


@pytest.mark.anyio
async def test_late_rows_go_out_at_once_until_back_on_schedule() -> None:
    clock = FakeClock()
    rows = [point(second, second + 2) for second in range(5)]
    published, _ = await play_timed(rows, clock, [2.0, 0.1, 0.1, 0.1, 0.1])
    assert published == pytest.approx([100.0, 102.0, 102.1, 103.0, 104.0])
    assert clock.sleeps == pytest.approx([0.8, 0.9])


@pytest.mark.anyio
async def test_lag_is_logged_at_most_once_a_minute(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    rows = [point(second, second + 2) for second in range(4)]
    with caplog.at_level(logging.INFO, logger="app.replay"):
        await play_timed(rows, clock, [3.0] * 4)
    lag = [r.getMessage() for r in caplog.records if "replay lag_sec=" in r.getMessage()]
    assert lag == ["replay lag_sec=2.0"]
    assert clock.sleeps == []


@pytest.mark.anyio
async def test_invalid_row_neither_waits_nor_sets_the_anchor() -> None:
    clock = FakeClock()
    rows = [point(0, 2, unit_id="x"), point(5, 3), point(6, 4)]
    published, stats = await play_timed(rows, clock, [0.0, 0.0])
    assert published == pytest.approx([100.0, 101.0])
    assert clock.sleeps == [1.0]
    assert stats.skipped == {"unit_id": 1}


@pytest.mark.anyio
async def test_resume_anchors_on_the_first_row_after_the_cursor(tmp_path: Path) -> None:
    traffic = tmp_path / "traffic.csv"
    traffic.write_text(
        HEADER
        + "1,2,2026-01-06 00:00:01,False,,,,\n"
        + "1,2,2026-01-06 00:00:02,False,,,,\n"
        + "1,2,2026-01-06 00:00:03,False,,,,\n"
        + "1,2,2026-01-06 00:00:04,False,,,,\n",
        encoding="utf-8",
    )
    sink = Sink()
    clock = FakeClock()
    await sort_and_play(
        traffic,
        ReplayConfig(speedup=1),
        ZONE,
        {1: 2},
        work_dir(tmp_path),
        after=(datetime(2026, 1, 6, 0, 0, 2, tzinfo=UTC), 3),
        save=sink.save,
        skip=sink.skip,
        delivered=sink.delivered,
        sleep=clock.sleep,
        clock=clock,
    )
    assert sink.lines == [4, 5]
    assert clock.sleeps == [1.0]


def stub_run(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Keep main() away from real settings, databases and dataset files."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused@127.0.0.1:1/unused")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")

    async def fake_run(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("app.replay._run_with_signals", fake_run)


def test_removed_new_run_flag_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_run(monkeypatch, AssertionError("replay must not start"))
    with pytest.raises(SystemExit) as caught:
        main(["--new-run"])
    assert caught.value.code == 2


def test_completed_run_exits_with_code_3_and_points_to_reset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run_id = uuid4()
    stub_run(monkeypatch, ReplayCompleted(run_id))
    with caplog.at_level(logging.WARNING, logger="app.replay"):
        assert main([]) == 3
    assert f"replay run {run_id} is already completed" in caplog.text
    assert "scripts/reset-demo.sh (or scripts/reset-demo.ps1)" in caplog.text

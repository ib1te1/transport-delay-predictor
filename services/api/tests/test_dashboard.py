import asyncio
import json
import random
from datetime import timedelta

import pytest
from factories import at, plan_stop, prediction_row, stop_event, telemetry_record
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import ApiConfig
from app.dashboard import Dashboard, StateUnavailable, VehicleNotFound
from app.live import LiveState
from app.state import FleetState

CONFIG = ApiConfig()
WINDOW = timedelta(seconds=CONFIG.request.telemetry_window_sec)
PLAN = {7: [plan_stop(7, 20, at(720), address="Lenina 1")], 9: [plan_stop(9, 30, at(600))]}


class Stop(Exception):
    pass


class Bus:
    """Stands in for the Redis client: records what is published, can fail or hang."""

    def __init__(self, *, fail_on=(), hang_on=(), jitter: bool = False) -> None:
        self.fail_on = set(fail_on)
        self.hang_on = set(hang_on)
        self.jitter = jitter
        self.calls = 0
        self.channels: set[str] = set()
        self.messages: list[dict] = []

    async def publish(self, channel: str, message: str) -> int:
        self.calls += 1
        if self.calls in self.fail_on:
            raise RedisConnectionError("down")
        if self.calls in self.hang_on:
            await asyncio.sleep(3600)
        if self.jitter:
            await asyncio.sleep(random.random() / 1000)
        self.channels.add(channel)
        self.messages.append(json.loads(message))
        return 1

    def seqs(self) -> list[int]:
        return [m["data"]["seq"] for m in self.messages]

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]


def attached(bus: Bus, *points, **kwargs) -> Dashboard:
    fleet = FleetState(WINDOW)
    for point in points:
        fleet.add_telemetry(point)
    dashboard = Dashboard(bus, CONFIG, channel="test-channel", **kwargs)
    dashboard.attach(LiveState(fleet, PLAN))
    return dashboard


def row_at(seconds: int, **overrides):
    values = {"sample_id": f"7_{seconds}", "tr_id": 7, "t": at(seconds), "target_stop_id": 20}
    return prediction_row(**(values | overrides))


@pytest.mark.anyio
async def test_before_the_loop_is_up_the_snapshot_is_empty_and_nothing_is_published() -> None:
    bus = Bus()
    dashboard = Dashboard(bus, CONFIG)

    await dashboard.refresh()
    snapshot = dashboard.snapshot()

    assert bus.messages == []
    assert (snapshot.seq, snapshot.clock, snapshot.vehicles, snapshot.alerts) == (0, None, [], [])
    assert (snapshot.live_mae_s, snapshot.checked_predictions) == (None, 0)
    assert snapshot.summary.risk.model_dump() == {"green": 0, "yellow": 0, "red": 0, "none": 0}


@pytest.mark.anyio
async def test_refresh_publishes_the_clock_then_only_the_vehicles_that_changed() -> None:
    bus = Bus()
    dashboard = attached(bus, telemetry_record(7, at(0)), telemetry_record(8, at(10)))

    await dashboard.refresh()
    await dashboard.refresh()
    dashboard.live.fleet.add_telemetry(telemetry_record(7, at(20)))
    await dashboard.refresh()

    assert bus.types() == ["clock", "vehicles", "clock", "clock", "vehicles"]
    assert bus.seqs() == [1, 2, 3, 4, 5]
    first, second = bus.messages[1]["data"], bus.messages[4]["data"]
    assert [v["tr_id"] for v in first["vehicles"]] == [7, 8]
    assert (first["clock"], first["removed"]) == ("2026-01-06T08:00:10Z", [])
    assert [v["tr_id"] for v in second["vehicles"]] == [7]
    assert second["vehicles"][0]["last_seen"] == "2026-01-06T08:00:20Z"
    assert bus.channels == {"test-channel"}


@pytest.mark.anyio
async def test_a_vehicle_that_left_the_window_is_reported_removed() -> None:
    bus = Bus()
    dashboard = attached(bus, telemetry_record(7, at(0)), telemetry_record(8, at(1800)))
    await dashboard.refresh()

    dashboard.live.fleet.add_telemetry(telemetry_record(8, at(1801)))
    await dashboard.refresh()

    last = bus.messages[-1]
    assert (last["type"], last["data"]["removed"]) == ("vehicles", [7])
    assert [v["tr_id"] for v in last["data"]["vehicles"]] == [8]
    assert [v.tr_id for v in dashboard.snapshot().vehicles] == [8]


@pytest.mark.anyio
async def test_a_new_prediction_is_published_and_reaches_the_map_with_the_next_diff() -> None:
    bus = Bus()
    dashboard = attached(bus, telemetry_record(7, at(0)))
    await dashboard.refresh()

    await dashboard.publish_predictions([row_at(0, risk_level="red")])
    await dashboard.refresh()

    assert bus.types() == ["clock", "vehicles", "prediction", "clock", "vehicles"]
    prediction = bus.messages[2]["data"]
    assert (prediction["seq"], prediction["tr_id"]) == (3, 7)
    assert prediction["prediction"]["sample_id"] == "7_0"
    assert prediction["prediction"]["target_address"] == "Lenina 1"
    [vehicle] = bus.messages[4]["data"]["vehicles"]
    assert vehicle["prediction"]["risk_level"] == "red"
    assert dashboard.snapshot().summary.risk.red == 1


@pytest.mark.anyio
async def test_seq_ascends_across_types_when_two_tasks_publish_at_once() -> None:
    bus = Bus(jitter=True)
    dashboard = attached(bus, telemetry_record(7, at(0)))

    async def predictions() -> None:
        for n in range(20):
            await dashboard.publish_predictions([row_at(n)])

    async def refreshes() -> None:
        for _ in range(20):
            await dashboard.refresh()

    await asyncio.gather(predictions(), refreshes())

    assert bus.seqs() == list(range(1, len(bus.messages) + 1))
    assert {"clock", "vehicles", "prediction"} <= set(bus.types())
    assert dashboard.seq == len(bus.messages)


@pytest.mark.anyio
async def test_a_failed_publication_spends_its_number() -> None:
    bus = Bus(fail_on={2})
    dashboard = attached(bus, telemetry_record(7, at(0)))

    await dashboard.publish_predictions([row_at(0), row_at(1), row_at(2)])

    assert bus.seqs() == [1, 3]
    assert dashboard.seq == dashboard.snapshot().seq == 3


@pytest.mark.anyio
@pytest.mark.timeout(10)
async def test_a_hung_publication_is_given_up_and_spends_its_number() -> None:
    bus = Bus(hang_on={1})
    dashboard = attached(bus, telemetry_record(7, at(0)), publish_timeout=0.05)

    await dashboard.refresh()

    assert bus.types() == ["vehicles"]
    assert bus.seqs() == [2]


@pytest.mark.anyio
async def test_a_diff_that_failed_to_publish_is_still_in_the_snapshot() -> None:
    bus = Bus(fail_on={2})
    dashboard = attached(bus, telemetry_record(7, at(0)))

    await dashboard.refresh()

    snapshot = dashboard.snapshot()
    assert bus.types() == ["clock"]
    assert (snapshot.seq, [v.tr_id for v in snapshot.vehicles]) == (2, [7])


@pytest.mark.anyio
async def test_snapshot_is_the_last_published_view_not_a_fresh_one() -> None:
    bus = Bus()
    dashboard = attached(bus, telemetry_record(7, at(0)))
    await dashboard.refresh()

    dashboard.live.fleet.add_telemetry(telemetry_record(8, at(30)))
    snapshot = dashboard.snapshot()

    assert (snapshot.seq, snapshot.clock) == (2, at(0))
    assert [v.tr_id for v in snapshot.vehicles] == [7]
    assert snapshot.summary.freshness.active == 1


@pytest.mark.anyio
async def test_a_new_loop_run_keeps_the_published_view_until_its_own_diff() -> None:
    bus = Bus()
    dashboard = attached(bus, telemetry_record(7, at(0)), telemetry_record(8, at(0)))
    await dashboard.refresh()

    rebuilt = FleetState(WINDOW)
    rebuilt.add_telemetry(telemetry_record(8, at(5)))
    dashboard.attach(LiveState(rebuilt, PLAN))
    assert [v.tr_id for v in dashboard.snapshot().vehicles] == [7, 8]

    await dashboard.refresh()

    assert bus.messages[-1]["data"]["removed"] == [7]
    assert [v["tr_id"] for v in bus.messages[-1]["data"]["vehicles"]] == [8]


@pytest.mark.anyio
async def test_run_refreshes_once_a_period() -> None:
    bus = Bus()
    dashboard = attached(bus, telemetry_record(7, at(0)))
    periods: list[float] = []

    async def sleep(period: float) -> None:
        periods.append(period)
        if len(periods) == 3:
            raise Stop

    with pytest.raises(Stop):
        await dashboard.run(sleep=sleep)

    assert periods == [1.0, 1.0, 1.0]
    assert bus.types().count("clock") == 3


async def no_predictions(tr_id, since):
    return []


@pytest.mark.anyio
async def test_publishing_predictions_before_the_loop_is_up_is_an_error() -> None:
    with pytest.raises(StateUnavailable):
        await Dashboard(Bus(), CONFIG).publish_predictions([row_at(0)])


@pytest.mark.anyio
async def test_card_before_the_loop_is_up_is_unavailable() -> None:
    with pytest.raises(StateUnavailable):
        await Dashboard(Bus(), CONFIG).card(7, no_predictions)


@pytest.mark.anyio
async def test_card_of_a_vehicle_neither_planned_nor_seen_is_not_found() -> None:
    dashboard = attached(Bus(), telemetry_record(8, at(0)))
    await dashboard.refresh()

    with pytest.raises(VehicleNotFound):
        await dashboard.card(99, no_predictions)
    assert (await dashboard.card(8, no_predictions)).vehicle.tr_id == 8


@pytest.mark.anyio
async def test_card_of_a_planned_vehicle_outside_the_window_has_no_vehicle() -> None:
    dashboard = attached(Bus(), telemetry_record(8, at(0)))
    await dashboard.refresh()

    card = await dashboard.card(9, no_predictions)

    assert (card.seq, card.clock, card.vehicle) == (2, at(0), None)
    assert [s.stop_id for s in card.stops] == [30]


@pytest.mark.anyio
async def test_card_windows_follow_card_track_sec() -> None:
    clock = at(3600)
    span = timedelta(seconds=CONFIG.card_track_sec)
    second = timedelta(seconds=1)
    fleet = FleetState(WINDOW)
    for t in (clock - span - second, clock - span, clock):
        fleet.add_telemetry(telemetry_record(7, t))
    plan = {
        7: [
            plan_stop(7, 1, clock - span - second),
            plan_stop(7, 2, clock - span),
            plan_stop(7, 3, clock + span),
            plan_stop(7, 4, clock + span + second),
        ]
    }
    fleet.add_stop_event(stop_event(7, 2, clock - span, clock - span + 40 * second))
    dashboard = Dashboard(Bus(), CONFIG)
    dashboard.attach(LiveState(fleet, plan))
    await dashboard.refresh()
    calls = []

    async def load(tr_id, since):
        calls.append((tr_id, since))
        return [row_at(3600, target_stop_id=3), row_at(3540, target_stop_id=3)]

    card = await dashboard.card(7, load)

    assert (card.seq, card.clock) == (dashboard.snapshot().seq, clock)
    assert calls == [(7, clock - span)]
    assert [p.event_time for p in card.track] == [clock - span, clock]
    assert [s.stop_id for s in card.stops] == [2, 3]
    assert card.stops[0].delay_s == 40.0
    assert [p.sample_id for p in card.predictions] == ["7_3600", "7_3540"]
    assert card.vehicle.tr_id == 7

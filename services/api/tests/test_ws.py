import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta

import pytest
from factories import at, plan_stop, prediction_row, telemetry_record
from fastapi.testclient import TestClient
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

import app.ws as ws_module
from app.dashboard import Dashboard
from app.live import LiveState
from app.main import app
from app.schemas import WS_MESSAGE, PredictionMessage
from app.state import FleetState
from app.ws import Hub, relay
from common.bus import DASHBOARD_CHANNEL, BusMessage, publish
from common.testing import subscriber_count, wait_for_subscribers


class FakeSocket:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, text: str) -> None:
        if self.mode == "hang":
            await asyncio.sleep(3600)
        if self.mode == "fail":
            raise RuntimeError("connection lost")
        self.sent.append(text)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class Stop(Exception):
    pass


def recording_sleep(delays: list[float], stop_after: int) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == stop_after:
            raise Stop

    return sleep


@pytest.mark.anyio
async def test_broadcast_drops_slow_and_broken_clients() -> None:
    hub = Hub(send_timeout=0.05)
    ok, slow, broken = FakeSocket(), FakeSocket("hang"), FakeSocket("fail")
    for socket in (ok, slow, broken):
        hub.add(socket)

    await asyncio.wait_for(hub.broadcast("first"), 1)
    await hub.broadcast("second")

    assert ok.sent == ["first", "second"]
    assert len(hub) == 1
    assert slow.closed and broken.closed


@pytest.mark.anyio
async def test_relay_backs_off_while_the_bus_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unreachable(redis, channel):
        raise RedisConnectionError("down")
        yield

    monkeypatch.setattr(ws_module, "subscribe", unreachable)
    delays: list[float] = []

    with pytest.raises(Stop):
        await relay("redis://unused:6379/0", "ch", Hub(), sleep=recording_sleep(delays, 6))

    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]


@pytest.mark.anyio
async def test_relay_backs_off_on_an_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(redis, channel):
        raise ValueError("malformed redis_url")
        yield

    monkeypatch.setattr(ws_module, "subscribe", broken)
    delays: list[float] = []

    with pytest.raises(Stop):
        await relay("redis://unused:6379/0", "ch", Hub(), sleep=recording_sleep(delays, 3))

    assert delays == [0.5, 1.0, 2.0]


@pytest.mark.anyio
async def test_relay_resets_backoff_after_a_delivered_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = BusMessage(type="ping", data={})
    attempts = 0

    async def flaky(redis, channel):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            yield message
        raise RedisConnectionError("down")

    monkeypatch.setattr(ws_module, "subscribe", flaky)
    socket = FakeSocket()
    hub = Hub()
    hub.add(socket)
    delays: list[float] = []

    with pytest.raises(Stop):
        await relay("redis://unused:6379/0", "ch", hub, sleep=recording_sleep(delays, 4))

    assert delays == [0.5, 1.0, 0.5, 1.0]
    assert socket.sent == [message.model_dump_json()]


# database_url is not used directly: TestClient(app) runs the lifespan,
# which builds ServiceSettings, where database_url is required. Drop the
# marker and the test fails on a validation error instead of skipping.
@pytest.mark.usefixtures("database_url", "no_prediction_loop")
@pytest.mark.timeout(30)
def test_ws_delivers_bus_messages(redis_url: str) -> None:
    sent = BusMessage(type="test", data={"n": 1})

    with Redis.from_url(redis_url) as redis:
        before = subscriber_count(redis, DASHBOARD_CHANNEL)
        with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
            wait_for_subscribers(redis, DASHBOARD_CHANNEL, before + 1)
            publish(redis, DASHBOARD_CHANNEL, sent)

            assert BusMessage.model_validate_json(websocket.receive_text()) == sent


# Same reason as above: database_url is unused here too, but TestClient(app)
# still requires it through ServiceSettings at lifespan startup.
@pytest.mark.usefixtures("database_url", "no_prediction_loop")
@pytest.mark.timeout(30)
def test_ws_discards_client_on_disconnect() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws"):
            assert len(client.app.state.hub) == 1
        assert len(client.app.state.hub) == 0


# Same reason as above: database_url is required by the lifespan's settings.
@pytest.mark.usefixtures("database_url", "no_prediction_loop")
@pytest.mark.timeout(30)
def test_ws_delivers_dashboard_predictions_through_redis(redis_url: str) -> None:
    fleet = FleetState(timedelta(seconds=1800))
    fleet.add_telemetry(telemetry_record(7, at(0)))
    live = LiveState(fleet, {7: [plan_stop(7, 20, at(720), address="Lenina 1")]})
    row = prediction_row(sample_id="ws-1", tr_id=7, target_stop_id=20, risk_level="red")

    with Redis.from_url(redis_url) as redis:
        before = subscriber_count(redis, DASHBOARD_CHANNEL)
        with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
            wait_for_subscribers(redis, DASHBOARD_CHANNEL, before + 1)
            dashboard: Dashboard = client.app.state.dashboard
            dashboard.attach(live)
            # The dashboard belongs to the app's event loop, run by TestClient's portal.
            client.portal.call(dashboard.publish_predictions, [row])

            message = WS_MESSAGE.validate_json(websocket.receive_text())

    assert isinstance(message, PredictionMessage)
    assert (message.data.seq, message.data.tr_id) == (1, 7)
    assert message.data.prediction.sample_id == "ws-1"
    assert message.data.prediction.target_address == "Lenina 1"
    assert message.data.prediction.risk_level == "red"

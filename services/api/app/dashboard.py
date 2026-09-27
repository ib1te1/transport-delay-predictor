"""The dashboard's side of api: numbered WebSocket messages and the view they add up to.

One ``Dashboard`` per api process publishes every message the dashboard
channel carries: the per-second clock and vehicles diff, a message per
prediction row of a tick and one per alert that changed. Each message
gets the next ``seq`` and is published under one lock, so the channel
carries them in ``seq`` order whichever task sends them. The snapshot is the last published view and
its ``seq``: snapshot and stream come from the same place, and a client
that applies every message after the snapshot's ``seq`` ends up where
api is.

The ``Dashboard`` outlives restarts of the prediction loop: each run
attaches its ``LiveState``, and until the new run publishes, the
snapshot keeps showing the last known view instead of emptiness.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import ApiConfig
from app.live import LiveState
from app.models import AlertRow, PredictionRow
from app.schemas import (
    AlertData,
    AlertMessage,
    AlertView,
    ClockData,
    ClockMessage,
    PredictionData,
    PredictionMessage,
    StateSnapshot,
    VehicleCard,
    VehiclesData,
    VehiclesMessage,
    VehicleView,
)
from app.views import alert_view, card_stops, diff_views, summarize, track
from common.bus import DASHBOARD_CHANNEL, BusMessage

log = logging.getLogger(__name__)

# Real seconds between two clock-and-diff rounds.
REFRESH_SEC = 1.0
# A publish that takes longer is given up: it holds the lock every other
# message waits on.
PUBLISH_TIMEOUT_SEC = 1.0

type Sleep = Callable[[float], Awaitable[None]]
type LoadPredictions = Callable[[int, datetime], Awaitable[list[PredictionRow]]]
type Message = ClockMessage | VehiclesMessage | PredictionMessage | AlertMessage


class StateUnavailable(Exception):
    """The prediction loop has not built its state yet."""


class VehicleNotFound(Exception):
    """The vehicle is neither in the plan nor in the telemetry window."""


def envelope(message: Message) -> str:
    """The message as the bus carries it: a ``common.bus.BusMessage`` in JSON."""
    data = message.data.model_dump(mode="json")
    return BusMessage(type=message.type, data=data).model_dump_json()


class Dashboard:
    """Publishes the dashboard's messages and remembers the view they add up to.

    Not thread-safe; everything runs on the api event loop.
    """

    def __init__(
        self,
        redis: Redis,
        config: ApiConfig,
        *,
        channel: str = DASHBOARD_CHANNEL,
        publish_timeout: float = PUBLISH_TIMEOUT_SEC,
    ) -> None:
        self._redis = redis
        self._config = config
        self._channel = channel
        self._publish_timeout = publish_timeout
        self._lock = asyncio.Lock()
        self._seq = 0
        self._clock: datetime | None = None
        self._views: dict[int, VehicleView] = {}
        self._live: LiveState | None = None
        self._alerts: dict[int, AlertView] = {}
        self._live_mae_s: float | None = None
        self._checked_predictions = 0

    @property
    def seq(self) -> int:
        """The number of the last message published, ``0`` before the first."""
        return self._seq

    @property
    def live(self) -> LiveState | None:
        """The state of the current loop run, ``None`` until the first run attaches one."""
        return self._live

    def attach(self, live: LiveState) -> None:
        """Take ``live`` as the state to diff and serve cards from.

        The published view stays as it is; the next refresh publishes how
        ``live`` differs from it.
        """
        self._live = live

    async def publish_predictions(self, rows: Sequence[PredictionRow]) -> None:
        """Make each row its vehicle's current prediction and publish a ``prediction`` message.

        The map is not touched here: the vehicle's new prediction goes out
        with the next ``vehicles`` diff.
        """
        live = self._live
        if live is None:
            raise StateUnavailable("no loop state attached")
        async with self._lock:
            for row in rows:
                live.current.put(row)
                data = PredictionData(
                    seq=self._next_seq(), tr_id=row.tr_id, prediction=live.prediction_view(row)
                )
                await self._send(PredictionMessage(type="prediction", data=data))

    def set_alerts(self, rows: Sequence[AlertRow]) -> None:
        """Take ``rows`` as the open alerts, without a message: the loop reloads them on start."""
        self._alerts = {row.id: alert_view(row) for row in rows if row.status == "open"}

    async def publish_alerts(self, rows: Sequence[AlertRow]) -> None:
        """Keep the open alerts for the snapshot and publish an ``alert`` message per row."""
        async with self._lock:
            for row in rows:
                view = alert_view(row)
                if row.status == "open":
                    self._alerts[row.id] = view
                else:
                    self._alerts.pop(row.id, None)
                data = AlertData(seq=self._next_seq(), alert=view)
                await self._send(AlertMessage(type="alert", data=data))

    def set_accuracy(self, live_mae_s: float | None, checked_predictions: int) -> None:
        """The live MAE and how many predictions it covers, as the snapshot shows them."""
        self._live_mae_s = live_mae_s
        self._checked_predictions = checked_predictions

    async def refresh(self) -> None:
        """Publish the clock, then the vehicles whose view changed since the last diff.

        Nothing is published before the loop has attached a state with a
        known clock.
        """
        live = self._live
        if live is None:
            return
        async with self._lock:
            views = live.views(self._config)
            clock = live.fleet.clock
            if clock is None:
                return
            self._clock = clock
            await self._send(
                ClockMessage(type="clock", data=ClockData(seq=self._next_seq(), clock=clock))
            )
            changed, removed = diff_views(self._views, views)
            if changed or removed:
                self._views = views
                data = VehiclesData(
                    seq=self._next_seq(), clock=clock, vehicles=changed, removed=removed
                )
                await self._send(VehiclesMessage(type="vehicles", data=data))

    async def run(self, *, period: float = REFRESH_SEC, sleep: Sleep = asyncio.sleep) -> None:
        """Refresh once a ``period`` of real time until cancelled.

        A refresh that raises is logged and retried after ``period``; it
        does not stop the loop. Cancellation is not caught here and still
        stops it.
        """
        while True:
            try:
                await self.refresh()
            except Exception:
                log.exception("dashboard refresh failed")
            await sleep(period)

    def snapshot(self) -> StateSnapshot:
        """The last published view and the ``seq`` of the last published message."""
        views = [self._views[tr_id] for tr_id in sorted(self._views)]
        alerts = sorted(self._alerts.values(), key=lambda a: (a.opened_at, a.id), reverse=True)
        return StateSnapshot(
            seq=self._seq,
            clock=self._clock,
            vehicles=views,
            summary=summarize(views),
            alerts=alerts,
            live_mae_s=self._live_mae_s,
            checked_predictions=self._checked_predictions,
        )

    async def card(self, tr_id: int, load_predictions: LoadPredictions) -> VehicleCard:
        """One vehicle in detail as of the last published view.

        ``load_predictions(tr_id, since)`` reads the vehicle's predictions
        from the table, newest first; whatever it raises propagates.
        Raises ``StateUnavailable`` before the loop has attached a state
        and ``VehicleNotFound`` for a vehicle neither planned nor seen.
        """
        live = self._live
        if live is None:
            raise StateUnavailable("the prediction loop has not built its state yet")
        seq, clock = self._seq, self._clock
        vehicle = self._views.get(tr_id)
        if vehicle is None and tr_id not in live.plan:
            raise VehicleNotFound(tr_id)
        if clock is None:
            return VehicleCard(
                seq=seq, clock=None, vehicle=None, predictions=[], track=[], stops=[]
            )
        span = timedelta(seconds=self._config.card_track_sec)
        points = track(live.fleet.telemetry(tr_id), clock - span, clock)
        stops = card_stops(
            live.plan.get(tr_id, []), live.fleet.stop_events(tr_id), clock - span, clock + span
        )
        rows = await load_predictions(tr_id, clock - span)
        return VehicleCard(
            seq=seq,
            clock=clock,
            vehicle=vehicle,
            predictions=[live.prediction_view(row) for row in rows],
            track=points,
            stops=stops,
        )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _send(self, message: Message) -> None:
        """Publish one message; a failure is logged and the number stays spent.

        A client sees the gap in ``seq`` and takes the snapshot again,
        which already includes what the lost message carried.
        """
        try:
            async with asyncio.timeout(self._publish_timeout):
                await self._redis.publish(self._channel, envelope(message))
        except (RedisError, OSError) as exc:  # TimeoutError is an OSError
            log.warning("dashboard message %d not published: %r", message.data.seq, exc)

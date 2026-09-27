"""Figures of the prediction loop for ``GET /metrics``.

Kept in memory over the last five minutes of real time. One ``LoopMetrics``
per api process: it outlives restarts of the loop, like the dashboard.
"""

import math
import time
from collections import deque
from collections.abc import Callable, Sequence

from app.models import PredictionRow
from app.schemas import Spread

WINDOW_SEC = 300.0


def spread(values: Sequence[float]) -> Spread | None:
    """p50, p95 and max by nearest rank, ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)

    def rank(q: float) -> float:
        return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]

    return Spread(
        p50=round(rank(0.5), 1),
        p95=round(rank(0.95), 1),
        max=round(ordered[-1], 1),
        count=len(ordered),
    )


def entry_time(entry_id: str) -> float:
    """When Redis appended a stream entry, in seconds since the epoch."""
    return int(entry_id.split("-", 1)[0]) / 1000


class LoopMetrics:
    """Calls to predictor, ticks and written rows of the last ``window_sec`` real seconds.

    Not thread-safe; everything runs on the api event loop.
    """

    def __init__(
        self,
        window_sec: float = WINDOW_SEC,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._window = window_sec
        self._monotonic = monotonic
        self._wall = wall
        self._started = monotonic()
        # (when, ms, ok) per call to predictor
        self._predict: deque[tuple[float, float, bool]] = deque()
        # (when, ms) per tick that had something to score
        self._ticks: deque[tuple[float, float]] = deque()
        # (when, rows, degraded rows) per batch written
        self._rows: deque[tuple[float, int, int]] = deque()
        self.stream_lag_s: float | None = None

    def predict_call(self, ms: float, *, ok: bool) -> None:
        """One call to predictor, answered or not."""
        self._predict.append((self._monotonic(), ms, ok))

    def tick(self, ms: float) -> None:
        """One tick from planning to publishing."""
        self._ticks.append((self._monotonic(), ms))

    def rows_written(self, rows: Sequence[PredictionRow]) -> None:
        """Prediction rows that went into the table."""
        degraded = sum(row.degraded for row in rows)
        self._rows.append((self._monotonic(), len(rows), degraded))

    def telemetry_read(self, entry_id: str) -> None:
        """A telemetry entry was read; its lag is how long it waited in the stream."""
        self.stream_lag_s = round(max(self._wall() - entry_time(entry_id), 0.0), 3)

    def predict_latency_ms(self) -> Spread | None:
        return spread([ms for _, ms, _ in self._recent(self._predict)])

    def predict_failures(self) -> int:
        return sum(not ok for _, _, ok in self._recent(self._predict))

    def scoring_tick_ms(self) -> Spread | None:
        return spread([ms for _, ms in self._recent(self._ticks)])

    def predictions_per_min(self) -> float:
        """Rows written per real minute, over the window or since start if that is shorter."""
        span = min(self._window, self._monotonic() - self._started)
        rows = sum(n for _, n, _ in self._recent(self._rows))
        return round(rows * 60 / span, 1) if span > 0 else 0.0

    def degraded_share(self) -> float | None:
        """Share of rows written with ``degraded``, ``None`` before the first row."""
        batches = self._recent(self._rows)
        rows = sum(n for _, n, _ in batches)
        return round(sum(d for _, _, d in batches) / rows, 3) if rows else None

    def _recent[T: tuple](self, items: deque[T]) -> list[T]:
        cutoff = self._monotonic() - self._window
        while items and items[0][0] < cutoff:
            items.popleft()
        return list(items)

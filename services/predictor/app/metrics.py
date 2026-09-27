"""How long ``POST /predict`` takes, served at ``GET /metrics``.

A call is timed from the moment it arrives until the answer is ready: reading and parsing the
body, the model, writing the answer. That is what the api's ``predict_timeout_ms`` has to
cover, apart from the network between the containers.
"""

from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from pydantic import BaseModel

# The figures cover this many last calls. A whole day replayed at 60x makes about 1500.
WINDOW = 5000


class Spread(BaseModel):
    """Percentiles of one quantity over the window."""

    p50: float
    p95: float
    p99: float
    max: float


class PredictMetrics(BaseModel):
    """``POST /predict`` since the service started."""

    calls: int
    # answered with an error status, e.g. 422 on a batch that breaks the contract
    failed: int
    # successful calls the figures below are over
    window: int
    latency_ms: Spread | None = None
    batch_size: Spread | None = None
    body_kb: Spread | None = None


class Metrics(BaseModel):
    """What ``GET /metrics`` answers."""

    model_version: str
    started: datetime
    predict: PredictMetrics


def spread(values: Sequence[float]) -> Spread:
    """p50, p95, p99 and max, rounded to 0.1."""
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return Spread(
        p50=round(float(p50), 1),
        p95=round(float(p95), 1),
        p99=round(float(p99), 1),
        max=round(float(max(values)), 1),
    )


class PredictTimings:
    """Counts every call; keeps duration, batch size and body size of the last successful ones."""

    def __init__(self, window: int = WINDOW) -> None:
        self.calls = 0
        self.failed = 0
        self._recent: deque[tuple[float, int, int]] = deque(maxlen=window)

    def add(self, ms: float, *, ok: bool, batch: int, body_bytes: int) -> None:
        """Record one call."""
        self.calls += 1
        if ok:
            self._recent.append((ms, batch, body_bytes))
        else:
            self.failed += 1

    def summary(self) -> PredictMetrics:
        """The counters and the percentiles over the window; no percentiles before a call."""
        out = PredictMetrics(calls=self.calls, failed=self.failed, window=len(self._recent))
        if self._recent:
            ms, batch, body = zip(*self._recent, strict=True)
            out.latency_ms = spread(ms)
            out.batch_size = spread(batch)
            out.body_kb = spread([b / 1024 for b in body])
        return out

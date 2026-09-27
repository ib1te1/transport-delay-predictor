from factories import prediction_row

from app.metrics import LoopMetrics, entry_time, spread
from app.schemas import Spread


class Clock:
    """A settable clock for both of LoopMetrics' time sources."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_spread_is_by_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]

    assert spread(values) == Spread(p50=50.0, p95=95.0, max=100.0, count=100)
    assert spread([7.0]) == Spread(p50=7.0, p95=7.0, max=7.0, count=1)
    assert spread([]) is None


def test_entry_time_is_the_stream_id_in_seconds() -> None:
    assert entry_time("1767686400123-4") == 1767686400.123


def test_figures_cover_only_the_window() -> None:
    clock = Clock()
    metrics = LoopMetrics(300, monotonic=clock)
    metrics.predict_call(900.0, ok=False)
    metrics.tick(1000.0)
    clock.now += 200
    metrics.predict_call(40.0, ok=True)
    metrics.predict_call(60.0, ok=True)
    metrics.tick(80.0)
    clock.now += 150

    assert metrics.predict_latency_ms() == Spread(p50=40.0, p95=60.0, max=60.0, count=2)
    assert metrics.predict_failures() == 0
    assert metrics.scoring_tick_ms() == Spread(p50=80.0, p95=80.0, max=80.0, count=1)


def test_rows_per_minute_and_degraded_share() -> None:
    clock = Clock()
    metrics = LoopMetrics(300, monotonic=clock)
    assert (metrics.predictions_per_min(), metrics.degraded_share()) == (0.0, None)

    metrics.rows_written([prediction_row(), prediction_row(degraded=True)])
    clock.now += 60
    metrics.rows_written([prediction_row(), prediction_row()])

    # four rows in the first minute since start
    assert metrics.predictions_per_min() == 4.0
    assert metrics.degraded_share() == 0.25
    clock.now += 600
    assert (metrics.predictions_per_min(), metrics.degraded_share()) == (0.0, None)


def test_stream_lag_is_how_long_the_last_entry_waited() -> None:
    metrics = LoopMetrics(monotonic=Clock(), wall=Clock(1767686402.5))
    assert metrics.stream_lag_s is None

    metrics.telemetry_read("1767686400000-0")

    assert metrics.stream_lag_s == 2.5

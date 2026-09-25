from datetime import timedelta

import pytest
from factories import at, plan_stop, stop_event, telemetry_record

from app.config import ApiConfig
from app.models import PredictionRow
from app.predictor_client import PredictorError
from app.scoring import degraded_reason, run_tick
from app.state import FleetState
from contracts import PredictRequest, PredictResponse, ReasonCode

CONFIG = ApiConfig()
T = at(3600)
SEC = timedelta(seconds=1)
TARGET = plan_stop(7, 1, T + timedelta(minutes=12))
PLAN = {7: [TARGET]}


def state_at_t(*, last_seen=T, since=T - 1800 * SEC, with_stop_event: bool = True) -> FleetState:
    """Vehicle 7 seen first at ``since`` and last at ``last_seen``; the clock at T."""
    state = FleetState(timedelta(seconds=CONFIG.request.telemetry_window_sec))
    state.add_telemetry(telemetry_record(7, since))
    state.add_telemetry(telemetry_record(7, last_seen))
    state.add_telemetry(telemetry_record(None, T, unit_id=99))
    if with_stop_event:
        plan = T - timedelta(minutes=5)
        state.add_stop_event(stop_event(7, 0, plan, plan + 90 * SEC))
    return state


class Sink:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.saved: list[PredictionRow] = []
        self.published: list[PredictionRow] = []

    async def save(self, rows: list[PredictionRow]) -> None:
        self.events.append("save")
        self.saved.extend(rows)

    async def publish(self, rows: list[PredictionRow]) -> None:
        self.events.append("publish")
        self.published.extend(rows)


def model(prediction_s: float, reasons=(ReasonCode.accumulated_delay,)):
    async def predict(requests: list[PredictRequest]) -> list[PredictResponse]:
        return [
            PredictResponse(
                sample_id=r.sample_id,
                prediction_s=prediction_s,
                p_late=0.9,
                reasons=list(reasons),
                model_version="m1",
            )
            for r in requests
        ]

    return predict


async def down(requests: list[PredictRequest]) -> list[PredictResponse]:
    raise PredictorError("down")


async def tick(state: FleetState, predict, sink: Sink | None = None) -> list[PredictionRow]:
    sink = sink or Sink()
    return await run_tick(
        T, state, PLAN, CONFIG, predict=predict, save=sink.save, publish=sink.publish
    )


@pytest.mark.anyio
async def test_tick_saves_the_model_answer_then_publishes_it() -> None:
    sink = Sink()

    [row] = await tick(state_at_t(), model(400.0), sink)

    assert row.sample_id == f"7_{int(T.timestamp())}"
    assert (row.tr_id, row.t, row.target_stop_id, row.target_time_begin) == (
        7,
        T,
        1,
        TARGET.time_plan,
    )
    assert (row.cur_dev_s, row.prediction_s, row.p_late) == (90.0, 400.0, 0.9)
    assert row.reasons == ["accumulated_delay"]
    assert (row.risk_level, row.degraded, row.degraded_reason) == ("red", False, None)
    assert row.model_version == "m1"
    assert (row.actual_delay_s, row.abs_error_s) == (None, None)
    assert sink.events == ["save", "publish"]
    assert sink.saved == sink.published == [row]


@pytest.mark.anyio
async def test_predictor_failure_falls_back_to_the_current_deviation() -> None:
    [row] = await tick(state_at_t(), down)

    assert (row.prediction_s, row.p_late, row.reasons) == (90.0, None, ["unknown"])
    assert row.model_version == "fallback"
    assert (row.degraded, row.degraded_reason) == (True, "predictor_unavailable")
    assert row.risk_level == "green"


@pytest.mark.anyio
async def test_fallback_without_a_deviation_predicts_zero() -> None:
    [row] = await tick(state_at_t(with_stop_event=False), down)

    assert (row.cur_dev_s, row.prediction_s) == (None, 0.0)


@pytest.mark.anyio
async def test_next_tick_tries_the_predictor_again() -> None:
    calls: list[int] = []

    async def flaky(requests: list[PredictRequest]) -> list[PredictResponse]:
        calls.append(len(requests))
        if len(calls) == 1:
            raise PredictorError("down")
        return await model(10.0)(requests)

    state = state_at_t()
    [first] = await tick(state, flaky)
    [second] = await tick(state, flaky)

    assert (first.model_version, second.model_version) == ("fallback", "m1")
    assert second.degraded is False
    assert calls == [1, 1]


@pytest.mark.anyio
async def test_stale_vehicle_is_flagged_and_gets_the_stale_reason() -> None:
    [row] = await tick(state_at_t(last_seen=T - 121 * SEC), model(30.0))

    assert row.reasons == ["accumulated_delay", "stale_telemetry"]
    assert (row.degraded, row.degraded_reason) == (True, "stale_telemetry")


@pytest.mark.anyio
async def test_stale_vehicle_with_predictor_down_reports_the_predictor() -> None:
    [row] = await tick(state_at_t(last_seen=T - 121 * SEC), down)

    assert row.reasons == ["unknown", "stale_telemetry"]
    assert row.degraded_reason == "predictor_unavailable"


@pytest.mark.anyio
async def test_stale_reason_is_not_duplicated() -> None:
    answer = model(30.0, reasons=(ReasonCode.stale_telemetry,))

    [row] = await tick(state_at_t(last_seen=T - 121 * SEC), answer)

    assert row.reasons == ["stale_telemetry"]


@pytest.mark.anyio
async def test_incomplete_window_is_flagged_as_warm_up_without_touching_reasons() -> None:
    [row] = await tick(state_at_t(since=T - 60 * SEC), model(30.0))

    assert row.reasons == ["accumulated_delay"]
    assert (row.degraded, row.degraded_reason) == (True, "warming_up")


@pytest.mark.parametrize(
    ("fallback", "stale", "warming_up", "reason"),
    [
        (True, True, True, "predictor_unavailable"),
        (False, True, True, "stale_telemetry"),
        (False, False, True, "warming_up"),
        (False, False, False, None),
    ],
)
def test_degraded_reason_precedence(fallback: bool, stale: bool, warming_up: bool, reason) -> None:
    assert degraded_reason(fallback=fallback, stale=stale, warming_up=warming_up) == reason


@pytest.mark.anyio
async def test_tick_without_candidates_calls_nothing() -> None:
    async def unexpected(requests: list[PredictRequest]) -> list[PredictResponse]:
        raise AssertionError("predictor must not be called")

    sink = Sink()
    rows = await run_tick(
        T, state_at_t(), {}, CONFIG, predict=unexpected, save=sink.save, publish=sink.publish
    )

    assert rows == []
    assert sink.events == []

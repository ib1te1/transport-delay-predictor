from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from contracts import (
    PredictRequest,
    PredictResponse,
    ReasonCode,
    StopEvent,
    TelemetryRecord,
    make_sample_id,
)

T = datetime(2026, 1, 6, 3, 35, tzinfo=UTC)


def telemetry(**overrides):
    fields = dict(
        tr_id=131672,
        unit_id=664030,
        event_time=T,
        lat=55.8,
        lon=37.4,
        location_valid=True,
        speed_kmh=21.0,
        heading_deg=90.0,
        source="replay",
    )
    fields.update(overrides)
    return TelemetryRecord(**fields)


def test_sample_id_matches_the_organizers_format():
    assert make_sample_id(131672, T) == "131672_1767670500"


def test_naive_datetime_is_rejected():
    with pytest.raises(ValidationError):
        telemetry(event_time=datetime(2026, 1, 6, 3, 35))


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        telemetry(route="42")


def test_unknown_source_is_rejected():
    with pytest.raises(ValidationError):
        telemetry(source="csv")


def test_telemetry_without_schedule_match_is_allowed():
    assert telemetry(tr_id=None).tr_id is None


def test_stop_event_round_trips_through_json():
    event = StopEvent(stop_id=53699433970, tr_id=122658, time_plan=T, time_fact=T, delay_s=0.0)
    assert StopEvent.model_validate_json(event.model_dump_json()) == event


def test_predict_request_round_trips_through_json():
    request = PredictRequest(
        sample_id=make_sample_id(131672, T),
        tr_id=131672,
        T=T,
        target_stop_id=53700172828,
        target_time_begin=datetime(2026, 1, 6, 3, 50, tzinfo=UTC),
        cur_dev_s=274.0,
        telemetry=[],
        schedule=[],
    )
    assert PredictRequest.model_validate_json(request.model_dump_json()) == request


def test_reason_codes_are_a_closed_set():
    with pytest.raises(ValidationError):
        PredictResponse(
            sample_id="x", prediction_s=1.0, p_late=None, reasons=["weather"], model_version="m"
        )
    response = PredictResponse(
        sample_id="x",
        prediction_s=1.0,
        p_late=0.2,
        reasons=[ReasonCode.slow_approach],
        model_version="m",
    )
    assert response.reasons == [ReasonCode.slow_approach]

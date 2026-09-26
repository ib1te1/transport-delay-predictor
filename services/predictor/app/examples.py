"""Made-up requests: the example on the Swagger page and the tests.

A bus goes east along a straight street with a stop every ~310 m and every minute, and
reaches every stop the same number of seconds late. The example on the Swagger page is such
a bus, so "Try it out" gets an answer from the model and not the baseline.
"""

from datetime import UTC, datetime, timedelta

from contracts import PredictRequest, ScheduledStop, TelemetryPoint, make_sample_id

START = datetime(2026, 1, 6, 4, 0, tzinfo=UTC)


def stops(tr_id: int, n: int = 120) -> list[ScheduledStop]:
    """The street of vehicle ``tr_id``, a stop a minute, a 10 minute layover after stop 59."""
    return [
        ScheduledStop(
            stop_id=tr_id * 1000 + k,
            time_plan=START + timedelta(minutes=k + (10 if k >= 60 else 0)),
            lat=55.75 + tr_id * 0.01,
            lon=37.60 + 0.005 * k,
            time_fact=None,
        )
        for k in range(n)
    ]


def fixes(
    schedule: list[ScheduledStop], late_s: float, until: datetime, step_s: float = 10
) -> list[TelemetryPoint]:
    """A bus that reaches every stop ``late_s`` after the plan, a fix every ``step_s``."""
    points = []
    for a, b in zip(schedule, schedule[1:], strict=False):
        leave, arrive = a.time_plan + timedelta(seconds=late_s + 20), b.time_plan
        arrive += timedelta(seconds=late_s)
        t = a.time_plan + timedelta(seconds=late_s)
        while t < arrive and t <= until:
            share = 0.0 if t <= leave else (t - leave) / (arrive - leave)
            points.append(
                TelemetryPoint(
                    event_time=t,
                    lat=a.lat,
                    lon=a.lon + (b.lon - a.lon) * share,
                    location_valid=True,
                    speed_kmh=0.0 if t <= leave else 30.0,
                    heading_deg=90.0,
                )
            )
            t += timedelta(seconds=step_s)
    return points


def bus_request(
    tr_id: int, late_s: float, cur_dev_s: float | None = None, step_s: float = 10
) -> PredictRequest:
    """The request the api sends 40 minutes into the trip: the plan up to the first stop
    10-15 minutes ahead and the telemetry up to ``T``."""
    T = START + timedelta(minutes=40)
    schedule = stops(tr_id)
    target = next(s for s in schedule if s.time_plan > T + timedelta(minutes=10))
    return PredictRequest(
        sample_id=make_sample_id(tr_id, T),
        tr_id=tr_id,
        T=T,
        target_stop_id=target.stop_id,
        target_time_begin=target.time_plan,
        cur_dev_s=cur_dev_s,
        telemetry=fixes(schedule, late_s, T, step_s),
        schedule=[s for s in schedule if s.time_plan <= target.time_plan],
    )

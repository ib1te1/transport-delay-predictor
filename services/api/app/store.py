"""Postgres side of api: the plan in, prediction rows and alerts out and back.

Synchronous like the rest of ``common.db``; callers on the event loop go
through ``asyncio.to_thread``. Nothing here commits.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import NamedTuple

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from app.models import AlertRow, AlertStatus, PlanStop, PredictionRow
from app.planner import PlanIndex
from app.schemas import NetworkStop
from common.db import fetch_models, insert_models
from contracts import StopEvent


class Accuracy(NamedTuple):
    """Live MAE over the predictions checked against arrivals so far."""

    live_mae_s: float | None
    checked_predictions: int


class AlertCounts(NamedTuple):
    """Alerts per status and the mean lead time of the confirmed ones."""

    open: int
    confirmed: int
    cancelled: int
    mean_lead_time_s: float | None


def load_plan(conn: psycopg.Connection) -> PlanIndex:
    """Planned stops by vehicle, each list ordered by ``time_plan``."""
    plan: PlanIndex = {}
    stops = fetch_models(
        conn,
        PlanStop,
        "SELECT stop_id, tr_id, time_plan, lat, lon, address FROM stops_plan"
        " ORDER BY tr_id, time_plan, stop_id",
    )
    for stop in stops:
        plan.setdefault(stop.tr_id, []).append(stop)
    return plan


def load_network_stops(conn: psycopg.Connection) -> list[NetworkStop]:
    """All planned stops, numbered within each vehicle run."""
    return fetch_models(
        conn,
        NetworkStop,
        "SELECT tr_id AS route_id,"
        " ROW_NUMBER() OVER (PARTITION BY tr_id ORDER BY time_plan, stop_id) AS stop_order,"
        " stop_id, address, lat, lon, time_plan FROM stops_plan"
        " ORDER BY route_id, stop_order",
    )


def save_predictions(conn: psycopg.Connection, rows: list[PredictionRow]) -> list[PredictionRow]:
    """Insert prediction rows that are not already stored; does not commit.

    A ``sample_id`` already stored keeps its first row: a tick repeated
    after a restart must not rewrite a prediction already shown. Returns
    the rows that were inserted, in input order — for a ``sample_id``
    repeated within the batch, only its first row.
    """
    if not rows:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sample_id FROM predictions WHERE sample_id = ANY(%s)",
            ([row.sample_id for row in rows],),
        )
        seen = {sample_id for (sample_id,) in cur.fetchall()}
    insert_models(conn, "predictions", rows, on_conflict="ON CONFLICT (sample_id) DO NOTHING")
    saved = []
    for row in rows:
        if row.sample_id not in seen:
            saved.append(row)
            seen.add(row.sample_id)
    return saved


def load_latest_predictions(conn: psycopg.Connection, since: datetime) -> list[PredictionRow]:
    """The newest prediction of each vehicle made at ``since`` or later, by ``tr_id``.

    Restores the current predictions after a restart; whether each still
    applies is decided by the caller.
    """
    return fetch_models(
        conn,
        PredictionRow,
        "SELECT DISTINCT ON (tr_id) * FROM predictions WHERE t >= %s ORDER BY tr_id, t DESC",
        (since,),
    )


def load_vehicle_predictions(
    conn: psycopg.Connection, tr_id: int, since: datetime
) -> list[PredictionRow]:
    """The vehicle's predictions made at ``since`` or later, newest first."""
    return fetch_models(
        conn,
        PredictionRow,
        "SELECT * FROM predictions WHERE tr_id = %s AND t >= %s ORDER BY t DESC",
        (tr_id, since),
    )


def check_predictions(conn: psycopg.Connection, events: Sequence[StopEvent]) -> int:
    """Put the fact on every prediction whose target stop is in ``events``; returns rows changed.

    A row that already holds the same fact is left alone, so checking the
    same arrivals again after a restart changes nothing.
    """
    if not events:
        return 0
    cur = conn.execute(
        "UPDATE predictions p"
        " SET actual_delay_s = e.delay_s, abs_error_s = abs(p.prediction_s - e.delay_s)"
        " FROM unnest(%s::bigint[], %s::double precision[]) AS e(stop_id, delay_s)"
        " WHERE p.target_stop_id = e.stop_id"
        " AND p.actual_delay_s IS DISTINCT FROM e.delay_s",
        ([e.stop_id for e in events], [e.delay_s for e in events]),
    )
    return cur.rowcount


def load_accuracy(conn: psycopg.Connection) -> Accuracy:
    """Mean ``abs_error_s`` over the checked predictions and how many there are."""
    mae, checked = conn.execute(
        "SELECT avg(abs_error_s), count(abs_error_s) FROM predictions"
    ).fetchone()
    return Accuracy(mae, checked)


def update_alerts(
    conn: psycopg.Connection,
    rows: Sequence[PredictionRow],
    segment_from: Mapping[str, int | None],
) -> list[AlertRow]:
    """Open, update or cancel the alert of each row's (vehicle, target) pair; returns the changes.

    A red row updates the open alert of its pair or opens one, starting
    the segment at ``segment_from[row.sample_id]``. A row below red
    cancels the open alert of its pair, if any.
    """
    changed = []
    with conn.cursor(row_factory=class_row(AlertRow)) as cur:
        for row in rows:
            pair = (row.tr_id, row.target_stop_id)
            if row.risk_level == "red":
                cur.execute(
                    "UPDATE alerts SET predicted_delay_s = %s, reasons = %s"
                    " WHERE status = 'open' AND tr_id = %s AND target_stop_id = %s RETURNING *",
                    (row.prediction_s, Jsonb(row.reasons), *pair),
                )
                alert = cur.fetchone()
                if alert is None:
                    cur.execute(
                        "INSERT INTO alerts (tr_id, target_stop_id, segment_from_stop_id, status,"
                        " opened_at, predicted_delay_s, reasons)"
                        " VALUES (%s, %s, %s, 'open', %s, %s, %s) RETURNING *",
                        (
                            *pair,
                            segment_from.get(row.sample_id),
                            row.t,
                            row.prediction_s,
                            Jsonb(row.reasons),
                        ),
                    )
                    alert = cur.fetchone()
            else:
                cur.execute(
                    "UPDATE alerts SET status = 'cancelled', closed_at = %s"
                    " WHERE status = 'open' AND tr_id = %s AND target_stop_id = %s RETURNING *",
                    (row.t, *pair),
                )
                alert = cur.fetchone()
            if alert is not None:
                changed.append(alert)
    return changed


def confirm_alerts(conn: psycopg.Connection, events: Sequence[StopEvent]) -> list[AlertRow]:
    """Confirm the open alerts whose target stop the vehicle reached; returns them.

    ``lead_time_s`` is how long before the arrival the alert was opened.
    """
    if not events:
        return []
    return fetch_models(
        conn,
        AlertRow,
        "UPDATE alerts a SET status = 'confirmed', closed_at = e.time_fact,"
        " actual_delay_s = e.delay_s,"
        " lead_time_s = extract(epoch FROM e.time_fact - a.opened_at)::double precision"
        " FROM unnest(%s::bigint[], %s::bigint[], %s::timestamptz[], %s::double precision[])"
        " AS e(tr_id, stop_id, time_fact, delay_s)"
        " WHERE a.status = 'open' AND a.tr_id = e.tr_id AND a.target_stop_id = e.stop_id"
        " RETURNING a.*",
        (
            [e.tr_id for e in events],
            [e.stop_id for e in events],
            [e.time_fact for e in events],
            [e.delay_s for e in events],
        ),
    )


def load_alerts(
    conn: psycopg.Connection, status: AlertStatus | None = None, limit: int | None = None
) -> list[AlertRow]:
    """Alerts with ``status`` (all if ``None``), newest first, at most ``limit``."""
    if status is None:
        return fetch_models(
            conn,
            AlertRow,
            "SELECT * FROM alerts ORDER BY opened_at DESC, id DESC LIMIT %s",
            (limit,),
        )
    return fetch_models(
        conn,
        AlertRow,
        "SELECT * FROM alerts WHERE status = %s ORDER BY opened_at DESC, id DESC LIMIT %s",
        (status, limit),
    )


def count_alerts(conn: psycopg.Connection) -> AlertCounts:
    """Alerts per status and the mean ``lead_time_s`` of the confirmed ones."""
    row = conn.execute(
        "SELECT count(*) FILTER (WHERE status = 'open'),"
        " count(*) FILTER (WHERE status = 'confirmed'),"
        " count(*) FILTER (WHERE status = 'cancelled'),"
        " avg(lead_time_s) FILTER (WHERE status = 'confirmed')"
        " FROM alerts"
    ).fetchone()
    return AlertCounts(*row)

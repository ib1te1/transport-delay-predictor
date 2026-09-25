"""Postgres side of api: the plan in, prediction rows out and back.

Synchronous like the rest of ``common.db``; callers on the event loop go
through ``asyncio.to_thread``.
"""

from datetime import datetime

import psycopg

from app.models import PlanStop, PredictionRow
from app.planner import PlanIndex
from common.db import fetch_models, insert_models


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


def save_predictions(conn: psycopg.Connection, rows: list[PredictionRow]) -> None:
    """Insert prediction rows; does not commit.

    A ``sample_id`` already stored keeps its first row: a tick repeated
    after a restart must not rewrite a prediction already shown.
    """
    insert_models(conn, "predictions", rows, on_conflict="ON CONFLICT (sample_id) DO NOTHING")


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

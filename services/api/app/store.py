"""Postgres side of the prediction loop: the plan in, prediction rows out.

Synchronous like the rest of ``common.db``; the loop calls these through
``asyncio.to_thread``.
"""

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

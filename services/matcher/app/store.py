"""Consume published telemetry, checkpoint detection, and deliver stop events."""

import json
import logging
from datetime import datetime, time, timedelta

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.dataset import Tick, Visit
from app.matching import Settings, StopDetector
from common.bus import STOP_EVENTS_STREAM, append
from contracts import StopEvent, TelemetryRecord

log = logging.getLogger(__name__)


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, default=lambda item: item.isoformat()))


def _vehicle_id(conn, row: dict) -> int | None:
    if row["tr_id"] is not None:
        return row["tr_id"]
    found = conn.execute(
        "SELECT tr_id FROM vehicles WHERE unit_id = %s", (row["unit_id"],)
    ).fetchone()
    return found[0] if found else None


class Store:
    def __init__(self, pool, settings: Settings):
        self.pool = pool
        self.settings = settings
        self._visits_cache: dict[tuple[int, object], list[Visit]] = {}

    def process_next(self) -> bool:
        """Process one published ingest row and advance the cursor atomically."""
        with self.pool.connection() as conn, conn.transaction():
            cursor = conn.execute(
                "SELECT last_telemetry_id FROM matcher_cursor WHERE id = true FOR UPDATE"
            ).fetchone()[0]
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM telemetry WHERE id > %s ORDER BY id LIMIT 1", (cursor,))
                row = cur.fetchone()
            if row is None or row["published_at"] is None:
                return False
            vehicle_id = _vehicle_id(conn, row)
            if vehicle_id is None:
                log.warning(
                    "skipping telemetry %d: unknown vehicle for unit %s", row["id"], row["unit_id"]
                )
            else:
                record = TelemetryRecord.model_validate(
                    {key: row[key] for key in TelemetryRecord.model_fields}
                )
                self._process(conn, vehicle_id, record)
            conn.execute(
                "UPDATE matcher_cursor SET last_telemetry_id = %s WHERE id = true", (row["id"],)
            )
            return True

    def _process(self, conn, tr_id: int, record: TelemetryRecord) -> None:
        vehicle = str(tr_id)
        at = record.event_time
        cache_key = (tr_id, at.date())
        visits = self._visits_cache.get(cache_key)
        if visits is None:
            day_start = datetime.combine(at.date(), time.min, tzinfo=at.tzinfo)
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT stop_id, time_plan, lon, lat FROM stops_plan "
                    "WHERE tr_id = %s AND time_plan BETWEEN %s AND %s "
                    "ORDER BY time_plan, stop_id",
                    (tr_id, day_start - timedelta(days=1), day_start + timedelta(days=2)),
                )
                visits = [
                    Visit(vehicle, str(row["stop_id"]), row["time_plan"], row["lon"], row["lat"])
                    for row in cur.fetchall()
                ]
            if visits:
                self._visits_cache[cache_key] = visits
        if not visits:
            log.warning("no planned stops near telemetry for vehicle %s at %s", tr_id, at)
            return
        detector = StopDetector(visits, self.settings)
        saved = conn.execute(
            "SELECT state FROM matcher_state WHERE tr_id = %s", (tr_id,)
        ).fetchone()
        if saved:
            detector.restore(vehicle, saved[0])
        tick = Tick(
            vehicle,
            at,
            record.lon if record.location_valid else None,
            record.lat if record.location_valid else None,
            record.speed_kmh,
        )
        detector.feed(tick)
        for event in detector.drain_events():
            key = (tr_id, int(event.visit_id), event.planned_at)
            existing = conn.execute(
                "SELECT 1 FROM stop_events WHERE tr_id = %s AND stop_id = %s AND time_plan = %s",
                key,
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE stop_events SET departure = COALESCE(%s, departure) "
                    "WHERE tr_id = %s AND stop_id = %s AND time_plan = %s",
                    (event.departure, *key),
                )
                continue
            contract = StopEvent(
                tr_id=tr_id,
                stop_id=int(event.visit_id),
                time_plan=event.planned_at,
                time_fact=event.arrival,
                delay_s=event.delay_sec,
            )
            conn.execute(
                "INSERT INTO stop_events "
                "(tr_id, stop_id, time_plan, time_fact, delay_s, "
                "available_at, departure, recovered) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    tr_id,
                    int(event.visit_id),
                    event.planned_at,
                    event.arrival,
                    event.delay_sec,
                    event.available_at,
                    event.departure,
                    event.recovered,
                ),
            )
            conn.execute(
                "INSERT INTO matcher_outbox (payload) VALUES (%s)",
                (Jsonb(contract.model_dump(mode="json")),),
            )
        conn.execute(
            "INSERT INTO matcher_state (tr_id, state) VALUES (%s, %s) "
            "ON CONFLICT (tr_id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
            (tr_id, Jsonb(_json_value(detector.snapshot(vehicle)))),
        )

    def publish_pending(self, redis, *, limit: int = 100) -> int:
        """At-least-once delivery; a crash after XADD may redeliver the event."""
        with self.pool.connection() as conn, conn.transaction():
            locked = conn.execute(
                "SELECT pg_try_advisory_xact_lock(hashtextextended('matcher:outbox', 0))"
            ).fetchone()[0]
            if not locked:
                return 0
            rows = conn.execute(
                "SELECT id, payload FROM matcher_outbox ORDER BY id LIMIT %s FOR UPDATE", (limit,)
            ).fetchall()
            for row_id, payload in rows:
                append(redis, STOP_EVENTS_STREAM, StopEvent.model_validate(payload))
                conn.execute("DELETE FROM matcher_outbox WHERE id = %s", (row_id,))
            return len(rows)

    def ready(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT last_telemetry_id FROM matcher_cursor WHERE id = true")

    def quality(self) -> dict:
        with self.pool.connection() as conn:
            return {
                "telemetry_cursor": conn.execute(
                    "SELECT last_telemetry_id FROM matcher_cursor WHERE id = true"
                ).fetchone()[0],
                "planned_visits": conn.execute("SELECT count(*) FROM stops_plan").fetchone()[0],
                "stop_events": conn.execute("SELECT count(*) FROM stop_events").fetchone()[0],
                "outbox_pending": conn.execute("SELECT count(*) FROM matcher_outbox").fetchone()[0],
            }

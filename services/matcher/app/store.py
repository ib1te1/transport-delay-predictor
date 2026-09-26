"""Read telemetry from the bus in batches, checkpoint detection, deliver stop events."""

import json
import logging
from collections import Counter
from dataclasses import asdict

from psycopg.types.json import Jsonb
from pydantic import ValidationError
from redis import RedisError

from app.dataset import Tick, Visit
from app.matching import Settings, StopDetector
from common.bus import STOP_EVENTS_STREAM, TELEMETRY_STREAM, append
from contracts import StopEvent, TelemetryRecord

log = logging.getLogger(__name__)

# Position before the first stream entry; used when no cursor row is saved.
START_ID = "0-0"
RESET_HINT = "scripts/reset-demo.sh (scripts/reset-demo.ps1 on Windows)"
SKIP_KINDS = ("malformed", "unknown_vehicle", "unplanned", "invalid_location")


class StateMismatch(RuntimeError):
    """Saved matcher state was built with other thresholds."""


class CursorMoved(RuntimeError):
    """The saved cursor changed under a running matcher, e.g. by a demo reset."""


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _id_key(entry_id: str) -> tuple[int, int]:
    ms, _, seq = entry_id.partition("-")
    return int(ms), int(seq or 0)


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, default=lambda item: item.isoformat()))


def _value(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):g}"
    return "missing" if value is None else str(value)


def _mismatch(saved: dict, configured: dict) -> str:
    changed = ", ".join(
        f"{name}: saved {_value(saved.get(name))}, configured {_value(configured.get(name))}"
        for name in sorted(saved.keys() | configured.keys())
        if saved.get(name) != configured.get(name)
    )
    return (
        f"matcher thresholds in config/assumptions.yaml differ from the saved state "
        f"({changed}). Restore the saved values or reset the demo with {RESET_HINT}."
    )


def _write_event(conn, event) -> bool:
    """Store a new event with its outbox row, or add the departure to a stored one.

    Returns False when a stored row has another arrival: a row of an earlier run.
    """
    key = (int(event.vehicle_id), int(event.visit_id), event.planned_at)
    existing = conn.execute(
        "SELECT time_fact FROM stop_events WHERE tr_id = %s AND stop_id = %s AND time_plan = %s",
        key,
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE stop_events SET departure = COALESCE(%s, departure) "
            "WHERE tr_id = %s AND stop_id = %s AND time_plan = %s",
            (event.departure, *key),
        )
        return existing[0] == event.arrival
    conn.execute(
        "INSERT INTO stop_events "
        "(tr_id, stop_id, time_plan, time_fact, delay_s, available_at, departure, recovered) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            *key,
            event.arrival,
            event.delay_sec,
            event.available_at,
            event.departure,
            event.recovered,
        ),
    )
    contract = StopEvent(
        tr_id=key[0],
        stop_id=key[1],
        time_plan=event.planned_at,
        time_fact=event.arrival,
        delay_s=event.delay_sec,
    )
    conn.execute(
        "INSERT INTO matcher_outbox (payload) VALUES (%s)",
        (Jsonb(contract.model_dump(mode="json")),),
    )
    return True


class Store:
    def __init__(
        self,
        pool,
        redis,
        settings: Settings,
        *,
        batch_size: int = 500,
        block_ms: int = 1000,
    ):
        self.pool = pool
        self.redis = redis
        self.settings = settings
        self.batch_size = batch_size
        self.block_ms = block_ms
        self.detector: StopDetector | None = None
        self.vehicles: dict[int, int] = {}
        self.planned: set[int] = set()
        self.skipped: Counter[str] = Counter()
        self.rejected_late_ticks = 0
        self._last_id = START_ID
        self._warned: set[tuple[str, object]] = set()

    def load(self) -> None:
        """Build the detector from the database; raises StateMismatch on other thresholds."""
        configured = asdict(self.settings)
        with self.pool.connection() as conn, conn.transaction():
            cursor = conn.execute("SELECT stream_id, settings FROM matcher_cursor").fetchone()
            if cursor is not None and cursor[1] != configured:
                raise StateMismatch(_mismatch(cursor[1], configured))
            vehicles = dict(conn.execute("SELECT unit_id, tr_id FROM vehicles").fetchall())
            plan = conn.execute(
                "SELECT tr_id, stop_id, time_plan, lon, lat FROM stops_plan"
            ).fetchall()
            states = conn.execute("SELECT tr_id, state FROM matcher_state").fetchall()
        visits = [Visit(str(tr), str(stop), at, lon, lat) for tr, stop, at, lon, lat in plan]
        detector = StopDetector(visits, self.settings)
        for tr_id, state in states:
            try:
                detector.restore(str(tr_id), state)
            except ValueError as exc:
                raise StateMismatch(
                    f"saved state of vehicle {tr_id} does not fit config/assumptions.yaml "
                    f"({exc}). Restore the saved values or reset the demo with {RESET_HINT}."
                ) from exc
        self.vehicles = vehicles
        self.planned = {row[0] for row in plan}
        self.detector = detector
        self._last_id = cursor[0] if cursor else START_ID
        log.info(
            "matcher loaded %d planned visits, %d vehicle states, cursor %s",
            len(plan),
            len(states),
            self._last_id,
        )

    def process_batch(self) -> int:
        """Match the next batch of stream entries; returns how many were read.

        Cursor, state of the touched vehicles, events and outbox rows commit
        together. After a failure the in-memory detector may be ahead of the
        database, so it is dropped and the next call reloads it and reads the
        same entries again.
        """
        if self.detector is None:
            self.load()
        reply = self.redis.xread(
            {TELEMETRY_STREAM: self._last_id}, count=self.batch_size, block=self.block_ms
        )
        entries = [
            (_decode(raw_id), fields) for _stream, batch in reply or [] for raw_id, fields in batch
        ]
        if not entries:
            return 0
        last_id = entries[-1][0]
        counts: Counter[str] = Counter()
        try:
            with self.pool.connection() as conn, conn.transaction():
                # FOR UPDATE locks nothing before the first cursor row exists.
                conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('matcher:input', 0))")
                row = conn.execute("SELECT stream_id FROM matcher_cursor").fetchone()
                saved = row[0] if row else START_ID
                if saved != self._last_id:
                    raise CursorMoved(f"saved cursor is {saved}, expected {self._last_id}")
                late = self.detector.rejected_late_ticks
                touched = set()
                for entry_id, fields in entries:
                    tick = self._tick(entry_id, fields, counts)
                    if tick is not None:
                        self.detector.feed(tick)
                        touched.add(tick.vehicle_id)
                late = self.detector.rejected_late_ticks - late
                stale = [e for e in self.detector.drain_events() if not _write_event(conn, e)]
                if stale:
                    log.warning(
                        "%d stop events collide with stored rows of another arrival: the "
                        "database holds a previous run. Reset the demo with %s",
                        len(stale),
                        RESET_HINT,
                    )
                self._save_state(conn, touched)
                self._save_cursor(conn, last_id)
        except Exception:
            self.detector = None
            raise
        self._last_id = last_id
        self.skipped.update(counts)
        self.rejected_late_ticks += late
        return len(entries)

    def _tick(self, entry_id: str, fields: dict, counts: Counter) -> Tick | None:
        raw = fields.get(b"data", fields.get("data"))
        try:
            record = TelemetryRecord.model_validate_json(raw)
        except ValidationError:
            self._skip(counts, "malformed", None, "malformed entry %s: %.200r", entry_id, raw)
            return None
        tr_id = record.tr_id if record.tr_id is not None else self.vehicles.get(record.unit_id)
        if tr_id is None:
            self._skip(counts, "unknown_vehicle", record.unit_id, "unknown unit %s", record.unit_id)
            return None
        if tr_id not in self.planned:
            self._skip(counts, "unplanned", tr_id, "no planned stops for vehicle %s", tr_id)
            return None
        tick = Tick(str(tr_id), record.event_time, record.lon, record.lat, record.speed_kmh)
        if not record.location_valid or not tick.valid:
            counts["invalid_location"] += 1
            # The detector keeps only its time, so later duplicates count as late.
            return Tick(tick.vehicle_id, tick.at, None, None, tick.speed)
        return tick

    def _skip(self, counts: Counter, kind: str, key: object, message: str, *args) -> None:
        counts[kind] += 1
        if (kind, key) not in self._warned:
            self._warned.add((kind, key))
            log.warning("skipping telemetry, " + message + " (logged once)", *args)

    def _save_state(self, conn, vehicles: set[str]) -> None:
        rows = [(int(v), Jsonb(_json_value(self.detector.snapshot(v)))) for v in sorted(vehicles)]
        if not rows:
            return
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO matcher_state (tr_id, state) VALUES (%s, %s) "
                "ON CONFLICT (tr_id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
                rows,
            )

    def _save_cursor(self, conn, stream_id: str) -> None:
        conn.execute(
            "INSERT INTO matcher_cursor (id, stream_id, settings) VALUES (true, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET stream_id = EXCLUDED.stream_id, "
            "settings = EXCLUDED.settings, updated_at = now()",
            (stream_id, Jsonb(asdict(self.settings))),
        )

    def check_stream(self) -> None:
        """Warn when the stream does not continue from the saved cursor."""
        if self._last_id == START_ID:
            return
        try:
            newest = self.redis.xrevrange(TELEMETRY_STREAM, count=1)
            oldest = self.redis.xrange(TELEMETRY_STREAM, count=1)
        except RedisError as exc:
            log.warning("could not check the telemetry stream: %s", exc)
            return
        cursor = _id_key(self._last_id)
        if not newest or _id_key(_decode(newest[0][0])) < cursor:
            log.warning(
                "telemetry stream ends before the saved cursor %s: the database holds "
                "a previous run. Reset the demo with %s",
                self._last_id,
                RESET_HINT,
            )
        elif _id_key(_decode(oldest[0][0])) > cursor:
            log.warning(
                "telemetry stream starts after the saved cursor %s: "
                "entries after it may have been trimmed",
                self._last_id,
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
            conn.execute("SELECT 1")

    def quality(self) -> dict:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT stream_id FROM matcher_cursor").fetchone()
            planned, events, outbox = conn.execute(
                "SELECT (SELECT count(*) FROM stops_plan), (SELECT count(*) FROM stop_events), "
                "(SELECT count(*) FROM matcher_outbox)"
            ).fetchone()
        cursor = row[0] if row else START_ID
        return {
            "stream_cursor": cursor,
            **self._stream_position(cursor),
            "planned_visits": planned,
            "stop_events": events,
            "outbox_pending": outbox,
            "rejected_late_ticks": self.rejected_late_ticks,
            "skipped": {kind: self.skipped[kind] for kind in SKIP_KINDS},
        }

    def _stream_position(self, cursor: str) -> dict:
        """Stream end against the cursor; None where Redis cannot tell."""
        try:
            newest = self.redis.xrevrange(TELEMETRY_STREAM, count=1)
        except RedisError:
            return {"stream_last_id": None, "unread": None, "lag_ms": None}
        if not newest:
            return {"stream_last_id": None, "unread": False, "lag_ms": None}
        last_id = _decode(newest[0][0])
        last, done = _id_key(last_id), _id_key(cursor)
        # No lag before the first batch or when the stream is behind the cursor.
        known = cursor != START_ID and last >= done
        return {
            "stream_last_id": last_id,
            "unread": last > done,
            "lag_ms": last[0] - done[0] if known else None,
        }

"""Shared setup for the end-to-end suite.

The suite talks to a running compose stack from the host: HTTP and
WebSocket to the published ports, raw TCP to the NDTP port, and
``docker compose exec`` into postgres and redis to look at tables and
streams. It does not start the stack itself; see tests/e2e/README.md.

If the api is not reachable the whole suite is skipped, unless
``E2E_REQUIRED=1`` is set, then it fails instead (CI sets it).
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
import synthetic
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSE_FILES = ["docker-compose.yml", "tests/e2e/docker-compose.e2e.yml"]


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


API_URL = _env("E2E_API_URL", "http://127.0.0.1:8000")
INGEST_URL = _env("E2E_INGEST_URL", "http://127.0.0.1:8001")
MATCHER_URL = _env("E2E_MATCHER_URL", "http://127.0.0.1:8002")
PREDICTOR_URL = _env("E2E_PREDICTOR_URL", "http://127.0.0.1:8003")
WEB_URL = _env("E2E_WEB_URL", "http://127.0.0.1:5173")
WS_URL = _env("E2E_WS_URL", API_URL.replace("http", "ws", 1) + "/ws")
NDTP_HOST = _env("E2E_NDTP_HOST", "127.0.0.1")
NDTP_PORT = int(_env("E2E_NDTP_PORT", "9201"))
READY_TIMEOUT = float(_env("E2E_READY_TIMEOUT", "240"))
# How long the pipeline may take to settle after a replay.
SETTLE_TIMEOUT = float(_env("E2E_SETTLE_TIMEOUT", "120"))
REQUIRED = os.environ.get("E2E_REQUIRED") == "1"
PG_USER = _env("POSTGRES_USER", "delay_predictor")
PG_DB = _env("POSTGRES_DB", "delay_predictor")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def wait_for[T](probe: Callable[[], T], what: str, timeout: float, interval: float = 1.0) -> T:
    """Call ``probe`` until it returns something truthy; fail with the last value or error."""
    deadline = time.monotonic() + timeout
    last: object = None
    while True:
        try:
            value = probe()
            if value:
                return value
            last = value
        except Exception as exc:  # the stack may still be starting
            last = repr(exc)
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}; last: {last}")
        time.sleep(interval)


class Compose:
    """docker compose with the e2e files, run from the repository root."""

    def __init__(self) -> None:
        files = os.environ.get("E2E_COMPOSE_FILES")
        paths = files.split(os.pathsep) if files else DEFAULT_COMPOSE_FILES
        self.env = {
            **os.environ,
            "COMPOSE_FILE": os.pathsep.join(paths),
            "COMPOSE_PATH_SEPARATOR": os.pathsep,
        }

    def run(
        self, *args: str, timeout: float = 300, check: bool = True, env: dict | None = None
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["docker", "compose", *args],
            cwd=ROOT,
            env={**self.env, **(env or {})},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if check and result.returncode != 0:
            pytest.fail(
                f"docker compose {' '.join(args)} exited with {result.returncode}\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return result

    def sql(self, query: str) -> list[list[str]]:
        """Rows of a query as lists of strings, through psql in the postgres container."""
        out = self.run(
            "exec", "-T", "postgres", "psql", "-U", PG_USER, "-d", PG_DB,
            "-v", "ON_ERROR_STOP=1", "-AtF", "\t", "-c", query,
        ).stdout  # fmt: skip
        return [line.split("\t") for line in out.splitlines() if line]

    def scalar(self, query: str) -> str:
        rows = self.sql(query)
        assert len(rows) == 1 and len(rows[0]) == 1, rows
        return rows[0][0]

    def count(self, table: str, where: str = "true") -> int:
        return int(self.scalar(f"SELECT count(*) FROM {table} WHERE {where}"))

    def redis(self, *args: str) -> Any:
        out = self.run("exec", "-T", "redis", "redis-cli", "--json", *args).stdout
        return json.loads(out) if out.strip() else None

    def xlen(self, stream: str) -> int:
        return int(self.redis("XLEN", stream))

    def stream(self, stream: str) -> list[dict]:
        """Every entry's ``data`` field of a stream, parsed; common.bus stores JSON there."""
        entries = self.redis("XRANGE", stream, "-", "+") or []
        result = []
        for _entry_id, fields in entries:
            pairs = dict(zip(fields[::2], fields[1::2], strict=True))
            result.append(json.loads(pairs["data"]))
        return result

    def run_replay(self) -> subprocess.CompletedProcess:
        # The command the README gives; -T because there is no terminal here.
        return self.run("--profile", "replay", "run", "--rm", "-T", "replay", check=False)

    def reset_demo(self) -> subprocess.CompletedProcess:
        """scripts/reset-demo, the flow the jury uses; compose picks our files from the env."""
        if sys.platform == "win32":
            script = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
            command = [*script, str(ROOT / "scripts" / "reset-demo.ps1")]
        else:
            command = ["sh", str(ROOT / "scripts" / "reset-demo.sh")]
        return subprocess.run(
            command,
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )


@dataclass
class Stack:
    compose: Compose
    http: httpx2.Client

    def get(self, url: str, **kwargs) -> httpx2.Response:
        return self.http.get(url, **kwargs)

    def json(self, url: str) -> Any:
        response = self.http.get(url)
        assert response.status_code == 200, (url, response.status_code, response.text)
        return response.json()

    def state(self) -> dict:
        return self.json(f"{API_URL}/api/state")

    def wait_ready(self, timeout: float = READY_TIMEOUT) -> None:
        """Every long-running service answers; matcher also says its workers run."""

        def ok(url: str) -> bool:
            return self.http.get(url).status_code == 200

        for name, url in (
            ("api", f"{API_URL}/health"),
            ("ingest", f"{INGEST_URL}/health"),
            ("matcher", f"{MATCHER_URL}/ready"),
            ("predictor", f"{PREDICTOR_URL}/health"),
            ("web", f"{WEB_URL}/"),
        ):
            wait_for(lambda url=url: ok(url), f"{name} at {url}", timeout)


@pytest.fixture(scope="session")
def stack() -> Stack:
    http = httpx2.Client(timeout=10)
    try:
        http.get(f"{API_URL}/health", timeout=5)
    except httpx2.HTTPError as exc:
        http.close()
        message = f"the compose stack is not reachable at {API_URL} ({exc!r})"
        if REQUIRED:
            pytest.fail(message + "; E2E_REQUIRED=1")
        pytest.skip(message + "; start it as tests/e2e/README.md shows")
    result = Stack(Compose(), http)
    result.wait_ready()
    yield result
    http.close()


class WsRecorder:
    """Collects every dashboard message on one WebSocket in a background thread."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.messages: list[dict] = []
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._opened = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._opened.wait(15):
            raise AssertionError(f"websocket {self.url} did not open: {self.error!r}")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(10)

    def _run(self) -> None:
        try:
            with connect(self.url, open_timeout=10) as ws:
                self._opened.set()
                while not self._stop.is_set():
                    try:
                        raw = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    self.messages.append(json.loads(raw))
        except BaseException as exc:  # reported by the tests that read the recording
            self.error = exc
            self._opened.set()


@dataclass
class Replayed:
    """What the session replay left behind for the tests to check."""

    ws: WsRecorder
    replay_output: str
    settled: dict = field(default_factory=dict)


def expected_stop_events() -> int:
    return len(synthetic.BUSES) * synthetic.VISITED_STOPS


def wait_settled(stack: Stack) -> dict:
    """Wait until every stage has consumed the whole synthetic replay."""
    compose = stack.compose
    fixes = len(synthetic.all_fixes())
    last = synthetic.last_event_time()

    def telemetry_stored() -> bool:
        return (
            compose.count("telemetry", "source = 'replay'") == fixes
            and compose.count("telemetry", "published_at IS NULL") == 0
        )

    def matcher_done() -> dict | None:
        quality = stack.json(f"{MATCHER_URL}/quality")
        done = (
            quality["unread"] is False
            and quality["stop_events"] == expected_stop_events()
            and quality["outbox_pending"] == 0
        )
        return quality if done else None

    def api_caught_up() -> dict | None:
        state = stack.state()
        if state["clock"] is None or parse_time(state["clock"]) < last:
            return None
        return state

    def every_bus_scored() -> bool:
        rows = compose.sql("SELECT DISTINCT tr_id FROM predictions")
        return {int(r[0]) for r in rows} >= {bus.tr_id for bus in synthetic.BUSES}

    wait_for(telemetry_stored, "telemetry stored and published", SETTLE_TIMEOUT)
    quality = wait_for(matcher_done, "matcher to emit every stop event", SETTLE_TIMEOUT)
    state = wait_for(api_caught_up, "api clock to reach the last point", SETTLE_TIMEOUT)
    wait_for(every_bus_scored, "a prediction for every bus", SETTLE_TIMEOUT)
    return {"quality": quality, "state": state}


@pytest.fixture(scope="session")
def replayed(stack: Stack) -> Replayed:
    """Play the synthetic dataset once per session from a clean run.

    A replay left by an earlier session is cleared with scripts/reset-demo
    first, so every session sees the same fresh run.
    """
    compose = stack.compose
    if compose.count("ingest_runs", "mode = 'replay'") or compose.xlen("telemetry"):
        result = compose.reset_demo()
        assert result.returncode == 0, result.stdout + result.stderr
        stack.wait_ready()
    recorder = WsRecorder(WS_URL)
    recorder.start()
    try:
        result = compose.run_replay()
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        settled = wait_settled(stack)
        # A few more seconds of clock and diff messages after the last tick.
        time.sleep(3)
    finally:
        recorder.stop()
    return Replayed(ws=recorder, replay_output=output, settled=settled)

"""Scenario 1: every service is up, migrated, seeded and serving."""

import re

import pytest
import synthetic
from conftest import API_URL, INGEST_URL, MATCHER_URL, PREDICTOR_URL, ROOT, WEB_URL, Stack


@pytest.mark.parametrize(
    ("service", "url"),
    [
        ("api", API_URL),
        ("ingest", INGEST_URL),
        ("matcher", MATCHER_URL),
        ("predictor", PREDICTOR_URL),
    ],
)
def test_health(stack: Stack, service: str, url: str) -> None:
    assert stack.json(f"{url}/health") == {"status": "ok", "service": service}


def test_matcher_ready(stack: Stack) -> None:
    assert stack.json(f"{MATCHER_URL}/ready") == {"status": "ready", "service": "matcher"}


def test_web_serves_the_dashboard_entry(stack: Stack) -> None:
    index = stack.get(f"{WEB_URL}/")
    assert index.status_code == 200
    assert '<div id="root">' in index.text
    entry = re.search(r'<script type="module" src="(/src/[^"]+)"', index.text)
    assert entry, index.text
    # The dev server compiles the entry on request, so this also proves it builds.
    script = stack.get(f"{WEB_URL}{entry[1]}")
    assert script.status_code == 200
    assert "createRoot" in script.text


def test_predictor_loads_the_committed_model(stack: Stack) -> None:
    model = stack.json(f"{PREDICTOR_URL}/model")
    assert model["kind"] == "model", model
    assert model["model_version"].startswith("current@")


def test_api_openapi_has_ws_messages(stack: Stack) -> None:
    schemas = stack.json(f"{API_URL}/openapi.json")["components"]["schemas"]
    assert "WsMessage" in schemas
    paths = stack.json(f"{API_URL}/openapi.json")["paths"]
    assert "/api/state" in paths and "/api/vehicles/{tr_id}" in paths


def _head_revision() -> str:
    revisions, parents = set(), set()
    for path in (ROOT / "migrations" / "versions").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        revisions.add(re.search(r'^revision: str = "([^"]+)"', text, re.M)[1])
        parent = re.search(r'^down_revision: str \| None = "([^"]+)"', text, re.M)
        if parent:
            parents.add(parent[1])
    heads = revisions - parents
    assert len(heads) == 1, heads
    return heads.pop()


def test_migrations_at_head(stack: Stack) -> None:
    assert stack.compose.scalar("SELECT version_num FROM alembic_version") == _head_revision()


def test_seed_loaded_the_synthetic_period(stack: Stack) -> None:
    compose = stack.compose
    assert compose.count("stops_plan") == len(synthetic.BUSES) * synthetic.PLANNED_STOPS
    pairs = {(int(u), int(t)) for u, t in compose.sql("SELECT unit_id, tr_id FROM vehicles")}
    expected = {(b.unit_id, b.tr_id) for b in synthetic.BUSES}
    assert pairs == expected | {(synthetic.IDLE_UNIT_ID, synthetic.IDLE_TR_ID)}
    # seed never loads facts, and the plan keeps the address column
    bus = synthetic.BUSES[0]
    row = compose.sql(
        f"SELECT lat, lon, address FROM stops_plan WHERE stop_id = {synthetic.stop_id(bus, 3)}"
    )
    assert row == [
        [str(synthetic.stop_lat(bus)), str(synthetic.stop_lon(3)), synthetic.address(bus, 3)]
    ]


def test_e2e_config_differs_from_the_main_one_only_where_intended() -> None:
    main = (ROOT / "config" / "system.yaml").read_text(encoding="utf-8")
    replay = (ROOT / "tests" / "e2e" / "config" / "system.yaml").read_text(encoding="utf-8")
    ingest = (ROOT / "tests" / "e2e" / "config" / "ingest-ndtp.yaml").read_text(encoding="utf-8")

    def body(text: str) -> str:
        # drop the e2e header comment above the main file's own first line
        return text[text.index("# Runtime settings") :]

    anchor = synthetic.NDTP_ANCHOR.strftime("%Y-%m-%dT%H:%M:%SZ")
    faster = (
        main.replace(
            "    telemetry_window_sec: 1800\n",
            "    # E2E keeps the full history to verify the model on synthetic trips.\n"
            "    telemetry_window_sec: 9000\n",
        )
        .replace(
            "  # Six and a half dataset hours play for about 39 real minutes.\n  speedup: 10\n",
            "  # E2E runs the synthetic period in about ten seconds.\n  speedup: 240\n",
        )
        .replace('  start_at: "2026-01-06T11:30:00Z"\n', "  start_at: null\n")
        .replace('  end_at: "2026-01-06T18:00:00Z"\n', "  end_at: null\n")
    )
    assert body(replay) == faster
    listening = faster.replace("  mode: replay\n", "  mode: emulator\n").replace(
        "    dataset_anchor: null\n", f"    dataset_anchor: {anchor}\n"
    )
    assert body(ingest) == listening

"""The dashboard reads its data from api (system-design.md §2, backend-design.md §7)."""

from conftest import API_URL, WEB_URL, Stack


def test_dashboard_source_uses_the_api(stack: Stack) -> None:
    app = stack.get(f"{WEB_URL}/src/App.tsx")
    client = stack.get(f"{WEB_URL}/src/client.ts")
    hook = stack.get(f"{WEB_URL}/src/useDashboard.ts")
    network = stack.get(f"{WEB_URL}/src/useNetwork.ts")
    assert all(response.status_code == 200 for response in (app, client, hook, network))
    assert "useDashboard" in app.text and "FleetMap" in app.text
    assert "useNetwork" in app.text
    assert "/api/state" in client.text
    assert "/api/vehicles/" in hook.text
    assert "/ws" in client.text
    assert "/api/stops" in network.text
    state = stack.state()
    assert isinstance(state["vehicles"], list) and isinstance(state["seq"], int)


def test_planned_network_has_ordered_stops_for_each_run(stack: Stack) -> None:
    response = stack.get(f"{API_URL}/api/stops")
    assert response.status_code == 200
    stops = response.json()
    assert stops
    routes: dict[int, list[int]] = {}
    for stop in stops:
        assert {"route_id", "stop_order", "stop_id", "lat", "lon", "time_plan"} <= stop.keys()
        routes.setdefault(stop["route_id"], []).append(stop["stop_order"])
    assert any(len(orders) >= 2 for orders in routes.values())
    assert all(orders == list(range(1, len(orders) + 1)) for orders in routes.values())


def test_api_lets_the_dashboard_origin_read_it(stack: Stack) -> None:
    origin = WEB_URL.replace("127.0.0.1", "localhost")
    response = stack.get(f"{API_URL}/api/state", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {origin, "*"}

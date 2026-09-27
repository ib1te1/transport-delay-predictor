"""The dashboard reads its data from api (system-design.md §2, backend-design.md §7)."""

from conftest import API_URL, WEB_URL, Stack


def test_dashboard_source_uses_the_api(stack: Stack) -> None:
    app = stack.get(f"{WEB_URL}/src/App.tsx")
    client = stack.get(f"{WEB_URL}/src/client.ts")
    hook = stack.get(f"{WEB_URL}/src/useDashboard.ts")
    assert all(response.status_code == 200 for response in (app, client, hook))
    assert "useDashboard" in app.text and "FleetMap" in app.text
    assert "/api/state" in client.text
    assert "/api/vehicles/" in hook.text
    assert "/ws" in client.text
    state = stack.state()
    assert isinstance(state["vehicles"], list) and isinstance(state["seq"], int)


def test_api_lets_the_dashboard_origin_read_it(stack: Stack) -> None:
    origin = WEB_URL.replace("127.0.0.1", "localhost")
    response = stack.get(f"{API_URL}/api/state", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {origin, "*"}

"""The dashboard reads its data from api (system-design.md §2, backend-design.md §7)."""

import pytest
from conftest import API_URL, WEB_URL, Stack


@pytest.mark.xfail(
    reason="web is still a scaffold: nothing in it calls /api/state or /ws", strict=False
)
def test_dashboard_source_uses_the_api(stack: Stack) -> None:
    app = stack.get(f"{WEB_URL}/src/App.tsx")
    assert app.status_code == 200
    assert "/api/state" in app.text and "/ws" in app.text


def test_api_lets_the_dashboard_origin_read_it(stack: Stack) -> None:
    origin = WEB_URL.replace("127.0.0.1", "localhost")
    response = stack.get(f"{API_URL}/api/state", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {origin, "*"}

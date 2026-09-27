"""The dashboard's origin may read the API (backend-design.md §7)."""

from fastapi.testclient import TestClient

from app.main import app

DASHBOARD = "http://localhost:5173"


def test_preflight_allows_the_dashboard_origin() -> None:
    response = TestClient(app).options(
        "/api/state",
        headers={"Origin": DASHBOARD, "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DASHBOARD


def test_get_answers_the_dashboard_origin() -> None:
    response = TestClient(app).get("/health", headers={"Origin": DASHBOARD})

    assert response.headers["access-control-allow-origin"] == DASHBOARD


def test_get_does_not_allow_another_origin() -> None:
    response = TestClient(app).get("/health", headers={"Origin": "http://example.com"})

    assert "access-control-allow-origin" not in response.headers


def test_preflight_rejects_a_write_method() -> None:
    response = TestClient(app).options(
        "/api/state",
        headers={"Origin": DASHBOARD, "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 400

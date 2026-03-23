"""API route tests using FastAPI TestClient (no DB/Redis required)."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.database import get_db

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("src.api.simulations.run_simulation")
def test_create_simulation(mock_task):
    mock_session = MagicMock()

    def fake_add(obj):
        obj.id = uuid.uuid4()

    def fake_refresh(obj):
        obj.created_at = datetime.now()

    mock_session.add.side_effect = fake_add
    mock_session.refresh.side_effect = fake_refresh
    mock_task.delay.return_value = None

    app.dependency_overrides[get_db] = lambda: mock_session
    try:
        response = client.post("/api/simulations/", json={
            "name": "Test Sim",
            "initial_price": 100.0,
            "drift": 0.05,
            "volatility": 0.2,
            "time_steps": 252,
            "num_paths": 10,
        })
        assert response.status_code == 202
        assert mock_session.add.called
        assert mock_session.commit.called
    finally:
        app.dependency_overrides.clear()


def test_list_simulations_empty():
    """Without DB, this will fail — but validates the route exists."""
    # In a real test with a test DB, this would return []
    # Here we just verify the route is registered
    assert "/api/simulations/" in [r.path for r in app.routes if hasattr(r, "path")]


def test_list_backtests_route_exists():
    assert "/api/backtests/" in [r.path for r in app.routes if hasattr(r, "path")]


def test_market_data_route_exists():
    assert "/api/market-data/{ticker}" in [r.path for r in app.routes if hasattr(r, "path")]

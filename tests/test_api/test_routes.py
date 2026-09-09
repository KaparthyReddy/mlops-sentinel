from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: MagicMock()

    app.state.registry = MagicMock()
    app.state.registry.get_version.return_value = "1"
    app.state.registry.has_challenger.return_value = False

    app.state.predictor = MagicMock()
    app.state.predictor.predict.return_value = {
        "request_id": "abc-123",
        "model_alias": "champion",
        "model_version": "1",
        "prediction": 0,
        "probability": 0.05,
    }

    app.state.metrics_service = MagicMock()
    app.state.metrics_service.get_prediction_summary.return_value = {
        "window_minutes": 60, "total_predictions": 0, "by_model_alias": {}, "average_probability": None,
    }
    app.state.metrics_service.compute_and_store_drift.return_value = {
        "status": "INSUFFICIENT_DATA", "sample_size": 2,
    }

    app.state.retrain_trigger = MagicMock()

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_predict_endpoint(client):
    payload = {f"feature_{i}": 0.5 for i in range(10)}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_alias"] == "champion"
    assert body["prediction"] == 0


def test_drift_endpoint_insufficient_data(client):
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    assert response.json()["status"] == "INSUFFICIENT_DATA"


def test_promote_challenger_fails_without_challenger(client):
    response = client.post("/registry/promote-challenger")
    assert response.status_code == 400

import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # In demo mode (no model file), model_loaded should be False
    assert "model_loaded" in data


def test_predict_endpoint_no_model():
    """Without a loaded model, /predict should return 503."""
    payload = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def test_predict_endpoint_invalid_input():
    """Missing the required 'features' key should return 422."""
    payload = {"invalid_key": [1.0, 2.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

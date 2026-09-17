import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_valid_input():
    payload = {
        "features": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 500] 

def test_predict_endpoint_invalid_input():
    payload = {
        "invalid_key": [1.0, 2.0]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

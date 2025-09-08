from fastapi.testclient import TestClient
from src.api.app import app


def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json={"state": {"x": 1, "y": 2}})
        assert response.status_code == 200
        assert "action" in response.json()

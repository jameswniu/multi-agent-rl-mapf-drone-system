from fastapi.testclient import TestClient
from src.api.app import app, StateInput

client = TestClient(app)

def test_predict():
    payload = StateInput(state={"x": 1, "y": 2})
    response = client.post("/predict", json=payload.model_dump())
    assert response.status_code == 200
    assert "action" in response.json()

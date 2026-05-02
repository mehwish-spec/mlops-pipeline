from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    response = client.post("/predict", json={
        "features": [0.5, 0.1, -0.3, 0.8, 0.2, 0.6, -0.1, 0.4, 0.7, -0.2]
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "reports" in response.json()

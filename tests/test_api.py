from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "Real Estate Price Prediction API" in response.json()["message"]

def test_predict():
    test_data = {
        "bed": 3,
        "bath": 2,
        "acre_lot": 0.5,
        "house_size": 2000,
        "total_rooms": 5
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert "model_version" in response.json()
    assert "model_stage" in response.json() 
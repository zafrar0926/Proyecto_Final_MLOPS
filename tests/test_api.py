from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

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
    data = response.json()
    assert isinstance(data.get("predicted_price"), (int, float))
    assert isinstance(data.get("model_version"), str)
    assert isinstance(data.get("model_stage"), str) 
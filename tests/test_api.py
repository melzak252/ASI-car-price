import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_bmw(client):
    payload = {
        "vehicle_brand": "BMW", "vehicle_model": "X5", "production_year": 2020,
        "mileage_km": 50000, "power_hp": 300, "displacement_cm3": 3000,
        "fuel_type": "Diesel", "drive": "4x4 (permanent)", "type": "SUV",
        "doors_number": 5, "condition": "Used",
        "features": ["ABS", "Cruise control", "LED lights"],
    }
    resp = client.post("/predict", json=payload)
    if resp.status_code == 200:
        assert resp.json()["predicted_price_pln"] > 0
    elif resp.status_code == 503:
        pass


def test_predict_validation(client):
    resp = client.post("/predict", json={"fuel_type": "Diesel"})
    assert resp.status_code == 422


def test_predictions(client):
    resp = client.get("/predictions")
    assert resp.status_code == 200


def test_drift(client):
    resp = client.get("/drift")
    assert resp.status_code in (200, 503)

from io import StringIO
import json
import math

from fastapi.testclient import TestClient
import pytest

from app.main import app, handler

client = TestClient(app)
household = {
    "rooms": 3, "members": 4, "ac_hours_per_day": 6, "avg_temp_c": 32,
    "house_area_sqft": 1100, "solar_kw": 1.5, "connection": "domestic", "feeder": "Feeder-1",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("changes", [{}, {"avg_temp_c": 65, "ac_hours_per_day": 0}])
def test_predict(changes):
    response = client.post("/predict", json={**household, **changes})
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


@pytest.mark.parametrize("changes", [{"avg_temp_c": 65.1}, {"ac_hours_per_day": -0.1},
                                      {"record_id": "unexpected"}])
def test_reject_bad_inputs(changes):
    assert client.post("/predict", json={**household, **changes}).status_code == 422


def test_missing_input():
    assert client.post("/predict", json={k: v for k, v in household.items() if k != "rooms"}).status_code == 422


def test_prediction_emits_one_json_event(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(handler, "stream", stream)
    assert client.post("/predict", json=household).status_code == 200
    assert client.post("/predict", json={**household, "avg_temp_c": 67}).status_code == 422
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "prediction", "level": "INFO"}

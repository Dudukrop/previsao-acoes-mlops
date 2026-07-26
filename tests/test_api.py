import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_model_info_ok(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    assert "evaluation_metrics" in resp.json()


def test_predict_rejects_empty_closes(client):
    resp = client.post("/predict", json={"closes": []})
    assert resp.status_code == 422


def test_predict_rejects_negative_close(client):
    resp = client.post("/predict", json={"closes": [10.0, -5.0]})
    assert resp.status_code == 422


def test_predict_rejects_wrong_length(client):
    resp = client.post("/predict", json={"closes": [10.0, 11.0, 12.0]})
    assert resp.status_code == 400


def test_predict_by_ticker_rejects_unknown_ticker(client):
    resp = client.post("/predict/by-ticker", json={"ticker": "MSFT"})
    assert resp.status_code == 400

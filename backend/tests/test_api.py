import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app

PARITY = json.load(open(Path(__file__).resolve().parents[2] / "frontend/src/data/research/parity.json"))
SHAP = json.load(open(Path(__file__).resolve().parents[2] / "frontend/src/data/research/shap.json"))
F = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "sex", "fbs", "exang", "cp", "restecg", "slope", "thal"]


def body(row):
    d = dict(zip(F, row))
    for k in ("sex", "fbs", "exang", "cp", "restecg", "slope"):
        d[k] = int(d[k])
    for k in ("ca", "thal"):
        d[k] = None if d[k] is None else int(d[k])
    return {"features": d}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_matches_sklearn_fixture(client):
    for i in range(0, 61, 7):  # in-range test rows
        r = client.post("/api/v1/predict", json=body(PARITY["rows"][i]))
        assert r.status_code == 200, r.text
        for p in r.json()["predictions"]:
            assert abs(p["probability"] - PARITY["expected"][p["model"]][i]) < 1e-9


def test_missing_ca_thal_allowed(client):
    row = list(PARITY["rows"][0]); row[F.index("ca")] = None; row[F.index("thal")] = None
    assert client.post("/api/v1/predict", json=body(row)).status_code == 200


@pytest.mark.parametrize("field,value", [("age", 5), ("age", "abc"), ("sex", 2), ("cp", 9), ("trestbps", 10), ("oldpeak", -1), ("thal", 5)])
def test_invalid_inputs_rejected(client, field, value):
    b = body(PARITY["rows"][0]); b["features"][field] = value
    assert client.post("/api/v1/predict", json=b).status_code == 422


def test_unknown_field_rejected(client):
    b = body(PARITY["rows"][0]); b["features"]["lifestyle_salt"] = 3
    assert client.post("/api/v1/predict", json=b).status_code == 422


def test_rf_shap_reproduces_published_values(client):
    for i in (0, 5, 30):
        r = client.post("/api/v1/explain/rf", json=body(PARITY["rows"][i])).json()
        got = {c["feature"]: c["contribution"] for c in r["contributions"]}
        for j, f in enumerate(SHAP["features"]):
            assert abs(got[f] - SHAP["S_rf"][i][j]) < 1e-6
        assert abs(r["base_value"] + sum(got.values()) - r["prediction_check"]) < 0.05


def test_lr_shap_additive(client):
    r = client.post("/api/v1/explain/lr", json=body(PARITY["rows"][3])).json()
    assert abs(r["base_value"] + sum(c["contribution"] for c in r["contributions"]) - r["prediction_check"]) < 1e-9


def test_unknown_explain_model(client):
    assert client.post("/api/v1/explain/svm", json=body(PARITY["rows"][0])).status_code == 404


def test_lr_shap_reproduces_published_values(client):
    for i in (0, 7, 40):
        r = client.post("/api/v1/explain/lr", json=body(PARITY["rows"][i])).json()
        got = {c["feature"]: c["contribution"] for c in r["contributions"]}
        for j, f in enumerate(SHAP["features"]):
            assert abs(got[f] - SHAP["S_lr"][i][j]) < 1e-9

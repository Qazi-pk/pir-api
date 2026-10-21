import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, fit_table

client = TestClient(app)


def _csv(df):
    return ("data.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (400, 3))
    return pd.DataFrame({"m": X[:, 0], "v": X[:, 1], "c": X[:, 2],
                         "y": X[:, 0] * X[:, 1] / (X[:, 1] + X[:, 2])})


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_page():
    assert "PIR formula finder" in client.get("/").text


def test_fit_recovers_rational_law(data):
    r = client.post("/fit", files={"file": _csv(data)}, data={"target": "y", "gates": "all"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["found_by"] == "rational"
    assert j["r2_heldout_rows"] > 0.999999
    assert set(j["inputs"]) == {"m", "v", "c"}


def test_missing_target_is_422(data):
    r = client.post("/fit", files={"file": _csv(data)}, data={"target": "nope"})
    assert r.status_code == 422 and "not found" in r.json()["detail"]


def test_bad_gates_is_422(data):
    r = client.post("/fit", files={"file": _csv(data)}, data={"target": "y", "gates": "bogus"})
    assert r.status_code == 422


def test_too_few_rows():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    with pytest.raises(ValueError):
        fit_table(df, "y")


def test_non_numeric_column():
    df = pd.DataFrame({"x": ["a"] * 30, "y": np.arange(30.0)})
    with pytest.raises(ValueError):
        fit_table(df, "y")

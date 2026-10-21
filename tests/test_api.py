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


@pytest.mark.parametrize("name,target", [("pendulum", "period_s"), ("lens", "focal_cm"),
                                         ("gravity", "force"), ("decay", "N")])
def test_examples_recover_their_law(name, target):
    r = client.get(f"/examples/{name}.csv")
    assert r.status_code == 200
    df = pd.read_csv(io.StringIO(r.text))
    j = fit_table(df, target)
    assert j["r2_heldout_rows"] > 0.99


def test_unknown_example_is_404():
    assert client.get("/examples/nope.csv").status_code == 404


def test_page_has_column_picker_and_examples():
    html = client.get("/").text
    assert 'select id="target"' in html and "data-ex=" in html and "__EXAMPLES__" not in html


def test_readable_formula_rounds_constants_and_keeps_column_names():
    from app.main import _readable
    assert _readable("420.0432477890403*RR**(1/3)") == "420.043*RR**(1/3)"
    assert _readable("1.0000000000000009*N0*exp(-time/tau)") == "N0*exp(-time/tau)"

import pytest
from fastapi.testclient import TestClient

from readmission.predict import BUNDLE_PATH

pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(),
                                reason="run scripts/04_finalize.py first")


def test_predict_returns_a_probability():
    from api.app import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    low = client.post("/predict", json={}).json()
    high = client.post("/predict", json={"number_inpatient": 6, "n_prior_encounters": 8,
                                         "number_emergency": 4, "time_in_hospital": 12}).json()
    assert 0 <= low["probability_readmission_30d"] <= 1
    # heavy prior hospital use must raise the risk (sanity check of the business logic)
    assert high["probability_readmission_30d"] > low["probability_readmission_30d"]

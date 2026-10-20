"""Fast checks of the leakage-sensitive pieces. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from readmission.calibration import ProbabilityCalibrator
from readmission.data import add_prior_encounters
from readmission.evaluation import expected_calibration_error
from readmission.features import ReadmissionFeatures, group_icd9


@pytest.mark.parametrize("code,expected", [
    ("250.83", "Diabetes"), ("428", "Circulatory"), ("V57", "Supplementary"),
    ("E888", "External"), ("?", "Unknown"), ("786", "Symptoms"),
])
def test_icd9_grouping(code, expected):
    assert group_icd9(code) == expected


def test_prior_encounters_only_count_the_past():
    df = pd.DataFrame({"encounter_id": [30, 10, 20, 5], "patient_nbr": [1, 1, 1, 2]})
    out = add_prior_encounters(df)
    # rows keep their order; patient 1's visits in time order are 10, 20, 30
    assert out["n_prior_encounters"].tolist() == [2, 0, 1, 0]


def test_rare_categories_are_learned_on_fit_data_only(api_like_rows):
    tr = api_like_rows.copy()
    tr.loc[:, "medical_specialty"] = ["Cardiology"] * 60 + ["Rare"] * 5
    fe = ReadmissionFeatures(min_category_count=50).fit(tr)
    new = tr.head(2).copy()
    new["medical_specialty"] = ["Rare", "NeverSeen"]
    assert fe.transform(new)["medical_specialty"].tolist() == ["Other", "Other"]


def test_imputer_learns_from_training_rows_only(api_like_rows):
    """The median used to fill a missing value must come from the training fold only."""
    from readmission.config import load_config
    from readmission.pipelines import build_pipeline

    train = api_like_rows.copy()
    train["time_in_hospital"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 6 + [np.nan] * 5
    y = np.array([0, 1] * 32 + [0])
    held_out = api_like_rows.head(10).copy()
    held_out["time_in_hospital"] = 14            # very different values on unseen rows

    pipe = build_pipeline("logreg", {"C": 1.0}, load_config()).fit(train, y)
    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    numeric = list(pipe.named_steps["preprocess"].transformers_[0][2])
    learned = imputer.statistics_[numeric.index("time_in_hospital")]
    assert learned == np.nanmedian(train["time_in_hospital"]) == 5.5
    pipe.predict_proba(held_out)                  # transforming unseen rows must not refit
    assert imputer.statistics_[numeric.index("time_in_hospital")] == 5.5


def test_calibrators_are_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.6, 2000)
    y = rng.binomial(1, p / 2)
    for method in ["sigmoid", "isotonic"]:
        q = ProbabilityCalibrator(method).fit(p, y).transform(np.sort(p))
        assert np.all(np.diff(q) >= -1e-12) and q.min() >= 0 and q.max() <= 1


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat([0.1, 0.5], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)]])
    assert expected_calibration_error(y, p) < 1e-9

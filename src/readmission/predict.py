"""Load the frozen model bundle and score raw encounter rows (used by the API and tests)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

BUNDLE_PATH = PROJECT_ROOT / "models" / "final_model.joblib"


@lru_cache(maxsize=1)
def load_bundle(path: str = str(BUNDLE_PATH)) -> dict:
    return joblib.load(path)


def score(rows: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Raw rows -> calibrated 30-day readmission probability and follow-up flag."""
    bundle = bundle or load_bundle()
    p_raw = bundle["pipeline"].predict_proba(rows)[:, 1]
    p_cal = bundle["calibrator"].transform(p_raw)
    return pd.DataFrame({
        "probability_readmission_30d": np.round(p_cal, 4),
        "flag_for_follow_up": p_cal >= bundle["threshold"],
        "threshold": bundle["threshold"],
    }, index=rows.index)

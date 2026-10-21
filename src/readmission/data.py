"""Data loading, cohort definition, binary target and the honest dev/test split."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedGroupKFold

from .config import PROJECT_ROOT

CACHE = PROJECT_ROOT / "data" / "diabetes_raw.joblib"


def load_raw(cfg: dict) -> pd.DataFrame:
    """Raw OpenML frame (features + original 3-class target), cached locally."""
    if CACHE.exists():
        return joblib.load(CACHE)
    bunch = fetch_openml(data_id=cfg["data"]["openml_id"], as_frame=True)
    df = bunch.data.copy()
    df[cfg["data"]["target_column"]] = bunch.target.astype(str)
    CACHE.parent.mkdir(exist_ok=True)
    joblib.dump(df, CACHE)
    return df


def add_prior_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """Number of earlier encounters of the same patient present in the hospital records.

    Prediction-time test: at discharge the hospital knows the patient's past stays,
    never the future ones. encounter_id increases with time, so we only count rows of
    the same patient with a smaller id. A count over ALL rows of the patient would leak
    the future (a later row *is* the readmission) and is deliberately not built.
    """
    df = df.sort_values("encounter_id")
    df["n_prior_encounters"] = df.groupby("patient_nbr").cumcount()
    return df.sort_index()


def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return X, binary y and patient groups for the modelling cohort."""
    dcfg = cfg["data"]
    df = add_prior_encounters(load_raw(cfg))

    # Cohort: patients discharged dead or to hospice cannot (or are not meant to) come back.
    # Keeping them would teach the model "discharge = expired -> never readmitted", a
    # trivially true rule that is useless for bed planning.
    df = df[~df["discharge_disposition_id"].isin(dcfg["excluded_discharge_ids"])]
    # 3 rows have gender "Unknown/Invalid": too few to learn from, would break the audit.
    df = df[df["gender"] != "Unknown/Invalid"].reset_index(drop=True)

    y = (df.pop(dcfg["target_column"]) == dcfg["positive_label"]).astype(int)
    groups = df[dcfg["group_column"]].copy()
    return df, y, groups


def dev_test_split(X, y, groups, cfg):
    """Patient-disjoint, stratified 80/20 split. The test part is touched once, at the end."""
    n_folds = int(round(1 / cfg["data"]["test_size"]))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])
    dev_idx, test_idx = next(sgkf.split(X, y, groups))
    assert not set(groups.iloc[dev_idx]) & set(groups.iloc[test_idx])
    pick = lambda s, i: s.iloc[i].reset_index(drop=True)
    return (pick(X, dev_idx), pick(X, test_idx), pick(y, dev_idx), pick(y, test_idx),
            pick(groups, dev_idx), pick(groups, test_idx))


def cv_splitter(cfg) -> StratifiedGroupKFold:
    """Same patient-grouped, stratified folds for every experiment (fair comparison)."""
    return StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                                random_state=cfg["seed"])


def load_splits(cfg):
    X, y, groups = build_dataset(cfg)
    return dev_test_split(X, y, groups, cfg)


def cv_folds(cfg, X, y, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialised folds, so every model/trial sees exactly the same partitions."""
    return list(cv_splitter(cfg).split(X, y, groups))

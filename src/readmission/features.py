"""Feature engineering as a scikit-learn transformer, so it is fitted inside every CV fold.

Stateless features (ratios, counts, ICD-9 groups) do not learn anything from data.
The only learned rule, which rare categories to pool into "Other", is learned in
``fit`` on the training fold only, then reused unchanged on validation/test/API rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide.metformin", "glipizide.metformin", "glimepiride.pioglitazone",
    "metformin.rosiglitazone", "metformin.pioglitazone",
]

RAW_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]

ENGINEERED_NUMERIC = [
    "age_years",            # ordinal age bracket -> midpoint (keeps the order, 1 column)
    "total_prior_visits",   # outpatient + emergency + inpatient visits in the past year
    "n_meds_active",        # diabetes drugs the patient is on (not "No")
    "n_meds_changed",       # drugs whose dose went Up/Down during the stay
    "meds_per_day",         # treatment intensity normalised by length of stay
    "labs_per_day",         # monitoring intensity normalised by length of stay
    "a1c_tested",           # HbA1c measured during the stay (care-quality signal)
    "glucose_tested",       # max glucose serum measured
    "diabetes_primary",     # primary diagnosis is diabetes itself (ICD-9 250.xx)
]

# Integer ids are codes, not quantities (Session 1: storage type != semantic type).
ID_CATEGORICAL = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

LOW_CARD_CATEGORICAL = (
    ["gender", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
     "change", "diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group"]
    + ID_CATEGORICAL + MEDICATIONS
)

# ~700 distinct ICD-9 codes each: one-hot would explode, grouping loses detail,
# so the raw codes are additionally target-encoded (cross-fitted, see pipelines.py).
HIGH_CARD_CATEGORICAL = ["diag_1", "diag_2", "diag_3"]

AGE_MIDPOINT = {f"[{10 * i}-{10 * (i + 1)})": 10 * i + 5 for i in range(10)}

ICD9_RANGES = [
    (1, 139, "Infectious"), (140, 239, "Neoplasms"), (240, 279, "Endocrine"),
    (280, 289, "Blood"), (290, 319, "Mental"), (320, 389, "Nervous"),
    (390, 459, "Circulatory"), (460, 519, "Respiratory"), (520, 579, "Digestive"),
    (580, 629, "Genitourinary"), (630, 679, "Pregnancy"), (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"), (740, 759, "Congenital"), (760, 779, "Perinatal"),
    (780, 799, "Symptoms"), (800, 999, "Injury"),
]


def group_icd9(code) -> str:
    """Map an ICD-9 code to its chapter (CDC/CMS ranges); 250.xx is split out as Diabetes."""
    code = str(code)
    if code in ("?", "nan", "None"):
        return "Unknown"
    if code.startswith("V"):
        return "Supplementary"
    if code.startswith("E"):
        return "External"
    try:
        num = float(code)
    except ValueError:
        return "Unknown"
    if int(num) == 250:
        return "Diabetes"
    for low, high, name in ICD9_RANGES:
        if low <= int(num) <= high:
            return name
    return "Other"


def feature_names(use_prior_encounters: bool) -> tuple[list, list, list]:
    numeric = RAW_NUMERIC + ENGINEERED_NUMERIC
    if use_prior_encounters:
        numeric = numeric + ["n_prior_encounters"]
    return numeric, list(LOW_CARD_CATEGORICAL), list(HIGH_CARD_CATEGORICAL)


class ReadmissionFeatures(BaseEstimator, TransformerMixin):
    """Clean raw columns, add engineered features, pool rare categories (learned in fit)."""

    def __init__(self, min_category_count: int = 50, use_prior_encounters: bool = True):
        self.min_category_count = min_category_count
        self.use_prior_encounters = use_prior_encounters

    def _base(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        for col in RAW_NUMERIC:
            out[col] = pd.to_numeric(X[col], errors="coerce")
        if self.use_prior_encounters:
            out["n_prior_encounters"] = pd.to_numeric(
                X.get("n_prior_encounters", 0), errors="coerce")

        out["age_years"] = X["age"].astype(str).map(AGE_MIDPOINT)
        out["total_prior_visits"] = (out["number_outpatient"] + out["number_emergency"]
                                     + out["number_inpatient"])
        meds = X[MEDICATIONS].astype(str)
        out["n_meds_active"] = (meds != "No").sum(axis=1)
        out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
        stay = out["time_in_hospital"].clip(lower=1)
        out["meds_per_day"] = out["num_medications"] / stay
        out["labs_per_day"] = out["num_lab_procedures"] / stay
        out["a1c_tested"] = (X["A1Cresult"].astype(str) != "None").astype(int)
        out["glucose_tested"] = (X["max_glu_serum"].astype(str) != "None").astype(int)

        for col in HIGH_CARD_CATEGORICAL:
            raw = X[col].astype(str).replace({"?": "Unknown", "nan": "Unknown"})
            out[col] = raw
            out[f"{col}_group"] = raw.map(group_icd9)
        out["diabetes_primary"] = (out["diag_1_group"] == "Diabetes").astype(int)

        for col in LOW_CARD_CATEGORICAL:
            if col in out:
                continue
            # "?" is how this dataset writes "missing": make it an explicit category,
            # because missingness itself can carry signal (e.g. no payer recorded).
            out[col] = X[col].astype(str).replace({"?": "Unknown", "nan": "Unknown"})
        return out

    def fit(self, X, y=None):
        base = self._base(X)
        self.kept_categories_ = {}
        for col in LOW_CARD_CATEGORICAL:
            counts = base[col].value_counts()
            self.kept_categories_[col] = set(counts[counts >= self.min_category_count].index)
        numeric, low, high = feature_names(self.use_prior_encounters)
        self.feature_names_out_ = numeric + low + high
        return self

    def transform(self, X):
        base = self._base(X)
        for col, kept in self.kept_categories_.items():
            base[col] = base[col].where(base[col].isin(kept), "Other")
        return base[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

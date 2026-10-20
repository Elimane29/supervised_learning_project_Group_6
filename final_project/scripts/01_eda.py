"""EDA on the DEVELOPMENT split only: figures + tables that justify the modelling choices.

Run: python scripts/01_eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from readmission.config import load_config
from readmission.data import build_dataset, dev_test_split, load_raw
from readmission.features import ReadmissionFeatures

cfg = load_config()
FIG, TAB = cfg["paths"]["figures"], cfg["paths"]["tables"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# --- 1. Raw target and cohort definition ------------------------------------------------
raw = load_raw(cfg)
X, y, groups = build_dataset(cfg)
cohort = pd.DataFrame({
    "step": ["raw encounters", "after removing expired/hospice + invalid gender"],
    "rows": [len(raw), len(X)],
    "patients": [raw["patient_nbr"].nunique(), groups.nunique()],
})
cohort.to_csv(TAB / "eda_cohort.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
raw["readmitted"].value_counts().reindex(["NO", ">30", "<30"]).plot.bar(ax=ax[0], color="#888")
ax[0].set_title("Original 3-class target")
y.value_counts().sort_index().rename({0: "0: no / >30 d", 1: "1: <30 d"}).plot.bar(
    ax=ax[1], color=["#888", "#c0392b"])
ax[1].set_title(f"Binary target (prevalence {y.mean():.1%})")
for a in ax:
    a.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(FIG / "eda_target.png")

# --- From here on, only the development split is explored (the test set stays unseen) -----
X_dev, _, y_dev, _, g_dev, _ = dev_test_split(X, y, groups, cfg)
dev = X_dev.copy()
dev["target"] = y_dev.values

# Missingness ("?" is the dataset's missing marker, "None" means "test not done").
miss = pd.DataFrame({
    "pct_question_mark": (X_dev.astype(str) == "?").mean(),
    "pct_None": (X_dev.astype(str) == "None").mean(),
}).query("pct_question_mark > 0 or pct_None > 0").sort_values("pct_question_mark",
                                                                ascending=False)
miss.to_csv(TAB / "eda_missingness.csv", float_format="%.3f")

dup = pd.Series({
    "duplicated_rows": int(X_dev.duplicated().sum()),
    "patients": int(g_dev.nunique()),
    "encounters_per_patient_max": int(g_dev.value_counts().max()),
    "share_patients_multiple_encounters": float((g_dev.value_counts() > 1).mean()),
})
dup.to_csv(TAB / "eda_integrity.csv")

# Engineered view (fitted on dev only) to inspect the features we will feed the models.
feat = ReadmissionFeatures(cfg["features"]["min_category_count"]).fit_transform(X_dev)
feat["target"] = y_dev.values


def rate_by(df, col, min_n=200, top=None):
    t = df.groupby(col, observed=True)["target"].agg(n="size", rate="mean")
    # ordinal/count features keep their natural order; categories are sorted by rate
    t = t[t["n"] >= min_n]
    t = t.sort_index() if pd.api.types.is_numeric_dtype(df[col]) else t.sort_values("rate")
    return t.tail(top) if top else t


panels = [
    ("number_inpatient", "Inpatient visits, past year", feat.assign(
        number_inpatient=feat["number_inpatient"].clip(upper=6))),
    ("n_prior_encounters", "Prior encounters in records", feat.assign(
        n_prior_encounters=feat["n_prior_encounters"].clip(upper=6))),
    ("discharge_disposition_id", "Discharge disposition id", feat),
    ("diag_1_group", "Primary diagnosis chapter", feat),
    ("age_years", "Age (bracket midpoint)", feat),
    ("A1Cresult", "HbA1c result", feat),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
base = y_dev.mean()
for ax, (col, title, df) in zip(axes.ravel(), panels):
    t = rate_by(df, col, top=12)
    ax.barh(t.index.astype(str), t["rate"], color="#2c7fb8")
    ax.axvline(base, color="#c0392b", ls="--", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("30-day readmission rate")
fig.suptitle("Readmission rate by feature (dev set; red = overall rate)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "eda_signal.png")

# Numeric features: mean by class (the team's boxplots showed heavy overlap).
num_cols = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses",
            "number_inpatient", "number_emergency", "total_prior_visits",
            "n_prior_encounters", "n_meds_changed"]
feat.groupby("target")[num_cols].mean().T.rename(
    columns={0: "mean_class0", 1: "mean_class1"}).to_csv(TAB / "eda_numeric_by_class.csv",
                                                         float_format="%.3f")

# Sensitive attribute: base rates and group sizes, needed to read the fairness audit.
fair = []
for col in ["race", "gender", "age"]:
    t = dev.groupby(col, observed=True)["target"].agg(n="size", readmission_rate="mean")
    t["attribute"] = col
    fair.append(t.reset_index().rename(columns={col: "group"}))
pd.concat(fair).to_csv(TAB / "eda_sensitive_base_rates.csv", index=False, float_format="%.4f")

print(cohort.to_string(index=False))
print(f"dev prevalence {y_dev.mean():.4f}")
print(dup.to_string())
print(miss.head(8).to_string())
print(pd.read_csv(TAB / "eda_numeric_by_class.csv").to_string(index=False))

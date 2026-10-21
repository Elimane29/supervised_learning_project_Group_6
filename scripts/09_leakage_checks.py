"""Make the leakage audit measurable: rebuild the leaky variants on purpose and show how
much they would have inflated the cross-validated score (development split only).

  temporal  n_prior_encounters = earlier stays only (ours)
            vs  all other stays of the patient, past AND future (leaky)
            vs  future stays only (pure leakage: a later stay is often the readmission)
  group     StratifiedGroupKFold by patient (ours) vs StratifiedKFold (patients mixed)

Same model (LightGBM, default params), same metrics. A higher score for a leaky variant is
NOT a better model: in production the future is unknown and new patients are unseen.

Run: python scripts/09_leakage_checks.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from readmission.config import load_config
from readmission.data import cv_folds, load_splits
from readmission.evaluation import evaluate_cv, log_run, setup_mlflow
from readmission.pipelines import build_pipeline, default_params

warnings.filterwarnings("ignore")
cfg = load_config()
setup_mlflow(cfg)
X, _, y, _, groups, _ = load_splits(cfg)            # test split not used
grouped = cv_folds(cfg, X, y, groups)
mixed = list(StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True,
                             random_state=cfg["seed"]).split(X, y))

# Variants of the history feature. `past` comes from data.add_prior_encounters (all earlier
# stays in the records, including ones later excluded from the cohort, which is legitimate
# history). `future` counts the patient's LATER stays in this cohort: information that does
# not exist at discharge.
past = X["n_prior_encounters"]
rank = X.sort_values("encounter_id").groupby("patient_nbr").cumcount().sort_index()
count = X.groupby("patient_nbr")["encounter_id"].transform("count")
future = count - 1 - rank
assert (future >= 0).all() and (past >= rank).all()  # past-only never looks forward
variants = {
    "past stays only (ours)": past,
    "past + future stays (leaky)": past + future,
    "future stays only (leaky)": future,
}

rows = []
params = default_params("lightgbm")


def run(check, label, X_used, folds):
    metrics, _ = evaluate_cv(build_pipeline("lightgbm", params, cfg), X_used, y, folds)
    log_run(f"leakage:{label}", {"check": check, "variant": label}, metrics,
            tags={"stage": "leakage_check", "block": check})
    rows.append({"check": check, "variant": label, "cv_ap": metrics["cv_ap_mean"],
                 "cv_ap_sd": metrics["cv_ap_std"], "cv_auc": metrics["cv_auc_mean"]})
    print(f"{check:<9} {label:<42} AP={metrics['cv_ap_mean']:.4f}±{metrics['cv_ap_std']:.4f} "
          f"AUC={metrics['cv_auc_mean']:.4f}", flush=True)


for label, values in variants.items():
    run("temporal", label, X.assign(n_prior_encounters=values), grouped)
run("group", "grouped by patient (ours)", X, grouped)
run("group", "patients mixed across folds (leaky)", X, mixed)

res = pd.DataFrame(rows)
res.to_csv(cfg["paths"]["tables"] / "leakage_checks.csv", index=False, float_format="%.4f")
shared = sum(len(set(groups.iloc[tr]) & set(groups.iloc[va])) for tr, va in mixed)
print(f"patients appearing in both train and validation (mixed folds, summed): {shared}")

fig, ax = plt.subplots(figsize=(8, 3.6))
lab = res["check"] + ": " + res["variant"]
colors = ["#2c7fb8" if "ours" in v else "#e67e22" for v in res["variant"]]
ax.barh(lab[::-1], res["cv_ap"][::-1], xerr=res["cv_ap_sd"][::-1], color=colors[::-1])
ax.axvline(y.mean(), color="#999", ls="--", label=f"random (prevalence {y.mean():.3f})")
ax.set_xlabel("5-fold CV average precision (LightGBM, dev split)")
ax.set_title("What the leaks would have done (orange = leaky, blue = ours)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(cfg["paths"]["figures"] / "leakage_checks.png", dpi=130)

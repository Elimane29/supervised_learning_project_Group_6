"""Fairness audit on race + one mitigation, compared before/after on the untouched test set.

Definition chosen: EQUAL OPPORTUNITY (Hardt et al., 2016) - equal true-positive rate
across groups. Being flagged gives the patient follow-up care and a planned bed; a lower
TPR for one group means its patients who WILL come back are more often left without it.
Demographic parity is reported but not targeted: base readmission rates genuinely differ
between groups, so forcing equal flag rates would ignore real clinical need.

Mitigation: post-processing with group-specific thresholds (the deterministic form of
fairlearn's ThresholdOptimizer with a TPR-parity constraint). Thresholds are fitted on the
dev out-of-fold predictions, so the test set is only used for the before/after report.

Run: python scripts/06_fairness.py
"""
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, false_positive_rate,
                               selection_rate, true_positive_rate)
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score

from readmission.config import load_config

warnings.filterwarnings("ignore")
cfg = load_config()
FIG, TAB, OUT = cfg["paths"]["figures"], cfg["paths"]["tables"], cfg["paths"]["models"]
preds = joblib.load(OUT / "predictions.joblib")
t = preds["threshold"]
c_fn, c_fp = cfg["business"]["cost_fn"], cfg["business"]["cost_fp"]

RACE = {"Caucasian": "Caucasian", "AfricanAmerican": "AfricanAmerican",
        "Hispanic": "Hispanic", "Asian": "Asian/Other", "Other": "Asian/Other",
        "?": "Unknown"}


def prepare(df):
    df = df.copy()
    df["group"] = df["race"].astype(str).map(RACE).fillna("Unknown")
    return df


dev, test = prepare(preds["dev"]), prepare(preds["test"])


def audit(df, decision, label):
    rate_metrics = {"n": lambda y, p: len(y), "base_rate": lambda y, p: np.mean(y),
                    "selection_rate": selection_rate, "TPR": true_positive_rate,
                    "FPR": false_positive_rate,
                    "precision": lambda y, p: precision_score(y, p, zero_division=0)}
    mf = MetricFrame(metrics=rate_metrics, y_true=df["y"], y_pred=decision,
                     sensitive_features=df["group"])
    score = MetricFrame(metrics={"AP": average_precision_score, "brier": brier_score_loss},
                        y_true=df["y"], y_pred=df["p"], sensitive_features=df["group"])
    table = pd.concat([mf.by_group, score.by_group], axis=1)
    table.loc["ALL"] = list(mf.overall.values) + list(score.overall.values)
    fn = int(((decision == 0) & (df["y"] == 1)).sum())
    fp = int(((decision == 1) & (df["y"] == 0)).sum())
    gaps = {
        "setting": label,
        "equal_opportunity_diff (TPR max-min)": mf.difference()["TPR"],
        "equalized_odds_diff": equalized_odds_difference(df["y"], decision,
                                                         sensitive_features=df["group"]),
        "demographic_parity_diff": demographic_parity_difference(
            df["y"], decision, sensitive_features=df["group"]),
        "overall_recall": mf.overall["TPR"], "overall_precision": mf.overall["precision"],
        "alert_rate": float(np.mean(decision)),
        "cost_per_patient": (c_fn * fn + c_fp * fp) / len(df),
    }
    return table, gaps


# --- Before: one global threshold for everybody -------------------------------------------
before_tab, before_gap = audit(test, (test["p"] >= t).astype(int), "before (global t)")

# --- Mitigation fitted on DEV out-of-fold predictions ------------------------------------
dev_pred = dev["p"] >= t
target_tpr = dev_pred[dev["y"] == 1].mean()      # keep the overall operating point
group_t = {}
for g, d in dev.groupby("group"):
    pos = d.loc[d["y"] == 1, "p"]
    # smallest threshold whose TPR on this group's dev positives reaches the target
    group_t[g] = float(np.quantile(pos, 1 - target_tpr)) if len(pos) >= 30 else t
pd.Series(group_t, name="threshold").to_csv(TAB / "fairness_group_thresholds.csv",
                                            float_format="%.4f")

after_dec = (test["p"] >= test["group"].map(group_t).fillna(t)).astype(int)
after_tab, after_gap = audit(test, after_dec, "after (group thresholds, equal opportunity)")

before_tab.to_csv(TAB / "fairness_race_before.csv", float_format="%.4f")
after_tab.to_csv(TAB / "fairness_race_after.csv", float_format="%.4f")
gaps = pd.DataFrame([before_gap, after_gap]).set_index("setting")
gaps.to_csv(TAB / "fairness_summary.csv", float_format="%.4f")

# Secondary attributes (audit only): gender and age bracket at the global threshold.
sec = []
for attr in ["gender", "age"]:
    mf = MetricFrame(metrics={"n": lambda y, p: len(y), "base_rate": lambda y, p: np.mean(y),
                              "TPR": true_positive_rate, "selection_rate": selection_rate},
                     y_true=test["y"], y_pred=(test["p"] >= t).astype(int),
                     sensitive_features=test[attr].astype(str))
    sec.append(mf.by_group.assign(attribute=attr))
pd.concat(sec).to_csv(TAB / "fairness_secondary_attributes.csv", float_format="%.4f")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
order = [g for g in before_tab.index if g != "ALL"]
x = np.arange(len(order))
for a, metric in zip(ax, ["TPR", "selection_rate"]):
    a.bar(x - 0.2, before_tab.loc[order, metric], 0.4, label="before", color="#e67e22")
    a.bar(x + 0.2, after_tab.loc[order, metric], 0.4, label="after", color="#2c7fb8")
    a.set_xticks(x, [f"{g}\n(n={int(before_tab.loc[g, 'n'])})" for g in order], fontsize=8)
    a.set_title(f"{metric} by race - test set")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fairness_race.png", dpi=130)
plt.close(fig)

print(before_tab.round(3).to_string())
print(after_tab.round(3).to_string())
print(pd.Series(group_t).round(4).to_string())
print(gaps.round(4).T.to_string())

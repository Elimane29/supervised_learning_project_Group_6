"""Select, calibrate, threshold and freeze the final model; evaluate it ONCE on the test set.

Every decision below (which model, which calibrator, which threshold) is made on
out-of-fold predictions of the development split. Only then is the frozen procedure
refitted on all development data and applied to the untouched test split.

Run: python scripts/04_finalize.py [--model lightgbm]
"""
import argparse
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve,
                             roc_curve)

from readmission.calibration import ProbabilityCalibrator, cross_fit_calibration
from readmission.config import load_config
from readmission.data import cv_folds, load_splits
from readmission.evaluation import all_metrics, evaluate_cv, setup_mlflow
from readmission.pipelines import MODEL_NAMES, build_pipeline

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="force the final model instead of the best CV AP")
args = parser.parse_args()

cfg = load_config()
setup_mlflow(cfg)
FIG, TAB, OUT = cfg["paths"]["figures"], cfg["paths"]["tables"], cfg["paths"]["models"]
X_dev, X_test, y_dev, y_test, g_dev, g_test = load_splits(cfg)
folds = cv_folds(cfg, X_dev, y_dev, g_dev)
cap, c_fn, c_fp = (cfg["business"][k] for k in ("review_capacity", "cost_fn", "cost_fp"))

# --- 1. Tuned models compared on identical folds (same metrics for all) ------------------
tuned, oof = {}, {}
for name in MODEL_NAMES:
    path = OUT / f"best_params_{name}.json"
    if not path.exists():
        continue
    best = json.loads(path.read_text())
    cache = OUT / f"oof_{name}.npy"
    metrics, p = evaluate_cv(build_pipeline(name, best["params"], cfg), X_dev, y_dev, folds,
                             capacity=cap)
    np.save(cache, p)
    tuned[name], oof[name] = {"params": best["params"], **metrics}, p
    print(f"tuned {name:<14} AP={metrics['cv_ap_mean']:.4f}±{metrics['cv_ap_std']:.4f} "
          f"AUC={metrics['cv_auc_mean']:.4f} Brier={metrics['brier']:.4f}", flush=True)

comparison = pd.DataFrame({k: {m: v for m, v in d.items() if m != "params"}
                           for k, d in tuned.items()}).T
comparison.to_csv(TAB / "tuned_models_cv.csv", float_format="%.4f")
final_name = args.model or comparison["cv_ap_mean"].idxmax()
final_params = tuned[final_name]["params"]
p_oof = oof[final_name]
print(f"final model: {final_name}")

# --- 2. Calibration method chosen on cross-fitted OOF predictions -------------------------
yv = y_dev.to_numpy()
calib_rows, calibrated_oof = [], {}
for method in ["none", "sigmoid", "isotonic"]:
    pc = p_oof if method == "none" else cross_fit_calibration(p_oof, yv, folds, method)
    calibrated_oof[method] = pc
    m = all_metrics(yv, pc, threshold=0.5, capacity=cap)
    calib_rows.append({"method": method, **{k: m[k] for k in
                       ("brier", "ece", "log_loss", "average_precision", "roc_auc")}})
calib = pd.DataFrame(calib_rows).set_index("method")
calib.to_csv(TAB / "calibration_methods_oof.csv", float_format="%.5f")
# Lowest Brier wins; sigmoid preferred over isotonic unless isotonic is clearly better
# (simpler, smoother, less prone to overfit - Session 7).
best_method = calib["brier"].idxmin()
if best_method == "isotonic" and calib.loc["sigmoid", "brier"] - calib.loc["isotonic", "brier"] < 5e-5:
    best_method = "sigmoid"
calibrator = ProbabilityCalibrator(best_method).fit(p_oof, yv)
p_oof_cal = calibrated_oof[best_method]
print(calib.round(4).to_string())
print(f"calibration: {best_method}")

# --- 3. Cost-based threshold on calibrated OOF probabilities (Session 6/7) ----------------
grid = np.round(np.arange(0.02, 0.61, 0.005), 3)
cost_rows = []
for t in grid:
    pred = p_oof_cal >= t
    fn, fp = int(((~pred) & (yv == 1)).sum()), int((pred & (yv == 0)).sum())
    cost_rows.append({"threshold": t, "FN": fn, "FP": fp, "alert_rate": pred.mean(),
                      "cost_per_patient": (c_fn * fn + c_fp * fp) / len(yv),
                      **{k: v for k, v in all_metrics(yv, p_oof_cal, t, cap).items()
                         if k in ("precision", "recall", "f1")}})
costs = pd.DataFrame(cost_rows)
costs.to_csv(TAB / "threshold_costs_oof.csv", index=False, float_format="%.4f")
threshold = float(costs.loc[costs["cost_per_patient"].idxmin(), "threshold"])
bayes_t = c_fp / (c_fp + c_fn)  # optimal for perfectly calibrated probabilities
print(f"threshold: empirical {threshold:.3f} | Bayes {bayes_t:.3f} "
      f"(C_FN={c_fn}, C_FP={c_fp})")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(costs["threshold"], costs["cost_per_patient"], color="#2c7fb8")
ax.axvline(threshold, color="#c0392b", ls="--", label=f"chosen t={threshold:.3f}")
ax.axvline(bayes_t, color="#555", ls=":", label=f"C_FP/(C_FP+C_FN)={bayes_t:.3f}")
ax.axhline(c_fn * yv.mean(), color="#999", lw=0.8, label="flag nobody")
ax.set_xlabel("decision threshold on calibrated probability")
ax.set_ylabel("expected cost per discharge")
ax.set_title("Threshold selection on dev out-of-fold predictions")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "threshold_cost_curve.png", dpi=130)
plt.close(fig)

# --- 4. Refit on all dev data, evaluate ONCE on the untouched test set --------------------
pipeline = build_pipeline(final_name, final_params, cfg).fit(X_dev, y_dev)
p_test_raw = pipeline.predict_proba(X_test)[:, 1]
p_test = calibrator.transform(p_test_raw)
yt = y_test.to_numpy()
test_raw = all_metrics(yt, p_test_raw, threshold=0.5, capacity=cap)
test_cal = all_metrics(yt, p_test, threshold=threshold, capacity=cap)
pred = p_test >= threshold
fn, fp = int(((~pred) & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
test_cal.update(cost_per_patient=(c_fn * fn + c_fp * fp) / len(yt),
                cost_flag_nobody=c_fn * yt.mean(),
                expected_readmissions=float(p_test.sum()), observed_readmissions=int(yt.sum()))
summary = pd.DataFrame({"cv_dev_oof (calibrated)": all_metrics(yv, p_oof_cal, threshold, cap),
                        "test_raw (t=0.5)": test_raw, "test_final": test_cal})
summary.to_csv(TAB / "final_test_metrics.csv", float_format="%.4f")
print(summary.round(4).to_string())

# Figures: reliability (raw vs calibrated) + histogram, PR/ROC, confusion matrix.
fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.3, 1]})
curves = ([(p_test_raw, f"{final_name} (no recalibration needed)", "#2c7fb8")]
          if best_method == "none" else
          [(p_test_raw, "raw model", "#e67e22"), (p_test, f"{best_method} calibrated", "#2c7fb8")])
for p, lab, c in curves:
    frac, mean = calibration_curve(yt, p, n_bins=10, strategy="quantile")
    ax[0].plot(mean, frac, "o-", color=c, label=lab, ms=4)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
lim = max(np.quantile(p_test_raw, 0.995), np.quantile(p_test, 0.995)) * 1.05
ax[0].set_xlim(0, lim)
ax[0].set_ylim(0, lim)
ax[0].set_xlabel("mean predicted probability (decile bins)")
ax[0].set_ylabel("observed readmission rate")
ax[0].set_title("Reliability diagram - test set")
ax[0].legend(fontsize=8)
ax[1].hist([p_test_raw, p_test], bins=40, label=["raw", "calibrated"],
           color=["#e67e22", "#2c7fb8"], alpha=0.8)
ax[1].set_xlabel("predicted probability")
ax[1].set_ylabel("patients")
ax[1].set_title("Where predictions sit (bin support)")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "calibration_test.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
prec, rec, _ = precision_recall_curve(yt, p_test)
ax[0].plot(rec, prec, color="#2c7fb8", label=f"{final_name} (AP={test_cal['average_precision']:.3f})")
ax[0].axhline(yt.mean(), color="#999", ls="--", label=f"random (prevalence {yt.mean():.3f})")
ax[0].set_xlabel("recall")
ax[0].set_ylabel("precision")
ax[0].set_title("Precision-Recall - test")
ax[0].legend(fontsize=8)
fpr, tpr, _ = roc_curve(yt, p_test)
ax[1].plot(fpr, tpr, color="#2c7fb8", label=f"ROC-AUC={test_cal['roc_auc']:.3f}")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[1].set_xlabel("false positive rate")
ax[1].set_ylabel("true positive rate (recall)")
ax[1].set_title("ROC - test")
ax[1].legend(fontsize=8)
ConfusionMatrixDisplay(confusion_matrix(yt, pred), display_labels=["no <30d", "<30d"]).plot(
    ax=ax[2], colorbar=False, cmap="Blues")
ax[2].set_title(f"Confusion matrix at t={threshold:.3f}")
fig.tight_layout()
fig.savefig(FIG / "final_test_curves.png", dpi=130)
plt.close(fig)

# --- 5. Freeze everything the API and the audits need --------------------------------------
bundle = {"pipeline": pipeline, "calibrator": calibrator, "threshold": threshold,
          "model_name": final_name, "params": final_params, "calibration": best_method,
          "input_columns": list(X_dev.columns)}
joblib.dump(bundle, OUT / "final_model.joblib")
keep = ["race", "gender", "age"]
joblib.dump({"dev": X_dev[keep].assign(y=yv, p_raw=p_oof, p=p_oof_cal),
             "test": X_test[keep].assign(y=yt, p_raw=p_test_raw, p=p_test),
             "threshold": threshold, "folds": folds}, OUT / "predictions.joblib")

with mlflow.start_run(run_name=f"FINAL_{final_name}"):
    mlflow.set_tags({"stage": "final_test", "model": final_name, "calibration": best_method,
                     "test_used_once": "true"})
    mlflow.log_params({**final_params, "threshold": threshold, "cost_fn": c_fn,
                       "cost_fp": c_fp, "calibration": best_method})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_cal.items()})
    mlflow.log_metrics({f"test_raw_{k}": v for k, v in test_raw.items()})
    mlflow.log_metrics({f"cv_{k}": v for k, v in tuned[final_name].items()
                        if k != "params"})
    for f in ["calibration_test.png", "final_test_curves.png", "threshold_cost_curve.png"]:
        mlflow.log_artifact(str(FIG / f))
    for f in ["tuned_models_cv.csv", "calibration_methods_oof.csv", "final_test_metrics.csv"]:
        mlflow.log_artifact(str(TAB / f))
    mlflow.log_artifact(str(OUT / "final_model.joblib"))
print("saved models/final_model.joblib")

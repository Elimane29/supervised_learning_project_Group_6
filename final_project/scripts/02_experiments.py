"""Untuned comparisons on identical patient-grouped folds, every run logged to MLflow.

  A. model progression      dummy -> logistic -> random forest -> LightGBM -> CatBoost
  B. imbalance remedies     none / class weights / under- / over-sampling (training folds only)
  C. feature ablations      what the engineered features and the target encoding buy

Run: python scripts/02_experiments.py
"""
import warnings

import numpy as np

from readmission.config import load_config
from readmission.data import cv_folds, load_splits
from readmission.evaluation import (all_metrics, evaluate_cv, log_run, results_table,
                                    setup_mlflow)
from readmission.pipelines import build_pipeline, default_params

warnings.filterwarnings("ignore", category=UserWarning)
cfg = load_config()
setup_mlflow(cfg)
X, _, y, _, groups, _ = load_splits(cfg)          # test split is NOT used here
folds = cv_folds(cfg, X, y, groups)
cap = cfg["business"]["review_capacity"]
ratio = float((y == 0).sum() / (y == 1).sum())     # ~7.8 negatives per positive
rows = []


def run(block, label, model, params=None, **pipe_kwargs):
    params = {**default_params(model), **(params or {})}
    pipe = build_pipeline(model, params, cfg, **pipe_kwargs)
    metrics, _ = evaluate_cv(pipe, X, y, folds, capacity=cap)
    log_run(f"{block}:{label}", {"model": model, **params, **pipe_kwargs}, metrics,
            tags={"block": block, "stage": "untuned_comparison", "cv": "StratifiedGroupKFold-5"})
    rows.append({"block": block, "config": label, **metrics})
    print(f"{block:<10} {label:<34} AP={metrics['cv_ap_mean']:.4f}±{metrics['cv_ap_std']:.4f} "
          f"AUC={metrics['cv_auc_mean']:.4f} Brier={metrics['brier']:.4f} "
          f"t={metrics['fit_seconds']:.0f}s", flush=True)


# A. Dummy baseline: predicting the prevalence for everyone gives AP = prevalence.
prior = np.full(len(y), y.mean())
dummy = all_metrics(y, prior, threshold=0.5, capacity=cap)
dummy.update(cv_ap_mean=y.mean(), cv_ap_std=0.0, cv_auc_mean=0.5, cv_auc_std=0.0)
log_run("progression:dummy_prior", {"model": "dummy"}, dummy, tags={"block": "progression"})
rows.append({"block": "progression", "config": "dummy (predict prevalence)", **dummy})

for model in ["logreg", "random_forest", "lightgbm", "catboost"]:
    run("progression", model, model)

# B. Imbalance remedies (LightGBM = fastest strong learner; logistic as a linear check).
run("imbalance", "lgbm none", "lightgbm")
run("imbalance", "lgbm scale_pos_weight=sqrt(ratio)", "lightgbm",
    {"scale_pos_weight": float(np.sqrt(ratio))})
run("imbalance", f"lgbm scale_pos_weight=ratio({ratio:.1f})", "lightgbm",
    {"scale_pos_weight": ratio})
run("imbalance", "lgbm undersample 1:2", "lightgbm", sampler="undersample")
run("imbalance", "lgbm oversample 1:2", "lightgbm", sampler="oversample")
run("imbalance", "logreg balanced", "logreg", {"class_weight": "balanced"})

# C. Feature ablations (LightGBM).
run("features", "full (prior enc. + target enc.)", "lightgbm")
run("features", "- prior encounters", "lightgbm", use_prior_encounters=False)
run("features", "- target encoding of ICD-9", "lightgbm", use_target_encoding=False)
run("features", "- both", "lightgbm", use_prior_encounters=False, use_target_encoding=False)

results_table(rows, cfg["paths"]["tables"] / "experiments_untuned.csv")
print("saved reports/tables/experiments_untuned.csv")

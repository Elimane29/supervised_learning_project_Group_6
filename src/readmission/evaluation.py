"""Metrics, cross-validated evaluation and MLflow logging shared by all scripts."""
from __future__ import annotations

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

from .config import PROJECT_ROOT


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Weighted mean |observed rate - mean prediction| over quantile bins (Session 7).

    Quantile bins are used because most predictions sit below 0.2: uniform bins would
    leave the upper bins almost empty and their gaps would be pure noise.
    """
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def threshold_metrics(y, p, threshold: float) -> dict:
    pred = (np.asarray(p) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "alert_rate": float(pred.mean()),
    }


def probability_metrics(y, p) -> dict:
    """Threshold-free metrics: ranking (ROC-AUC, AP) and probability quality (Brier, ECE)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def top_k_metrics(y, p, share: float) -> dict:
    """Precision/recall when the team can only follow up the top `share` of discharges."""
    y, p = np.asarray(y), np.asarray(p)
    k = int(round(share * len(p)))
    top = np.argsort(-p)[:k]
    hits = y[top].sum()
    return {f"precision_at_{int(share * 100)}pct": hits / max(k, 1),
            f"recall_at_{int(share * 100)}pct": hits / max(y.sum(), 1)}


def all_metrics(y, p, threshold: float = 0.5, capacity: float = 0.2) -> dict:
    out = probability_metrics(y, p)
    out.update(threshold_metrics(y, p, threshold))
    out.update(top_k_metrics(y, p, capacity))
    return {k: float(v) for k, v in out.items()}


def cross_validate_oof(pipeline, X, y, folds, trial=None, fit_params=None):
    """Fit a fresh clone per fold; return out-of-fold probabilities and per-fold AP.

    With an Optuna trial, the running mean AP is reported after each fold so that
    clearly bad configurations are pruned early (saves most of the tuning budget).
    """
    import optuna

    oof = np.full(len(y), np.nan)
    fold_ap, fold_auc = [], []
    for k, (tr, va) in enumerate(folds):
        model = clone(pipeline)
        model.fit(X.iloc[tr], y.iloc[tr], **(fit_params or {}))
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_ap.append(average_precision_score(y.iloc[va], oof[va]))
        fold_auc.append(roc_auc_score(y.iloc[va], oof[va]))
        if trial is not None:
            trial.report(float(np.mean(fold_ap)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return oof, np.array(fold_ap), np.array(fold_auc)


def evaluate_cv(pipeline, X, y, folds, capacity=0.2) -> tuple[dict, np.ndarray]:
    start = time.time()
    oof, fold_ap, fold_auc = cross_validate_oof(pipeline, X, y, folds)
    metrics = all_metrics(y, oof, threshold=0.5, capacity=capacity)
    metrics.update({"cv_ap_mean": fold_ap.mean(), "cv_ap_std": fold_ap.std(),
                    "cv_auc_mean": fold_auc.mean(), "cv_auc_std": fold_auc.std(),
                    "fit_seconds": time.time() - start})
    return {k: float(v) for k, v in metrics.items()}, oof


def setup_mlflow(cfg: dict) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + (PROJECT_ROOT / uri.removeprefix("sqlite:///")).as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


def log_run(run_name: str, params: dict, metrics: dict, tags: dict | None = None,
            artifacts: list | None = None, nested: bool = False) -> str:
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(tags or {})
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def results_table(rows: list[dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.4f")
    return df

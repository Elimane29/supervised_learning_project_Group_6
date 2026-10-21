"""Optuna tuning of each model family with patient-grouped 5-fold CV, logged to MLflow.

* Objective = mean out-of-fold Average Precision (PR-AUC), the metric for a rare positive.
* TPE sampler uses past trials to propose the next one (Session 5); a MedianPruner stops a
  trial after a fold if it is already worse than the median trial at that fold.
* One MLflow parent run per model, one nested run per trial (params + fold metrics).
* The test split is never loaded here; reported scores are CV scores on the dev split.
  Selecting the best of N trials makes the best CV score slightly optimistic, which is why
  the final model is judged once on the untouched test set (scripts/04_finalize.py).

Run: python scripts/03_tune.py lightgbm [logreg random_forest catboost]
"""
import json
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna

from readmission.config import load_config
from readmission.data import cv_folds, load_splits
from readmission.evaluation import cross_validate_oof, setup_mlflow
from readmission.pipelines import MODEL_NAMES, build_pipeline, suggest_params

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
cfg = load_config()
setup_mlflow(cfg)
X, _, y, _, groups, _ = load_splits(cfg)
folds = cv_folds(cfg, X, y, groups)
OUT = cfg["paths"]["models"]


def tune(model: str) -> None:
    n_trials = cfg["optuna"]["n_trials"][model]

    def objective(trial):
        params = suggest_params(model, trial)
        with mlflow.start_run(run_name=f"{model}_trial_{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            try:
                _, ap, auc = cross_validate_oof(build_pipeline(model, params, cfg), X, y,
                                                folds, trial=trial)
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                raise
            for k, (a, b) in enumerate(zip(ap, auc)):
                mlflow.log_metric("fold_ap", a, step=k)
                mlflow.log_metric("fold_auc", b, step=k)
            mlflow.log_metrics({"cv_ap_mean": ap.mean(), "cv_ap_std": ap.std(),
                                "cv_auc_mean": auc.mean()})
            mlflow.set_tag("state", "complete")
            trial.set_user_attr("cv_ap_std", float(ap.std()))
            trial.set_user_attr("cv_auc_mean", float(auc.mean()))
        print(f"  {model} trial {trial.number:>3}: AP={ap.mean():.4f}±{ap.std():.4f}", flush=True)
        return float(ap.mean())

    study = optuna.create_study(
        study_name=model, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    with mlflow.start_run(run_name=f"optuna_{model}") as parent:
        mlflow.set_tags({"stage": "tuning", "model": model, "objective": "cv_average_precision",
                         "cv": "StratifiedGroupKFold-5", "sampler": "TPE",
                         "pruner": "MedianPruner"})
        study.optimize(objective, n_trials=n_trials,
                       timeout=cfg["optuna"]["timeout_per_model_s"])

        trials = study.trials_dataframe()
        trials.to_csv(cfg["paths"]["tables"] / f"optuna_trials_{model}.csv", index=False)
        best = {"model": model, "params": study.best_params, "cv_ap_mean": study.best_value,
                "cv_ap_std": study.best_trial.user_attrs.get("cv_ap_std"),
                "cv_auc_mean": study.best_trial.user_attrs.get("cv_auc_mean"),
                "n_trials": len(trials),
                "n_pruned": int((trials["state"] == "PRUNED").sum())}
        best_path = OUT / f"best_params_{model}.json"
        best_path.write_text(json.dumps(best, indent=2))

        done = trials[trials["state"] == "COMPLETE"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(done["number"], done["value"], s=14, color="#2c7fb8", label="trial")
        ax.plot(done["number"], np.maximum.accumulate(done["value"]), color="#c0392b",
                label="best so far")
        ax.set_xlabel("trial")
        ax.set_ylabel("CV average precision")
        ax.set_title(f"Optuna history - {model}")
        ax.legend()
        fig.tight_layout()
        fig_path = cfg["paths"]["figures"] / f"optuna_history_{model}.png"
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"best_cv_ap_mean": study.best_value,
                            "n_pruned": best["n_pruned"]})
        for path in (best_path, fig_path):
            mlflow.log_artifact(str(path))
        print(f"{model}: best AP={study.best_value:.4f} after {len(trials)} trials "
              f"({best['n_pruned']} pruned) -> {best_path.name}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or MODEL_NAMES:
        tune(name)

"""One table with every metric for every model, labelled by the phase it comes from.

Phases:
  CV-validation  out-of-fold predictions on the DEVELOPMENT set: each row is scored by the
                 model trained on the other 4 folds (the CV-train part), never by itself.
  Test           the final model, trained on the whole development set, scored once on
                 the held-out TEST set.

Threshold metrics (accuracy, precision, recall, F1) are shown at the default 0.5 and at the
deployed cost-based threshold, so the effect of the threshold on each metric is visible.
No model is retrained: the script reads the saved out-of-fold and test predictions.

Run: python scripts/10_metrics_table.py
"""
import joblib
import numpy as np
import pandas as pd

from readmission.config import load_config
from readmission.data import load_splits
from readmission.evaluation import probability_metrics, threshold_metrics
from readmission.pipelines import MODEL_NAMES

cfg = load_config()
OUT, TAB = cfg["paths"]["models"], cfg["paths"]["tables"]
_, _, y_dev, _, _, _ = load_splits(cfg)
y_dev = y_dev.to_numpy()
preds = joblib.load(OUT / "predictions.joblib")
t_star = preds["threshold"]

rows = []


def add(phase, model, y, p):
    row = {"phase": phase, "model": model}
    row.update({k: v for k, v in probability_metrics(y, p).items()
                if k in ("roc_auc", "average_precision", "brier")})
    for t, tag in [(0.5, "t=0.5"), (t_star, f"t={t_star:.3f}")]:
        row.update({f"{k} @ {tag}": v for k, v in threshold_metrics(y, p, t).items()})
    rows.append(row)


add("CV-validation (development, out-of-fold)", "dummy (prevalence)", y_dev,
    np.full(len(y_dev), y_dev.mean()))
for name in MODEL_NAMES:
    path = OUT / f"oof_{name}.npy"
    if path.exists():
        add("CV-validation (development, out-of-fold)", name, y_dev, np.load(path))
add("Test (held out, once)", f"final {preds.get('model', 'catboost')}",
    preds["test"]["y"].to_numpy(), preds["test"]["p"].to_numpy())

table = pd.DataFrame(rows)
table.to_csv(TAB / "metrics_all_models.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(table.round(3).to_string(index=False))

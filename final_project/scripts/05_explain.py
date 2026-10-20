"""SHAP explanations of the frozen final model: global importance + 4 local explanations.

SHAP values are computed on the model's raw output (log-odds), before calibration: the
calibrator is a monotone map, so it changes probability levels but not which features
push a patient up or down. One-hot columns are summed back to their original feature.

Run: python scripts/05_explain.py
"""
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from readmission.config import load_config
from readmission.data import load_splits
from readmission.features import feature_names

warnings.filterwarnings("ignore")
cfg = load_config()
FIG, TAB, OUT = cfg["paths"]["figures"], cfg["paths"]["tables"], cfg["paths"]["models"]
bundle = joblib.load(OUT / "final_model.joblib")
pipe, name = bundle["pipeline"], bundle["model_name"]
_, X_test, _, y_test, _, _ = load_splits(cfg)

rng = np.random.default_rng(cfg["seed"])
sample = np.sort(rng.choice(len(X_test), size=3000, replace=False))
Xs = X_test.iloc[sample]
feats = pipe.named_steps["features"].transform(Xs)
model = pipe.named_steps["model"]
model = getattr(model, "model_", model)  # unwrap CatBoostModel

if "preprocess" in pipe.named_steps:
    pre = pipe.named_steps["preprocess"]
    Z = pre.transform(feats)
    cols = list(pre.get_feature_names_out())
    Z = pd.DataFrame(np.asarray(Z, dtype=float), columns=cols, index=Xs.index)
else:  # CatBoost consumes the engineered frame directly
    Z, cols = feats, list(feats.columns)

if name == "logreg":
    explainer = shap.LinearExplainer(model, Z)
else:
    explainer = shap.TreeExplainer(model)
sv = explainer(Z)
values = sv.values[..., 1] if sv.values.ndim == 3 else sv.values
base_value = np.ravel(sv.base_values)[0] if np.ndim(sv.base_values) < 2 else sv.base_values[0, 1]

# Map every model column back to the raw feature it came from (one-hot -> parent).
numeric, low, high = feature_names(cfg["features"]["use_prior_encounters"])
originals = sorted(numeric + low + high, key=len, reverse=True)


def parent(col):
    return next((o for o in originals if col == o or col.startswith(o + "_")), col)


groups = pd.Series([parent(c) for c in cols], index=cols)
abs_by_feature = (pd.DataFrame(np.abs(values), columns=cols).T.groupby(groups.values).sum().T
                  .mean().sort_values(ascending=False))
abs_by_feature.rename("mean_abs_shap").to_csv(TAB / "shap_global_importance.csv",
                                              float_format="%.5f")

fig, ax = plt.subplots(figsize=(7, 6))
top = abs_by_feature.head(20)[::-1]
ax.barh(top.index, top.values, color="#2c7fb8")
ax.set_xlabel("mean |SHAP| (log-odds), one-hot columns summed")
ax.set_title(f"Global importance - {name} (3,000 test patients)")
fig.tight_layout()
fig.savefig(FIG / "shap_global_bar.png", dpi=130)
plt.close(fig)

# Beeswarm on the 15 most important model columns (direction of each effect).
expl = shap.Explanation(values=values, base_values=np.full(len(Z), base_value),
                        data=np.asarray(Z if name != "catboost" else
                                        feats.apply(lambda s: pd.factorize(s)[0]
                                                    if s.dtype == object else s)),
                        feature_names=cols)
plt.figure()
shap.plots.beeswarm(expl, max_display=15, show=False)
plt.title(f"SHAP summary - {name}")
plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
plt.close("all")

# Local explanations: four contrasting test patients.
p_cal = bundle["calibrator"].transform(pipe.predict_proba(Xs)[:, 1])
y_s = y_test.iloc[sample].to_numpy()
t = bundle["threshold"]
cases = {
    "true_positive_high_risk": np.argmax(np.where(y_s == 1, p_cal, -1)),
    "false_negative_missed": np.argmin(np.where(y_s == 1, p_cal, 2)),
    "false_positive_alert": np.argmax(np.where(y_s == 0, p_cal, -1)),
    "true_negative_low_risk": np.argmin(np.where(y_s == 0, p_cal, 2)),
}
local_rows = []
for label, i in cases.items():
    plt.figure()
    shap.plots.waterfall(expl[i], max_display=10, show=False)
    plt.title(f"{label}: calibrated p={p_cal[i]:.3f}, actual y={y_s[i]}, t={t:.3f}",
              fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / f"shap_local_{label}.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    contrib = pd.Series(values[i], index=cols)
    top3 = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
    local_rows.append({"case": label, "p_calibrated": round(float(p_cal[i]), 4),
                       "y": int(y_s[i]),
                       "top_contributions": "; ".join(f"{k}={v:+.2f}" for k, v in top3.items())})
pd.DataFrame(local_rows).to_csv(TAB / "shap_local_cases.csv", index=False)
print(abs_by_feature.head(15).round(4).to_string())
print(pd.DataFrame(local_rows).to_string(index=False))

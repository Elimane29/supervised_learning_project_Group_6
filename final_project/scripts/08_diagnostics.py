"""Session 8 diagnostics of the FROZEN final model at the FROZEN threshold (an audit:
nothing below is used to change the model, so reading the test set here is allowed).

  1. Slice evaluation   recall/precision/AP per meaningful slice, with counts, prevalence
                        and a 95% Wilson interval on recall (is a gap real or noise?)
  2. Error analysis     what distinguishes missed readmissions (FN) from caught ones (TP),
                        in the raw data and in their SHAP contributions
  3. Feature importance three definitions side by side: built-in (CatBoost prediction-value
                        change, LightGBM split count and gain) vs permutation on held-out
                        data, incl. a joint shuffle of correlated history features
  4. PDP + ICE          the average learned relationship, and whether it hides
                        heterogeneous individual responses

None of these tools is causal: they describe the model, not why patients come back.

Run: python scripts/08_diagnostics.py
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

from readmission.config import load_config
from readmission.data import load_splits
from readmission.evaluation import setup_mlflow
from readmission.features import feature_names, group_icd9
from readmission.pipelines import build_pipeline

warnings.filterwarnings("ignore")
cfg = load_config()
FIG, TAB, OUT = cfg["paths"]["figures"], cfg["paths"]["tables"], cfg["paths"]["models"]
bundle = joblib.load(OUT / "final_model.joblib")
pipe, t = bundle["pipeline"], bundle["threshold"]
X_dev, X_test, y_dev, y_test, _, _ = load_splits(cfg)
y = y_test.to_numpy()
p = bundle["calibrator"].transform(pipe.predict_proba(X_test)[:, 1])
flag = p >= t
rng = np.random.default_rng(cfg["seed"])


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (stable for small n, unlike k/n ± 2se)."""
    if n == 0:
        return np.nan, np.nan
    phat = k / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# --- 1. Slices: interpretable, large enough, operationally relevant -------------------
age_mid = X_test["age"].astype(str).str.extract(r"\[(\d+)")[0].astype(int)
disp = X_test["discharge_disposition_id"].astype(int)
slices = {
    "age": pd.cut(age_mid, [-1, 49, 69, 79, 200], labels=["<50", "50-69", "70-79", "80+"]),
    "inpatient visits last year": pd.cut(X_test["number_inpatient"], [-1, 0, 1, 99],
                                         labels=["0", "1", "2+"]),
    "discharge destination": np.select([disp == 1, disp == 6, disp.isin([3, 22, 2, 5])],
                                       ["home", "home + health service",
                                        "care facility / other hospital"], "other"),
    "primary diagnosis": X_test["diag_1"].map(group_icd9).where(
        lambda s: s.isin(["Circulatory", "Diabetes", "Respiratory", "Digestive"]), "other"),
    "HbA1c tested": np.where(X_test["A1Cresult"].astype(str) == "None", "no", "yes"),
    "payer recorded": np.where(X_test["payer_code"].astype(str) == "?", "no", "yes"),
}
rows = []
for attr, groups in slices.items():
    groups = pd.Series(np.asarray(groups, dtype=str), index=X_test.index)
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        pos, tp, fl = int(y[m].sum()), int((flag & m & (y == 1)).sum()), int((flag & m).sum())
        lo, hi = wilson(tp, pos)
        prev = pos / m.sum()
        ap = average_precision_score(y[m], p[m]) if 0 < pos < m.sum() else np.nan
        rows.append({"attribute": attr, "slice": g, "n": int(m.sum()), "positives": pos,
                     "prevalence": prev, "recall": tp / max(pos, 1), "recall_ci_low": lo,
                     "recall_ci_high": hi, "precision": tp / fl if fl else np.nan,
                     "flag_rate": fl / m.sum(), "AP": ap, "AP_lift": ap / prev,
                     "reliable": pos >= 100})
sl = pd.DataFrame(rows)
overall_recall = (flag & (y == 1)).sum() / y.sum()
sl.to_csv(TAB / "diagnostics_slices.csv", index=False, float_format="%.4f")

fig, ax = plt.subplots(figsize=(8, 7))
lab = sl["attribute"] + ": " + sl["slice"] + " (pos=" + sl["positives"].astype(str) + ")"
ypos = np.arange(len(sl))[::-1]
ax.errorbar(sl["recall"], ypos, xerr=[sl["recall"] - sl["recall_ci_low"],
                                      sl["recall_ci_high"] - sl["recall"]],
            fmt="o", color="#2c7fb8", ecolor="#9ecae1", capsize=3)
ax.axvline(overall_recall, color="#c0392b", ls="--", label=f"overall {overall_recall:.2f}")
ax.set_yticks(ypos, lab, fontsize=8)
ax.set_xlabel(f"recall at frozen threshold t={t:.3f} (95% Wilson interval)")
ax.set_title("Slice diagnostics - test set, frozen model")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "diagnostics_slices.png", dpi=130)
plt.close(fig)

# --- 2. Error analysis: missed (FN) vs caught (TP) readmissions --------------------------
pos = y == 1
tp_m, fn_m = pos & flag, pos & ~flag
Xp = X_test.assign(p=p)
checks = {
    "no inpatient visit last year": Xp["number_inpatient"] == 0,
    "no earlier stay in records": Xp["n_prior_encounters"] == 0,
    "discharged home (id 1)": disp == 1,
    "HbA1c not tested": Xp["A1Cresult"].astype(str) == "None",
    "payer missing": Xp["payer_code"].astype(str) == "?",
    "predicted risk < 0.10": Xp["p"] < 0.10,
}
err = pd.DataFrame({
    "FN (missed)": {k: float(v[fn_m].mean()) for k, v in checks.items()},
    "TP (caught)": {k: float(v[tp_m].mean()) for k, v in checks.items()},
})
for col in ["time_in_hospital", "num_medications", "number_diagnoses", "age"]:
    s = age_mid if col == "age" else Xp[col]
    err.loc[f"mean {col}"] = [float(s[fn_m].mean()), float(s[tp_m].mean())]
err.loc["count"] = [int(fn_m.sum()), int(tp_m.sum())]
err.to_csv(TAB / "diagnostics_fn_vs_tp.csv", float_format="%.3f")

# SHAP of every actual readmission: which contributions keep FN scores low?
model = getattr(pipe.named_steps["model"], "model_", pipe.named_steps["model"])
feats = pipe.named_steps["features"].transform(X_test[pos])
if "preprocess" in pipe.named_steps:
    pre = pipe.named_steps["preprocess"]
    Z = pd.DataFrame(np.asarray(pre.transform(feats), float), columns=pre.get_feature_names_out())
else:
    Z = feats
sv = shap.TreeExplainer(model)(Z)
vals = sv.values[..., 1] if sv.values.ndim == 3 else sv.values
numeric, low, high = feature_names(cfg["features"]["use_prior_encounters"])
originals = sorted(numeric + low + high, key=len, reverse=True)
parent = [next((o for o in originals if c == o or c.startswith(o + "_")), c) for c in Z.columns]
shp = pd.DataFrame(vals, columns=Z.columns).T.groupby(parent).sum().T
is_fn = fn_m[pos]
shap_cmp = pd.DataFrame({"mean SHAP FN": shp[is_fn].mean(), "mean SHAP TP": shp[~is_fn].mean()})
shap_cmp["difference"] = shap_cmp["mean SHAP FN"] - shap_cmp["mean SHAP TP"]
shap_cmp = shap_cmp.reindex(shap_cmp["difference"].abs().sort_values(ascending=False).index)
shap_cmp.to_csv(TAB / "diagnostics_shap_fn_vs_tp.csv", float_format="%.4f")

top = shap_cmp.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
yy = np.arange(len(top))
ax.barh(yy - 0.2, top["mean SHAP TP"], 0.4, color="#2c7fb8", label=f"TP caught (n={tp_m.sum()})")
ax.barh(yy + 0.2, top["mean SHAP FN"], 0.4, color="#e67e22", label=f"FN missed (n={fn_m.sum()})")
ax.set_yticks(yy, top.index, fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("mean SHAP (log-odds) among actual readmissions")
ax.set_title("What pushes missed readmissions down?")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "diagnostics_shap_fn_vs_tp.png", dpi=130)
plt.close(fig)

# --- 3. Three definitions of feature importance ------------------------------------------
# (a) built-in, final CatBoost: average change of prediction when the feature's split is used
cb_imp = pd.Series(model.get_feature_importance(), index=model.feature_names_,
                   name="catboost_prediction_change")
# (b) built-in, tuned LightGBM refitted on dev: split count vs total gain
lgb_params = json.loads((OUT / "best_params_lightgbm.json").read_text())["params"]
lgb_pipe = build_pipeline("lightgbm", lgb_params, cfg).fit(X_dev, y_dev)
booster = lgb_pipe.named_steps["model"].booster_
lgb_cols = list(lgb_pipe.named_steps["preprocess"].get_feature_names_out())
lgb_parent = [next((o for o in originals if c == o or c.startswith(o + "_")), c)
              for c in lgb_cols]
lgb_split = pd.Series(booster.feature_importance("split"), index=lgb_cols).groupby(
    lgb_parent).sum().rename("lightgbm_split_count")
lgb_gain = pd.Series(booster.feature_importance("gain"), index=lgb_cols).groupby(
    lgb_parent).sum().rename("lightgbm_total_gain")

# (c) permutation on held-out rows, CatBoost, raw input columns, AP drop averaged over repeats
sub = np.sort(rng.choice(len(X_test), size=6000, replace=False))
Xs, ys = X_test.iloc[sub].reset_index(drop=True), y[sub]
score = lambda X: average_precision_score(ys, pipe.predict_proba(X)[:, 1])
base_ap = score(Xs)
skip = {"encounter_id", "patient_nbr", "weight", "examide", "citoglipton"}
perm_cols = [c for c in Xs.columns if c not in skip]  # race kept as a sanity check: not an input
groups_to_test = {c: [c] for c in perm_cols}
groups_to_test["[joint] prior utilisation"] = ["number_inpatient", "n_prior_encounters",
                                                "number_emergency", "number_outpatient"]
perm = {}
for name, cols in groups_to_test.items():
    drops = []
    for _ in range(3):
        Xp_ = Xs.copy()
        idx = rng.permutation(len(Xs))
        Xp_[cols] = Xs[cols].to_numpy()[idx]  # same permutation keeps the group consistent
        drops.append(base_ap - score(Xp_))
    perm[name] = (np.mean(drops), np.std(drops))
perm_df = pd.DataFrame(perm, index=["AP_drop_mean", "AP_drop_sd"]).T.sort_values(
    "AP_drop_mean", ascending=False)
perm_df.to_csv(TAB / "diagnostics_permutation_importance.csv", float_format="%.5f")

shap_global = pd.read_csv(TAB / "shap_global_importance.csv", index_col=0)["mean_abs_shap"]
imp = pd.concat([cb_imp, lgb_split, lgb_gain, shap_global.rename("mean_abs_shap"),
                 perm_df["AP_drop_mean"].rename("permutation_AP_drop")], axis=1)
ranks = imp.rank(ascending=False).add_suffix("_rank")
imp = pd.concat([imp, ranks], axis=1).sort_values("mean_abs_shap_rank")
imp.to_csv(TAB / "diagnostics_importance_comparison.csv", float_format="%.4f")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
pt = perm_df.head(12)[::-1]
ax[0].barh(pt.index, pt["AP_drop_mean"], xerr=pt["AP_drop_sd"], color="#2c7fb8")
ax[0].set_xlabel(f"AP drop when shuffled (baseline AP {base_ap:.3f}, 3 repeats)")
ax[0].set_title("Permutation importance - final CatBoost, test rows")
top_feats = ["number_inpatient", "n_prior_encounters", "discharge_disposition_id", "diag_1",
             "num_lab_procedures", "num_medications", "medical_specialty", "age_years"]
cmp = imp.loc[[f for f in top_feats if f in imp.index],
              ["lightgbm_split_count_rank", "lightgbm_total_gain_rank", "mean_abs_shap_rank"]]
cmp.plot.barh(ax=ax[1], color=["#bbbbbb", "#e67e22", "#2c7fb8"])
ax[1].invert_xaxis()
ax[1].set_xlabel("rank (1 = most important)")
ax[1].set_title("Same feature, different definitions")
ax[1].legend(["LightGBM split count", "LightGBM gain", "CatBoost mean |SHAP|"], fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "diagnostics_importance.png", dpi=130)
plt.close(fig)

# --- 4. PDP + ICE: replace one column for everyone, predict, average ---------------------
ice_rows = X_test.iloc[np.sort(rng.choice(len(X_test), size=300, replace=False))]
pdp_specs = {"number_inpatient": list(range(0, 9)),
             "time_in_hospital": list(range(1, 15)),
             "age": [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]}
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
pdp_out = {}
for a, (col, grid) in zip(ax, pdp_specs.items()):
    ice = np.column_stack([pipe.predict_proba(ice_rows.assign(**{col: v}))[:, 1] for v in grid])
    xs = np.arange(len(grid)) if col == "age" else np.array(grid)
    for line in ice[:80]:
        a.plot(xs, line, color="#bbbbbb", lw=0.6, alpha=0.6)
    a.plot(xs, ice.mean(0), color="#c0392b", lw=2.5, label="PDP (average)")
    a.set_xlabel(col if col != "age" else "age bracket (lower bound)")
    if col == "age":
        a.set_xticks(xs, [g[1:g.index("-")] for g in grid])
    a.set_ylabel("predicted readmission probability")
    a.set_title(f"PDP + ICE: {col}")
    a.legend(fontsize=8)
    pdp_out[col] = dict(zip(map(str, grid), np.round(ice.mean(0), 4)))
fig.tight_layout()
fig.savefig(FIG / "diagnostics_pdp_ice.png", dpi=130)
plt.close(fig)
pd.DataFrame(pdp_out).to_csv(TAB / "diagnostics_pdp.csv")
corr = X_test[["number_inpatient", "n_prior_encounters", "number_emergency"]].corr("spearman")

setup_mlflow(cfg)
with mlflow.start_run(run_name="diagnostics_session8"):
    mlflow.set_tags({"stage": "diagnostics", "data": "test (audit only)", "threshold": t})
    mlflow.log_metrics({"overall_recall": overall_recall, "perm_base_ap": base_ap})
    for f in ["diagnostics_slices.png", "diagnostics_shap_fn_vs_tp.png",
              "diagnostics_importance.png", "diagnostics_pdp_ice.png"]:
        mlflow.log_artifact(str(FIG / f))
    for f in ["diagnostics_slices.csv", "diagnostics_fn_vs_tp.csv",
              "diagnostics_shap_fn_vs_tp.csv", "diagnostics_permutation_importance.csv",
              "diagnostics_importance_comparison.csv", "diagnostics_pdp.csv"]:
        mlflow.log_artifact(str(TAB / f))

pd.set_option("display.width", 200)
print(sl.round(3).to_string(index=False))
print(err.round(3).to_string())
print(shap_cmp.head(8).round(3).to_string())
print(perm_df.head(12).round(4).to_string())
print(imp[[c for c in imp.columns if c.endswith("_rank")]].head(12).to_string())
print(pd.DataFrame(pdp_out).to_string())
print(corr.round(2).to_string())

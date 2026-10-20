# 30-day readmission risk for diabetic inpatients

Group: Muhammad TALHA, Jules BORDAIS, Elimane DIALLO, *Supervised Learning & Model Engineering* (Fall 2026)

**Dataset:** Diabetes 130-US hospitals 1999-2008 (OpenML id 4541), 101,766 encounters, 71,518 patients.

**Use case.** The hospital's beds are under pressure. At discharge we estimate, for each diabetic
inpatient, the probability of an unplanned readmission **within 30 days**. The discharge team
uses that probability to plan beds and to target follow-up calls.
The target is binary: `readmitted == "<30"` → 1, while `">30"` and `"NO"` → 0. Prevalence is 11.4%.

## Repository layout

```
configs/config.yaml      every choice in one place (cohort, CV, costs, Optuna budgets, MLflow)
src/readmission/         importable package
  data.py                cohort, binary target, patient-history feature, patient-disjoint split
  features.py            ReadmissionFeatures transformer (feature engineering, fitted per fold)
  pipelines.py           one imblearn Pipeline per model + Optuna search spaces
  evaluation.py          metrics (AP, ROC-AUC, Brier, ECE, top-k), CV loop, MLflow helpers
  calibration.py         Platt / isotonic calibrators fitted on out-of-fold predictions
  predict.py             loads the frozen bundle, scores raw rows (used by the API)
scripts/                 run in order, each writes figures/tables to reports/
  01_eda.py              cohort, target, missingness, signal, base rates by sensitive group
  02_experiments.py      untuned model progression, imbalance remedies, feature ablations
  03_tune.py             Optuna (TPE + median pruning), 5-fold patient-grouped CV, MLflow
  04_finalize.py         model choice, calibration, cost-based threshold, single test eval
  05_explain.py          SHAP global importance + 4 local explanations
  06_fairness.py         equal-opportunity audit on race + group-threshold mitigation
  07_export_runs.py      MLflow runs -> reports/tables/mlflow_runs.csv
  08_diagnostics.py      Session 8 audit: slices + CIs, FN vs TP, split/gain/permutation importance, PDP/ICE
  09_leakage_checks.py   rebuilds the leaky variants on purpose to measure what the audit prevents
  10_metrics_table.py    accuracy, precision, recall, F1, ROC-AUC, AP, Brier for every model, by phase
api/app.py               FastAPI endpoint (/predict, /predict_batch, /health)
tests/                   pytest: leakage-sensitive features, calibrators, API
notebooks/               supplementary only (team EDA)
reports/figures, tables  every number and plot used in the report and slides
```

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt && pip install -e .
python scripts/01_eda.py
python scripts/02_experiments.py
python scripts/03_tune.py lightgbm logreg random_forest catboost   # ~2-3 h on 12 cores
python scripts/04_finalize.py
python scripts/05_explain.py
python scripts/06_fairness.py
python scripts/08_diagnostics.py
python scripts/09_leakage_checks.py
python scripts/10_metrics_table.py
python scripts/07_export_runs.py
pytest -q
uvicorn api.app:app --reload          # http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Example request: `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"number_inpatient": 3, "diag_1": "428"}'`
(fields left out take the documented defaults, see `/docs`).

## Results

All five models below were scored on the same 5 patient-grouped folds. AP = average precision, i.e. PR-AUC; picking at random scores 0.114.

| Model (Optuna-tuned) | CV AP ± sd | CV ROC-AUC | Brier |
|---|---|---|---|
| Dummy (prevalence) | 0.114 | 0.500 | 0.101 |
| Logistic regression (L1) | 0.219 ± 0.012 | 0.670 | 0.097 |
| Random Forest | 0.227 ± 0.014 | 0.681 | 0.096 |
| LightGBM | 0.239 ± 0.013 | 0.686 | 0.097 |
| **CatBoost (final)** | **0.248 ± 0.015** | **0.688** | **0.095** |

The final model was evaluated once on the untouched test set (19,867 encounters):

* **Ranking:** AP 0.239 and ROC-AUC 0.683.
* **Probability quality:** Brier 0.095 and ECE 0.006. No recalibration was needed: the model expects 2,258 readmissions and 2,263 were observed.
* **Decision threshold:** 0.175, chosen on dev with C_FN = 5 × C_FP. At that threshold 15.4% of patients are flagged, recall is 33.6% and precision is 24.9%. Cost is −13% compared with flagging nobody.

Other findings:

* **Class weights and resampling** did not improve AP, and they degraded the Brier score (0.095 → 0.20 with weight 7.8).
* **Leakage, measured:** counting a patient's later stays would have inflated CV AP from 0.238 to 0.315. Mixing patients across folds barely matters here (0.240).
* **Diagnostics (Session 8):** half of all early readmissions are patients with no inpatient visit in the past year. Only 15% of them are flagged (95% CI 0.13–0.17), against 34% overall. The model relies on utilisation history.
* **Fairness on race** (equal opportunity): group thresholds cut the TPR gap from 0.29 to 0.13 at no cost. The gap is mostly driven by the small "Unknown" group.

Every figure and table is in `reports/`, and every run is in MLflow (`reports/tables/mlflow_runs.csv`).

## Methodological guard-rails

* **Cohort:** encounters that ended in death or hospice (discharge ids 11, 13, 14, 19, 20, 21) are removed, because these patients cannot be readmitted.
* **No patient on both sides of a split:** the 80/20 dev/test split and the 5 CV folds use `StratifiedGroupKFold` on `patient_nbr`.
* **The test set is used once,** in `04_finalize.py`, after the model, calibrator and threshold are frozen.
* **Everything that learns from data sits inside the Pipeline:** rare-category pooling, imputation, scaling, one-hot encoding, cross-fitted target encoding and resampling. It is therefore refitted in every fold.
* **Out-of-fold predictions on the dev split** are used to choose the calibrator and the decision threshold, and to fit the fairness thresholds.
* **`race` is the audited sensitive attribute** and is never a model input.

## Use of AI assistants
Claude (Anthropic) was used as a coding assistant to structure the package, write the scripts and draft the slides. The report states where it was used.

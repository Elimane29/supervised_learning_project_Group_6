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
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Using the prediction API

The API serves the frozen final model (CatBoost + cost-based threshold 0.175) as a discharge-time
risk score. The trained bundle `models/final_model.joblib` is versioned in the repository, so the API
works right after a clone; no retraining is needed.

### 1. Start it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt && pip install -e .
uvicorn api.app:app --reload
```

**From VS Code:** open the folder, select the `.venv` interpreter (`Ctrl+Shift+P` →
*Python: Select Interpreter*), then press **F5** and choose **"Run prediction API"**.
`.vscode/launch.json` starts the server, and VS Code shows the address.

Then open **http://127.0.0.1:8000/docs**: an interactive page where each endpoint can be tried
from the browser (**Try it out** → edit the patient → **Execute**).

### 2. Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Checks the service; returns the model name, calibration and threshold |
| POST | `/predict` | One discharged encounter → its 30-day readmission risk and a follow-up flag |
| POST | `/predict_batch` | A list of encounters (e.g. today's discharges) → one answer per patient |

### 3. What to send

Any field left out takes the default shown in `/docs`. The most influential inputs are:

| Field | Meaning | Example |
|---|---|---|
| `number_inpatient` | inpatient stays in the previous year | `3` |
| `n_prior_encounters` | earlier stays of this patient in the records | `4` |
| `discharge_disposition_id` | where the patient goes: `1` home, `6` home with health service, `3` skilled nursing | `3` |
| `diag_1`, `diag_2`, `diag_3` | ICD-9 diagnosis codes (primary first) | `"428"` (heart failure) |
| `age` | age bracket | `"[80-90)"` |
| `time_in_hospital` | length of stay in days, 1 to 14 (other values are rejected) | `9` |
| `insulin`, `metformin`, … | dose change during the stay: `No`, `Steady`, `Up`, `Down` | `"Up"` |
| `A1Cresult` | HbA1c result: `None` (not tested), `Norm`, `>7`, `>8` | `">8"` |

### 4. What comes back

```json
{"probability_readmission_30d": 0.3249, "flag_for_follow_up": true, "threshold": 0.175}
```

* `probability_readmission_30d`: calibrated risk of an early return. On the test set the predicted
  risks added up to 2,258 readmissions for 2,263 observed, so it can be summed to plan beds.
* `flag_for_follow_up`: `true` when the risk is at least 0.175. This threshold assumes a missed
  readmission costs five times a follow-up call.

### 5. Example patients

| Patient | Profile | Risk | Flag |
|---|---|---|---|
| A | 82, heart failure, 3 admissions last year, 9-day stay, insulin increased, sent to a care facility | 0.32 | yes |
| B | 55, diabetes, first admission, 3-day stay, sent home | 0.06 | no |
| C | 65, COPD + heart failure, 1 admission last year, home with health service | 0.19 | yes |

Patient A from a terminal (PowerShell):

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"age":"[80-90)","diag_1":"428","diag_2":"250.02","diag_3":"403","number_inpatient":3,"number_emergency":2,"n_prior_encounters":4,"time_in_hospital":9,"num_medications":22,"number_diagnoses":9,"discharge_disposition_id":3,"insulin":"Up","change":"Ch","A1Cresult":">8"}'
```

or with curl: `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"number_inpatient": 3, "diag_1": "428"}'`.

The model inherits the limits described in the report: it is a prioritisation aid, not a diagnosis,
and it under-detects first-time patients sent home (recall 15% for patients with no prior admission).
If you change the code or the libraries, rebuild the bundle with `python scripts/04_finalize.py`
(about 25 minutes, using the saved `models/best_params_*.json`).

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
Claude (Anthropic) was used as a coding and drafting assistant to help structure the Python package and scripts, support implementation of the experiment pipeline, and draft parts of the slides and report. ChatGPT (OpenAI) was also used to support debugging, conceptual review, interpretation of modelling choices and metrics, and preparation of the presentation.

The exploratory data analysis was carried out by the group, and the course material was studied independently in order to challenge AI-generated coding choices, verify that the modelling decisions were consistent with the concepts taught in class, and check the coherence of the project as a whole. All modelling decisions, validation choices, results and interpretations were reviewed by the group, and every member can explain and defend the final methodology and conclusions.

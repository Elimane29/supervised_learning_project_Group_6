# Graph Report - Supervised Learning Project - Group Muhammad-Jules-Elimane - Claudio  (2026-10-20)

## Corpus Check
- 29 files · ~59,384 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 34 file(s) not represented in the graph (top: .csv 30, .ipynb 2, (none) 1)

## Summary
- 254 nodes · 426 edges · 21 communities (10 shown, 11 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e95c280c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pipelines.py
- pandas
- evaluation.py
- numpy
- CatBoost (Final Model)
- data.py
- Project Instructions
- CatBoostModel
- Patient-Disjoint StratifiedGroupKFold Split
- MLflow Experiment Tracking
- Optuna Tuning (TPE + Median Pruning)
- excluded_discharge_ids [11,13,14,19,20,21]
- LightGBM
- FastAPI Prediction Endpoint (/predict, /predict_batch, /health)
- SHAP Global and Local Explanations
- readmission
- Dummy Prevalence Baseline
- Logistic Regression (L1)
- Random Forest

## God Nodes (most connected - your core abstractions)
1. `ReadmissionFeatures` - 12 edges
2. `Project Instructions` - 12 edges
3. `build_pipeline()` - 10 edges
4. `CatBoostModel` - 9 edges
5. `score()` - 8 edges
6. `ProbabilityCalibrator` - 7 edges
7. `build_dataset()` - 7 edges
8. `evaluate_cv()` - 7 edges
9. `predict()` - 6 edges
10. `predict_batch()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `test_rare_categories_are_learned_on_fit_data_only()` --uses--> `ReadmissionFeatures`  [INFERRED]
  tests/test_features.py → src/readmission/features.py
- `test_imputer_learns_from_training_rows_only()` --calls--> `load_config()`  [INFERRED]
  tests/test_features.py → src/readmission/config.py
- `test_imputer_learns_from_training_rows_only()` --calls--> `build_pipeline()`  [INFERRED]
  tests/test_features.py → src/readmission/pipelines.py
- `run()` --calls--> `evaluate_cv()`  [EXTRACTED]
  scripts/02_experiments.py → src/readmission/evaluation.py
- `run()` --calls--> `log_run()`  [EXTRACTED]
  scripts/02_experiments.py → src/readmission/evaluation.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Requirements for the End-to-End Supervised Learning Project** — projects_instructions_honest_data_splitting, projects_instructions_feature_engineering, projects_instructions_class_imbalance_handling, projects_instructions_model_comparison, projects_instructions_preprocessing_model_pipeline, projects_instructions_optuna_hyperparameter_tuning, projects_instructions_experiment_tracking, projects_instructions_shap_explainability, projects_instructions_calibration_diagnostics, projects_instructions_fairness_bias_audit, projects_instructions_served_prediction_endpoint [EXTRACTED 1.00]
- **Methodological Leakage Guard-rails** — readme_cohort_exclusion, readme_patient_grouped_split, readme_single_test_evaluation, readme_in_pipeline_learning, readme_out_of_fold_predictions, readme_race_sensitive_attribute [EXTRACTED 1.00]
- **Patient-Grouped CV Model Comparison** — readme_dummy_baseline, readme_logistic_regression_l1, readme_random_forest, readme_lightgbm, readme_catboost_final_model [EXTRACTED 1.00]
- **Finalization: Calibration, Threshold, Fairness from OOF Predictions** — readme_out_of_fold_predictions, readme_probability_calibration, readme_cost_based_threshold, readme_equal_opportunity_fairness_audit, configs_config_business_costs [INFERRED 0.85]

## Communities (21 total, 11 thin omitted)

### Community 0 - "pipelines.py"
Cohesion: 0.06
Nodes (12): run(), objective(), feature_names(), group_icd9(), ReadmissionFeatures, build_pipeline(), default_params(), make_model() (+4 more)

### Community 1 - "pandas"
Cohesion: 0.09
Nodes (9): EncounterBase, health(), predict(), predict_batch(), Prediction, results_table(), load_bundle(), score() (+1 more)

### Community 2 - "evaluation.py"
Cohesion: 0.08
Nodes (8): tune(), audit(), run(), load_config(), cross_validate_oof(), evaluate_cv(), log_run(), setup_mlflow()

### Community 3 - "numpy"
Cohesion: 0.09
Nodes (14): wilson(), add(), cross_fit_calibration(), _logit(), ProbabilityCalibrator, all_metrics(), expected_calibration_error(), probability_metrics() (+6 more)

### Community 4 - "CatBoost (Final Model)"
Cohesion: 0.10
Nodes (21): Central Configuration (config.yaml), Business Costs (cost_fn=5, cost_fp=1, review_capacity=0.20), Feature Settings (drop list, min_category_count, target encoding), positive_label: <30, selection_metric: average_precision, Binary Target: Readmitted <30 Days, Average Precision (PR-AUC), CatBoost (Final Model) (+13 more)

### Community 5 - "data.py"
Cohesion: 0.10
Nodes (8): add_prior_encounters(), build_dataset(), cv_folds(), cv_splitter(), dev_test_split(), load_raw(), load_splits(), test_prior_encounters_only_count_the_past()

### Community 6 - "Project Instructions"
Cohesion: 0.15
Nodes (12): Calibration Diagnostics, Class Imbalance Handling, Experiment Tracking, Fairness and Bias Audit, Feature Engineering, EDA and Honest Data Splitting, Comparison of at Least Three Models, Optuna Hyperparameter Tuning with Cross-Validation (+4 more)

### Community 9 - "Patient-Disjoint StratifiedGroupKFold Split"
Cohesion: 0.50
Nodes (3): CV Settings (n_splits=5, test_size=0.2, seed=42), group_column: patient_nbr, scikit-learn 1.9.1

### Community 10 - "MLflow Experiment Tracking"
Cohesion: 0.67
Nodes (3): MLflow Settings (sqlite tracking, diabetes_readmission_30d), MLflow Experiment Tracking, mlflow 3.17.0

### Community 11 - "Optuna Tuning (TPE + Median Pruning)"
Cohesion: 0.67
Nodes (3): Optuna Trial Budgets per Model, Optuna Tuning (TPE + Median Pruning), optuna 5.0.0

## Ambiguous Edges - Review These
- `30-day Readmission Risk for Diabetic Inpatients` → `xgboost 3.4.1`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to

## Knowledge Gaps
- **34 isolated node(s):** `readmission`, `MLflow Settings (sqlite tracking, diabetes_readmission_30d)`, `mlflow 3.17.0`, `Optuna Trial Budgets per Model`, `optuna 5.0.0` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 130 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ReadmissionFeatures` connect `pipelines.py` to `numpy`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **What connects `readmission`, `MLflow Settings (sqlite tracking, diabetes_readmission_30d)`, `mlflow 3.17.0` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pipelines.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06236786469344609 - nodes in this community are weakly interconnected._
- **Why does `CatBoostModel` connect `CatBoostModel` to `pipelines.py`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `pandas` be split into smaller, more focused modules?**
  _Cohesion score 0.08571428571428572 - nodes in this community are weakly interconnected._
- **Should `evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08048103607770583 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `30-day Readmission Risk for Diabetic Inpatients` and `xgboost 3.4.1`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
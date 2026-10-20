# Graph Report - Supervised Learning Project - Group Muhammad-Jules-Elimane - Claudio  (2026-10-18)

## Corpus Check
- Corpus is ~40,183 words - fits in a single context window. You may not need a graph.

## Summary
- 242 nodes · 377 edges · 22 communities (11 shown, 11 thin omitted)
- Extraction: 92% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 28 edges (avg confidence: 0.91)
- Token cost: 77,058 input · 0 output

## Community Hubs (Navigation)
- Modelling Pipelines & Tuning
- FastAPI Endpoint
- Script Utilities
- Evaluation & MLflow Logging
- Config & Business Framing
- Data Loading & Splitting
- Project Requirements
- Probability Calibration
- CatBoost sklearn Wrapper
- Patient-Grouped CV
- Experiment Tracking
- Optuna Budgets
- Cohort Exclusion
- LightGBM
- API Dependency
- SHAP Explanations
- Package Root
- Dummy Baseline
- Logistic Regression
- Random Forest

## God Nodes (most connected - your core abstractions)
1. `ReadmissionFeatures` - 12 edges
2. `Project Instructions` - 12 edges
3. `CatBoostModel` - 9 edges
4. `build_pipeline()` - 8 edges
5. `score()` - 8 edges
6. `ProbabilityCalibrator` - 7 edges
7. `build_dataset()` - 7 edges
8. `predict()` - 6 edges
9. `predict_batch()` - 6 edges
10. `tune()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `test_rare_categories_are_learned_on_fit_data_only()` --uses--> `ReadmissionFeatures`  [INFERRED]
  tests/test_features.py → src/readmission/features.py
- `test_calibrators_are_monotone_and_bounded()` --uses--> `ProbabilityCalibrator`  [INFERRED]
  tests/test_features.py → src/readmission/calibration.py
- `run()` --calls--> `build_pipeline()`  [EXTRACTED]
  scripts/02_experiments.py → src/readmission/pipelines.py
- `run()` --calls--> `default_params()`  [EXTRACTED]
  scripts/02_experiments.py → src/readmission/pipelines.py
- `objective()` --calls--> `cross_validate_oof()`  [EXTRACTED]
  scripts/03_tune.py → src/readmission/evaluation.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Requirements for the End-to-End Supervised Learning Project** — projects_instructions_honest_data_splitting, projects_instructions_feature_engineering, projects_instructions_class_imbalance_handling, projects_instructions_model_comparison, projects_instructions_preprocessing_model_pipeline, projects_instructions_optuna_hyperparameter_tuning, projects_instructions_experiment_tracking, projects_instructions_shap_explainability, projects_instructions_calibration_diagnostics, projects_instructions_fairness_bias_audit, projects_instructions_served_prediction_endpoint [EXTRACTED 1.00]
- **Methodological Leakage Guard-rails** — readme_cohort_exclusion, readme_patient_grouped_split, readme_single_test_evaluation, readme_in_pipeline_learning, readme_out_of_fold_predictions, readme_race_sensitive_attribute [EXTRACTED 1.00]
- **Patient-Grouped CV Model Comparison** — readme_dummy_baseline, readme_logistic_regression_l1, readme_random_forest, readme_lightgbm, readme_catboost_final_model [EXTRACTED 1.00]
- **Finalization: Calibration, Threshold, Fairness from OOF Predictions** — readme_out_of_fold_predictions, readme_probability_calibration, readme_cost_based_threshold, readme_equal_opportunity_fairness_audit, configs_config_business_costs [INFERRED 0.85]

## Communities (22 total, 11 thin omitted)

### Community 0 - "Modelling Pipelines & Tuning"
Cohesion: 0.07
Nodes (11): objective(), feature_names(), group_icd9(), ReadmissionFeatures, build_pipeline(), default_params(), make_model(), make_preprocessor() (+3 more)

### Community 1 - "FastAPI Endpoint"
Cohesion: 0.09
Nodes (9): EncounterBase, health(), predict(), predict_batch(), Prediction, results_table(), load_bundle(), score() (+1 more)

### Community 2 - "Script Utilities"
Cohesion: 0.09
Nodes (3): tune(), audit(), load_config()

### Community 3 - "Evaluation & MLflow Logging"
Cohesion: 0.13
Nodes (13): run(), all_metrics(), cross_validate_oof(), evaluate_cv(), expected_calibration_error(), log_run(), probability_metrics(), setup_mlflow() (+5 more)

### Community 4 - "Config & Business Framing"
Cohesion: 0.10
Nodes (21): Central Configuration (config.yaml), Business Costs (cost_fn=5, cost_fp=1, review_capacity=0.20), Feature Settings (drop list, min_category_count, target encoding), positive_label: <30, selection_metric: average_precision, Binary Target: Readmitted <30 Days, Average Precision (PR-AUC), CatBoost (Final Model) (+13 more)

### Community 5 - "Data Loading & Splitting"
Cohesion: 0.12
Nodes (8): add_prior_encounters(), build_dataset(), cv_folds(), cv_splitter(), dev_test_split(), load_raw(), load_splits(), test_prior_encounters_only_count_the_past()

### Community 6 - "Project Requirements"
Cohesion: 0.15
Nodes (12): Calibration Diagnostics, Class Imbalance Handling, Experiment Tracking, Fairness and Bias Audit, Feature Engineering, EDA and Honest Data Splitting, Comparison of at Least Three Models, Optuna Hyperparameter Tuning with Cross-Validation (+4 more)

### Community 7 - "Probability Calibration"
Cohesion: 0.21
Nodes (3): cross_fit_calibration(), _logit(), ProbabilityCalibrator

### Community 9 - "Patient-Grouped CV"
Cohesion: 0.50
Nodes (3): CV Settings (n_splits=5, test_size=0.2, seed=42), group_column: patient_nbr, scikit-learn 1.9.1

### Community 10 - "Experiment Tracking"
Cohesion: 0.67
Nodes (3): MLflow Settings (sqlite tracking, diabetes_readmission_30d), MLflow Experiment Tracking, mlflow 3.17.0

### Community 11 - "Optuna Budgets"
Cohesion: 0.67
Nodes (3): Optuna Trial Budgets per Model, Optuna Tuning (TPE + Median Pruning), optuna 5.0.0

## Ambiguous Edges - Review These
- `30-day Readmission Risk for Diabetic Inpatients` → `xgboost 3.4.1`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to

## Knowledge Gaps
- **34 isolated node(s):** `readmission`, `Dummy Prevalence Baseline`, `Logistic Regression (L1)`, `Random Forest`, `LightGBM` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 126 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ReadmissionFeatures` connect `Modelling Pipelines & Tuning` to `Evaluation & MLflow Logging`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **What connects `readmission`, `Dummy Prevalence Baseline`, `Logistic Regression (L1)` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Modelling Pipelines & Tuning` be split into smaller, more focused modules?**
  _Cohesion score 0.06585365853658537 - nodes in this community are weakly interconnected._
- **Why does `CatBoostModel` connect `CatBoost sklearn Wrapper` to `Modelling Pipelines & Tuning`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Should `FastAPI Endpoint` be split into smaller, more focused modules?**
  _Cohesion score 0.08571428571428572 - nodes in this community are weakly interconnected._
- **Should `Script Utilities` be split into smaller, more focused modules?**
  _Cohesion score 0.08739495798319327 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `30-day Readmission Risk for Diabetic Inpatients` and `xgboost 3.4.1`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
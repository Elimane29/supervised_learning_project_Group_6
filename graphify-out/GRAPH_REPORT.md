# Graph Report - Supervised Learning Project - Group Muhammad-Jules-Elimane  (2026-10-12)

## Corpus Check
- Corpus is ~1,025 words across the requirements, project brief, and notebook - fits in a single context window. You may not need a graph.

## Summary
- 36 nodes · 48 edges · 6 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: unmetered (host extraction usage unavailable; recorded counters are 0 input · 0 output)

## Community Hubs (Navigation)
- Supervised Learning Requirements
- Data and Visualization Stack
- Experiment and Fairness Tooling
- Prediction API Stack
- Diabetes Readmission Analysis
- Imbalanced Learning Stack

## God Nodes (most connected - your core abstractions)
1. `Python Requirements Manifest` - 18 edges
2. `Project Instructions` - 13 edges
3. `Data and Visualization` - 5 edges
4. `Tuning, Tracking, Explanations, and Fairness` - 5 edges
5. `Prediction API and Model Persistence` - 5 edges
6. `Supervised Learning and Imbalanced Data` - 3 edges
7. `Exploratory Data Analysis Notebook` - 2 edges
8. `Diabetes 130 Hospitals Dataset` - 2 edges
9. `Binary Readmission Within 30 Days` - 2 edges
10. `numpy` - 2 edges

## Surprising Connections (you probably didn't know these)
- `Project Instructions` --references--> `Diabetes 130 Hospitals Dataset`  [EXTRACTED]
  Projects_instructions.pdf → EDA.ipynb

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Requirements for the End-to-End Supervised Learning Project** — projects_instructions_honest_data_splitting, projects_instructions_feature_engineering, projects_instructions_class_imbalance_handling, projects_instructions_model_comparison, projects_instructions_preprocessing_model_pipeline, projects_instructions_optuna_hyperparameter_tuning, projects_instructions_experiment_tracking, projects_instructions_shap_explainability, projects_instructions_calibration_diagnostics, projects_instructions_fairness_bias_audit, projects_instructions_served_prediction_endpoint [EXTRACTED 1.00]

## Communities (6 total, 0 thin omitted)

### Community 0 - "Supervised Learning Requirements"
Cohesion: 0.15
Nodes (12): Calibration Diagnostics, Class Imbalance Handling, Experiment Tracking, Fairness and Bias Audit, Feature Engineering, EDA and Honest Data Splitting, Comparison of at Least Three Models, Optuna Hyperparameter Tuning with Cross-Validation (+4 more)

### Community 1 - "Data and Visualization Stack"
Cohesion: 0.60
Nodes (6): Data and Visualization, matplotlib, numpy, pandas, Python Requirements Manifest, seaborn

### Community 2 - "Experiment and Fairness Tooling"
Cohesion: 0.40
Nodes (5): fairlearn, mlflow, optuna, shap, Tuning, Tracking, Explanations, and Fairness

### Community 3 - "Prediction API Stack"
Cohesion: 0.40
Nodes (5): fastapi, joblib, Prediction API and Model Persistence, pydantic, uvicorn

### Community 4 - "Diabetes Readmission Analysis"
Cohesion: 0.50
Nodes (3): Binary Readmission Within 30 Days, Diabetes 130 Hospitals Dataset, Exploratory Data Analysis Notebook

### Community 5 - "Imbalanced Learning Stack"
Cohesion: 0.67
Nodes (3): imbalanced-learn, scikit-learn, Supervised Learning and Imbalanced Data

## Knowledge Gaps
- **11 isolated node(s):** `EDA and Honest Data Splitting`, `Feature Engineering`, `Class Imbalance Handling`, `Comparison of at Least Three Models`, `Single Preprocessing and Model Pipeline` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 13 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Python Requirements Manifest` connect `Data and Visualization Stack` to `Experiment and Fairness Tooling`, `Prediction API Stack`, `Imbalanced Learning Stack`?**
  _High betweenness centrality (0.218) - this node is a cross-community bridge._
- **What connects `EDA and Honest Data Splitting`, `Feature Engineering`, `Class Imbalance Handling` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Why does `Project Instructions` connect `Supervised Learning Requirements` to `Diabetes Readmission Analysis`?**
  _High betweenness centrality (0.192) - this node is a cross-community bridge._
- **Why does `Diabetes 130 Hospitals Dataset` connect `Diabetes Readmission Analysis` to `Supervised Learning Requirements`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
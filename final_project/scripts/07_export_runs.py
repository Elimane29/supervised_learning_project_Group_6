"""Export every MLflow run of the project to a CSV that is versioned with the code.

Run: python scripts/07_export_runs.py      (browse interactively: mlflow ui --backend-store-uri sqlite:///mlflow.db)
"""
import mlflow

from readmission.config import load_config
from readmission.evaluation import setup_mlflow

cfg = load_config()
setup_mlflow(cfg)
runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
keep = [c for c in runs.columns if c.startswith(("tags.mlflow.runName", "tags.stage",
                                                  "tags.block", "tags.model", "params.",
                                                  "metrics."))]
runs = runs[["run_id", "start_time", "status"] + keep].sort_values("start_time")
runs.to_csv(cfg["paths"]["tables"] / "mlflow_runs.csv", index=False, float_format="%.5f")
print(f"{len(runs)} runs exported")
print(runs.groupby(runs.get("tags.stage", "status").fillna("trial"), dropna=False).size())

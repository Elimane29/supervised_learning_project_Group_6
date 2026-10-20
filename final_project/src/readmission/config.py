"""Load the YAML configuration and resolve project paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# MLflow prints an assistant hint on import; keep script logs readable.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = PROJECT_ROOT / rel
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg

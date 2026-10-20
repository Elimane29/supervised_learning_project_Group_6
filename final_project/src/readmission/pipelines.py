"""One Pipeline per model family: raw rows in, readmission probability out.

Everything that learns from data (rare-category pooling, imputation, scaling, one-hot
vocabulary, target encoding, resampling) lives inside the Pipeline, so it is refitted
on the training part of every CV fold and never sees validation or test labels.
"""
from __future__ import annotations

from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .features import ReadmissionFeatures, feature_names

MODEL_NAMES = ["logreg", "random_forest", "lightgbm", "catboost"]


class CatBoostModel(ClassifierMixin, BaseEstimator):
    """Thin sklearn wrapper: CatBoostClassifier alters `cat_features` in its constructor,
    which breaks sklearn.clone (used for every CV fold). The booster is built in fit()."""

    def __init__(self, cat_features=(), iterations=600, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, auto_class_weights=None, random_seed=0):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            cat_features=list(self.cat_features), iterations=self.iterations,
            learning_rate=self.learning_rate, depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights=self.auto_class_weights, random_seed=self.random_seed,
            loss_function="Logloss", verbose=False, thread_count=-1,
            allow_writing_files=False).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_preprocessor(family: str, cfg: dict, use_target_encoding: bool | None = None):
    fcfg = cfg["features"]
    numeric, low_card, high_card = feature_names(fcfg["use_prior_encounters"])
    if use_target_encoding is None:
        use_target_encoding = fcfg["use_target_encoding"]

    if family == "catboost":
        # CatBoost encodes categoricals itself with ordered target statistics
        # (Session 4), so it receives the raw strings; only the order is fixed here.
        return "passthrough"

    linear = family == "logreg"
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if linear:  # regularised linear models need comparable scales (Session 2)
        num_steps.append(("scale", StandardScaler()))
    transformers = [
        ("num", SkPipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card),
    ]
    if use_target_encoding:
        # sklearn's TargetEncoder cross-fits during fit_transform (Session 5): a training
        # row is never encoded with its own label; smoothing shrinks rare codes to the mean.
        te_steps = [("te", TargetEncoder(target_type="binary", cv=KFold(
            5, shuffle=True, random_state=cfg["seed"])))]
        if linear:
            te_steps.append(("scale", StandardScaler()))
        transformers.append(("te", SkPipeline(te_steps), high_card))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_model(name: str, params: dict | None, cfg: dict):
    params = dict(params or {})
    seed = cfg["seed"]
    if name == "logreg":
        # Ridge (L2) with lbfgs or Lasso (L1) with liblinear. Elastic Net needs the saga
        # solver: measured 6x slower (95 s/fold) for no AP gain, so it is not searched.
        penalty = params.pop("penalty", "l2")
        solver, l1_ratio = ("liblinear", 1.0) if penalty == "l1" else ("lbfgs", 0.0)
        return LogisticRegression(solver=solver, l1_ratio=l1_ratio, max_iter=3000,
                                  random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, subsample_freq=1,
                              **params)
    if name == "catboost":
        if params.get("auto_class_weights") == "None":  # Optuna stores the choice as text
            params.pop("auto_class_weights")
        _, low_card, high_card = feature_names(cfg["features"]["use_prior_encounters"])
        return CatBoostModel(cat_features=tuple(low_card + high_card), random_seed=seed,
                             **params)
    raise ValueError(f"unknown model {name!r}")


def make_sampler(kind: str | None, seed: int):
    """Resampling of the TRAINING fold only (imblearn Pipeline skips it at predict time)."""
    if kind in (None, "none"):
        return None
    if kind == "undersample":
        return RandomUnderSampler(sampling_strategy=0.5, random_state=seed)
    if kind == "oversample":
        return RandomOverSampler(sampling_strategy=0.5, random_state=seed)
    raise ValueError(kind)


def build_pipeline(name: str, params: dict | None, cfg: dict, sampler: str | None = None,
                   use_prior_encounters: bool | None = None,
                   use_target_encoding: bool | None = None) -> Pipeline:
    fcfg = dict(cfg["features"])
    if use_prior_encounters is not None:
        fcfg["use_prior_encounters"] = use_prior_encounters
    local_cfg = {**cfg, "features": fcfg}

    steps = [("features", ReadmissionFeatures(fcfg["min_category_count"],
                                              fcfg["use_prior_encounters"]))]
    pre = make_preprocessor(name, local_cfg, use_target_encoding)
    if pre != "passthrough":
        steps.append(("preprocess", pre))
    smp = make_sampler(sampler, cfg["seed"])
    if smp is not None:
        steps.append(("sampler", smp))
    steps.append(("model", make_model(name, params, local_cfg)))
    return Pipeline(steps)


def default_params(name: str) -> dict:
    """Sensible untuned starting points, used for the baseline comparison."""
    return {
        "logreg": {"C": 1.0, "penalty": "l2"},
        "random_forest": {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt"},
        "lightgbm": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31,
                     "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.7},
        "catboost": {"iterations": 600, "learning_rate": 0.05, "depth": 6},
    }[name]


def suggest_params(name: str, trial) -> dict:
    """Optuna search spaces. Class weighting is a tuned hyperparameter (Session 6)."""
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),  # Session 2
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "random_forest":
        return {
            # n_estimators kept moderate: more trees only stabilise (Session 4), they do not
            # change the optimum of the other knobs, and every RF fit costs ~1 min here.
            "n_estimators": trial.suggest_int("n_estimators", 150, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "max_features": trial.suggest_float("max_features", 0.05, 0.4),
            "max_samples": trial.suggest_float("max_samples", 0.3, 0.7),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced_subsample"]),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["None", "SqrtBalanced", "Balanced"]),
        }
    raise ValueError(name)

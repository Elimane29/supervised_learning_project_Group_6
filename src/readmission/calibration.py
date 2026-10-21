"""Post-hoc probability calibration fitted on out-of-fold predictions (Session 7)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class ProbabilityCalibrator:
    """g(p_raw) -> p_calibrated. 'none' | 'sigmoid' (Platt on the logit) | 'isotonic'.

    Platt is applied to logit(p) rather than p, so that the identity map is inside the
    family (A=1, B=0): an already calibrated model is left essentially unchanged.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6).fit(_logit(p), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(p))[:, 1]
        if self.method == "isotonic":
            return self.model_.predict(p)
        return p


def cross_fit_calibration(p_oof, y, folds, method):
    """Calibrated OOF predictions where fold k's calibrator never saw fold k's labels.

    This lets us compare calibration methods on the dev set without the optimism of
    fitting and scoring the calibrator on the same rows.
    """
    y = np.asarray(y)
    out = np.empty_like(p_oof)
    for tr, va in folds:
        out[va] = ProbabilityCalibrator(method).fit(p_oof[tr], y[tr]).transform(p_oof[va])
    return out

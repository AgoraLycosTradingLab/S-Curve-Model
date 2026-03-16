# scurve/models/logistic.py
"""
Logistic (classification) model used by the S-Curve stack.

Goal:
- Provide a small, dependency-light logistic regression implementation for
  binary targets (0/1) with stable numerics and predictable behavior.
- Works with numpy arrays or pandas DataFrames.
- Intended for: regime / catalyst classification, probability-of-up-revision,
  probability-of-positive forward return, etc.

Notes:
- This is NOT intended to be a full ML framework.
- Uses Newton/IRLS with L2 regularization by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, List[float], List[List[float]]]


def _as_2d_array(X: ArrayLike) -> Tuple[np.ndarray, Optional[List[str]]]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, copy=False), list(X.columns)
    if isinstance(X, pd.Series):
        return X.to_frame().to_numpy(dtype=float, copy=False), [X.name or "x0"]
    Xn = np.asarray(X, dtype=float)
    if Xn.ndim == 1:
        Xn = Xn.reshape(-1, 1)
    return Xn, None


def _as_1d_array(y: ArrayLike) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must have exactly 1 column.")
        return y.iloc[:, 0].to_numpy(dtype=float, copy=False)
    if isinstance(y, pd.Series):
        return y.to_numpy(dtype=float, copy=False)
    yn = np.asarray(y, dtype=float)
    if yn.ndim != 1:
        yn = yn.reshape(-1)
    return yn


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _add_intercept(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=float), X])


def _clip_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = _clip_probs(p)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


@dataclass
class LogisticConfig:
    """
    Configuration for LogisticSCurveModel.

    l2:
        L2 regularization strength (lambda). Larger => more shrinkage.
        Intercept is NOT regularized.
    max_iter:
        Maximum IRLS iterations.
    tol:
        Convergence tolerance on parameter change (L2 norm).
    standardize:
        If True, z-score features using training mean/std.
    clip_z:
        Clips linear score z to [-clip_z, clip_z] before sigmoid for extra safety.
    """

    l2: float = 1.0
    max_iter: int = 50
    tol: float = 1e-6
    standardize: bool = True
    clip_z: float = 50.0


class LogisticSCurveModel:
    """
    Simple binary logistic regression (IRLS/Newton) with optional standardization.

    API:
        fit(X, y)
        predict_proba(X)
        predict(X, threshold=0.5)
        decision_function(X)
        metrics(X, y)
        to_dict() / from_dict()

    Target:
        y must be binary {0,1} (booleans okay).
    """

    def __init__(self, config: Optional[LogisticConfig] = None):
        self.config = config or LogisticConfig()

        self.feature_names_: Optional[List[str]] = None
        self.coef_: Optional[np.ndarray] = None  # includes intercept at index 0
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.fitted_: bool = False

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LogisticSCurveModel":
        return cls(LogisticConfig(**cfg))

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "LogisticSCurveModel":
        Xn, names = _as_2d_array(X)
        yn = _as_1d_array(y)

        if Xn.shape[0] != yn.shape[0]:
            raise ValueError(f"X rows ({Xn.shape[0]}) != y rows ({yn.shape[0]}).")

        # Coerce y to {0,1}
        yb = (yn > 0.5).astype(float)

        # weights
        if sample_weight is None:
            w = np.ones_like(yb, dtype=float)
        else:
            w = _as_1d_array(sample_weight).astype(float)
            if w.shape[0] != yb.shape[0]:
                raise ValueError("sample_weight length must match y.")
            w = np.clip(w, 0.0, np.inf)
            if np.all(w == 0):
                raise ValueError("All sample_weight are zero.")

        # Standardize
        Xz = Xn.copy()
        if self.config.standardize:
            mu = np.nanmean(Xz, axis=0)
            sd = np.nanstd(Xz, axis=0)
            sd = np.where(sd <= 0.0, 1.0, sd)

            # Fill NaNs with mean before standardizing
            Xz = np.where(np.isnan(Xz), mu, Xz)
            Xz = (Xz - mu) / sd

            self.mean_ = mu
            self.std_ = sd
        else:
            # Fill NaNs with 0.0 for determinism
            Xz = np.where(np.isnan(Xz), 0.0, Xz)
            self.mean_ = None
            self.std_ = None

        Xd = _add_intercept(Xz)
        n, p1 = Xd.shape  # p1 = p + 1 (intercept)
        self.n_features_ = p1 - 1
        self.feature_names_ = names

        # Initialize coefficients
        beta = np.zeros(p1, dtype=float)

        # L2 regularization matrix (do not penalize intercept)
        l2 = float(self.config.l2)
        reg = np.eye(p1, dtype=float) * l2
        reg[0, 0] = 0.0

        # IRLS / Newton
        for _ in range(int(self.config.max_iter)):
            z = Xd @ beta
            if self.config.clip_z is not None:
                cz = float(self.config.clip_z)
                z = np.clip(z, -cz, cz)

            p = _sigmoid(z)
            # Variance p*(1-p)
            v = p * (1.0 - p)
            v = np.clip(v, 1e-12, np.inf)

            # Weighted IRLS: solve (X^T W X + reg) beta = X^T W y + reg*beta? (MAP)
            # For L2 penalty, the Newton step uses reg directly in Hessian.
            # Gradient: X^T (w*(y-p)) - reg*beta (but intercept excluded via reg[0,0]=0)
            # Hessian: X^T diag(w*v) X + reg
            W = w * v  # shape (n,)
            XTWX = Xd.T @ (Xd * W[:, None])
            H = XTWX + reg

            grad = Xd.T @ (w * (yb - p)) - (reg @ beta)

            # Solve for step: H * step = grad
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                # Fallback: add jitter to diagonal
                jitter = 1e-6
                step = np.linalg.solve(H + np.eye(p1) * jitter, grad)

            beta_new = beta + step

            if np.linalg.norm(beta_new - beta) < float(self.config.tol):
                beta = beta_new
                break

            beta = beta_new

        self.coef_ = beta
        self.fitted_ = True
        return self

    def _transform_X(self, X: ArrayLike) -> np.ndarray:
        if not self.fitted_ or self.coef_ is None:
            raise RuntimeError("Model is not fitted.")

        Xn, _ = _as_2d_array(X)
        if self.n_features_ is None:
            raise RuntimeError("Model has missing n_features_.")

        if Xn.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {Xn.shape[1]}.")

        Xz = Xn.copy()
        if self.config.standardize:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Standardization stats missing.")
            mu = self.mean_
            sd = self.std_
            Xz = np.where(np.isnan(Xz), mu, Xz)
            Xz = (Xz - mu) / sd
        else:
            Xz = np.where(np.isnan(Xz), 0.0, Xz)

        return _add_intercept(Xz)

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        Xd = self._transform_X(X)
        z = Xd @ self.coef_
        if self.config.clip_z is not None:
            cz = float(self.config.clip_z)
            z = np.clip(z, -cz, cz)
        return z

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        z = self.decision_function(X)
        p1 = _sigmoid(z)
        # return [P(class0), P(class1)]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(threshold)).astype(int)

    def metrics(self, X: ArrayLike, y: ArrayLike, threshold: float = 0.5) -> Dict[str, float]:
        yb = (_as_1d_array(y) > 0.5).astype(int)
        p = self.predict_proba(X)[:, 1]
        yhat = (p >= float(threshold)).astype(int)

        # confusion
        tp = int(((yhat == 1) & (yb == 1)).sum())
        tn = int(((yhat == 0) & (yb == 0)).sum())
        fp = int(((yhat == 1) & (yb == 0)).sum())
        fn = int(((yhat == 0) & (yb == 1)).sum())

        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = 2 * prec * rec / max(1e-12, (prec + rec))

        return {
            "logloss": _logloss(yb.astype(float), p.astype(float)),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }

    def coef_table(self) -> pd.DataFrame:
        if not self.fitted_ or self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        names = ["intercept"]
        if self.feature_names_ is not None:
            names += list(self.feature_names_)
        else:
            names += [f"x{i}" for i in range(self.n_features_ or (len(self.coef_) - 1))]
        return pd.DataFrame({"feature": names, "coef": self.coef_.astype(float)})

    def to_dict(self) -> Dict[str, Any]:
        if not self.fitted_ or self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        return {
            "config": {
                "l2": self.config.l2,
                "max_iter": self.config.max_iter,
                "tol": self.config.tol,
                "standardize": self.config.standardize,
                "clip_z": self.config.clip_z,
            },
            "feature_names": self.feature_names_,
            "coef": self.coef_.tolist(),
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "n_features": self.n_features_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogisticSCurveModel":
        m = cls(LogisticConfig(**d.get("config", {})))
        m.feature_names_ = d.get("feature_names")
        m.coef_ = np.asarray(d["coef"], dtype=float)
        m.mean_ = None if d.get("mean") is None else np.asarray(d["mean"], dtype=float)
        m.std_ = None if d.get("std") is None else np.asarray(d["std"], dtype=float)
        m.n_features_ = int(d.get("n_features", len(m.coef_) - 1))
        m.fitted_ = True
        return m
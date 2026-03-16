# scurve/models/bass.py
"""
Bass Diffusion (S-curve) model.

Bass model is commonly used for adoption/diffusion processes:
- p: coefficient of innovation (external influence)
- q: coefficient of imitation (internal/social influence)
- m: market potential (upper asymptote / total adopters)

Cumulative adoption:
    F(t) = (1 - exp(-(p+q)t)) / (1 + (q/p) * exp(-(p+q)t))          (for p > 0)
    Y(t) = m * F(t)

Adoption rate ("sales" / new adopters per time):
    y(t) = m * f(t)
    f(t) = ((p+q)^2 / p) * exp(-(p+q)t) / (1 + (q/p)*exp(-(p+q)t))^2

This module provides:
- BassCurve: predicts cumulative and rate series, plus utility functions.
- BassFitter: deterministic derivative-free fitter (coarse-to-fine grid).
  Supports fitting to cumulative series or rate series.

Design goals:
- Deterministic, dependency-light (numpy + optional pandas), no black-box optimizer.
- Numerically stable with safe exponent clipping.

Limits:
- Not a high-performance optimizer.
- Best used with reasonably smooth time-series (weekly/monthly), not noisy tick-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

Array1D = Union[np.ndarray, pd.Series, list]


def _as_1d(x: Array1D) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=False)
    xn = np.asarray(x, dtype=float)
    if xn.ndim != 1:
        xn = xn.reshape(-1)
    return xn


def _safe_exp(x: np.ndarray, clip: float = 60.0) -> np.ndarray:
    return np.exp(np.clip(x, -clip, clip))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


@dataclass(frozen=True)
class BassParams:
    p: float  # innovation
    q: float  # imitation
    m: float  # market potential

    def validate(self) -> None:
        if not np.isfinite(self.p) or self.p <= 0:
            raise ValueError("p must be finite and > 0.")
        if not np.isfinite(self.q) or self.q < 0:
            raise ValueError("q must be finite and >= 0.")
        if not np.isfinite(self.m) or self.m <= 0:
            raise ValueError("m must be finite and > 0.")


class BassCurve:
    def __init__(self, params: BassParams):
        params.validate()
        self.params = params

    def cumulative_fraction(self, t: Array1D) -> np.ndarray:
        """
        F(t) in [0,1], cumulative adoption fraction.
        """
        t = _as_1d(t)
        p, q = self.params.p, self.params.q
        a = p + q
        e = _safe_exp(-a * t)
        denom = 1.0 + (q / p) * e
        F = (1.0 - e) / denom
        return np.clip(F, 0.0, 1.0)

    def cumulative(self, t: Array1D) -> np.ndarray:
        """
        Y(t) = m * F(t), cumulative adopters.
        """
        return self.params.m * self.cumulative_fraction(t)

    def rate_fraction(self, t: Array1D) -> np.ndarray:
        """
        f(t), adoption rate fraction (per unit of t).
        """
        t = _as_1d(t)
        p, q = self.params.p, self.params.q
        a = p + q
        e = _safe_exp(-a * t)
        denom = 1.0 + (q / p) * e
        # f(t) = ((p+q)^2/p) * e / denom^2
        f = ((a * a) / p) * e / (denom * denom)
        return np.clip(f, 0.0, np.inf)

    def rate(self, t: Array1D) -> np.ndarray:
        """
        y(t) = m * f(t), new adopters per unit of t.
        """
        return self.params.m * self.rate_fraction(t)

    def peak_time(self) -> float:
        """
        Time of peak adoption rate (if q > 0):
            t* = (1/(p+q)) * ln(q/p)
        If q <= p (or q==0), peak is at t=0.
        """
        p, q = self.params.p, self.params.q
        if q <= 0 or q <= p:
            return 0.0
        return float(np.log(q / p) / (p + q))

    def to_dict(self) -> Dict[str, Any]:
        return {"p": self.params.p, "q": self.params.q, "m": self.params.m}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BassCurve":
        return cls(BassParams(p=float(d["p"]), q=float(d["q"]), m=float(d["m"])))


@dataclass
class BassFitConfig:
    """
    Fit configuration.

    fit_to:
        "cumulative" or "rate"
    loss:
        "mse" or "mae"
    grid_sizes:
        (np, nq, nm) coarse grid sizes for p, q, m
    refine_steps:
        Number of refinement rounds
    refine_shrink:
        Shrink factor for bounds each refinement
    p_bounds:
        Absolute bounds for p
    q_bounds:
        Absolute bounds for q
    m_bounds_mult:
        If m_bounds not provided, infer as [max(y)*m0, max(y)*m1] for cumulative,
        or [cum_end*m0, cum_end*m1] if fitting rate with inferred cumulative.
    """

    fit_to: str = "cumulative"
    loss: str = "mse"
    grid_sizes: Tuple[int, int, int] = (24, 28, 20)
    refine_steps: int = 6
    refine_shrink: float = 0.55
    p_bounds: Tuple[float, float] = (1e-6, 5.0)
    q_bounds: Tuple[float, float] = (0.0, 20.0)
    m_bounds_mult: Tuple[float, float] = (0.9, 1.8)


class BassFitter:
    """
    Deterministic derivative-free fitter for Bass parameters (p, q, m).

    Typical:
        curve, info = BassFitter().fit(t, y, fit_to="cumulative")

    Guidance:
    - Use t as "time since launch" (0..T), evenly spaced if possible.
    - If your data is cumulative adopters, fit_to="cumulative".
    - If your data is new adopters per period, fit_to="rate".
    """

    def __init__(self, config: Optional[BassFitConfig] = None):
        self.config = config or BassFitConfig()

    def fit(
        self,
        t: Array1D,
        y: Array1D,
        *,
        fit_to: Optional[str] = None,
        loss: Optional[str] = None,
        p_bounds: Optional[Tuple[float, float]] = None,
        q_bounds: Optional[Tuple[float, float]] = None,
        m_bounds: Optional[Tuple[float, float]] = None,
        weights: Optional[Array1D] = None,
    ) -> Tuple[BassCurve, Dict[str, Any]]:
        t = _as_1d(t)
        y = _as_1d(y)

        if t.shape[0] != y.shape[0]:
            raise ValueError("t and y must have the same length.")
        if t.shape[0] < 6:
            raise ValueError("Need at least 6 observations to fit Bass model.")

        # sort by t
        idx = np.argsort(t)
        t = t[idx]
        y = y[idx]

        fit_to_eff = (fit_to or self.config.fit_to).lower().strip()
        if fit_to_eff not in {"cumulative", "rate"}:
            raise ValueError("fit_to must be 'cumulative' or 'rate'.")

        loss_eff = (loss or self.config.loss).lower().strip()
        if loss_eff not in {"mse", "mae"}:
            raise ValueError("loss must be 'mse' or 'mae'.")

        # weights
        if weights is not None:
            w = _as_1d(weights)[idx]
            w = np.clip(w, 0.0, np.inf)
            if np.all(w == 0):
                raise ValueError("All weights are zero.")
            w = w / np.mean(w)
        else:
            w = None

        def loss_fn(a: np.ndarray, b: np.ndarray) -> float:
            if w is None:
                return _mae(a, b) if loss_eff == "mae" else _mse(a, b)
            if loss_eff == "mae":
                return float(np.sum(w * np.abs(a - b)) / np.sum(w))
            return float(np.sum(w * (a - b) ** 2) / np.sum(w))

        # bounds
        p_lo, p_hi = map(float, (p_bounds or self.config.p_bounds))
        q_lo, q_hi = map(float, (q_bounds or self.config.q_bounds))
        p_lo = max(p_lo, 1e-12)
        p_hi = max(p_hi, p_lo * 1.0001)
        q_lo = max(q_lo, 0.0)
        q_hi = max(q_hi, q_lo + 1e-9)

        # infer m bounds if missing
        y_max = float(np.nanmax(y))
        if not np.isfinite(y_max):
            raise ValueError("y contains no finite values.")

        if m_bounds is None:
            m0, m1 = self.config.m_bounds_mult
            if fit_to_eff == "cumulative":
                anchor = max(1e-9, y_max)
            else:
                # If fitting rate, approximate cumulative end as sum(rate * dt) if dt stable.
                dt = np.diff(t)
                if np.all(dt > 0) and np.nanmean(dt) > 0:
                    # trapezoid integrate to estimate cumulative level at end
                    anchor = float(np.trapz(np.clip(y, 0.0, np.inf), t))
                    anchor = max(anchor, y_max)  # fallback safety
                else:
                    anchor = max(1e-9, y_max)
            m_bounds = (max(1e-9, anchor * m0), max(1e-9, anchor * m1))

        m_lo, m_hi = map(float, m_bounds)
        m_lo = max(m_lo, 1e-12)
        m_hi = max(m_hi, m_lo * 1.0001)

        # grid helpers
        def linspace(a: float, b: float, n: int) -> np.ndarray:
            if n <= 1:
                return np.array([(a + b) / 2.0], dtype=float)
            if a == b:
                return np.full(n, a, dtype=float)
            return np.linspace(a, b, n, dtype=float)

        # Use log grids for p (often small), and for (q+eps)
        npg, nqg, nmg = self.config.grid_sizes
        logp_lo, logp_hi = np.log(p_lo), np.log(p_hi)
        p_grid = np.exp(linspace(logp_lo, logp_hi, int(npg)))

        # q can be zero: grid in log( q + q_eps )
        q_eps = 1e-8
        logq_lo, logq_hi = np.log(q_lo + q_eps), np.log(q_hi + q_eps)
        q_grid = np.exp(linspace(logq_lo, logq_hi, int(nqg))) - q_eps
        q_grid = np.clip(q_grid, 0.0, np.inf)

        m_grid = linspace(m_lo, m_hi, int(nmg))

        def predict_series(p: float, q: float, m: float) -> np.ndarray:
            a = p + q
            e = _safe_exp(-a * t)
            denom = 1.0 + (q / p) * e
            if fit_to_eff == "cumulative":
                F = (1.0 - e) / denom
                return m * np.clip(F, 0.0, 1.0)
            # rate
            f = ((a * a) / p) * e / (denom * denom)
            return m * np.clip(f, 0.0, np.inf)

        # coarse search
        best = {"p": None, "q": None, "m": None, "loss": float("inf")}
        for pc in p_grid:
            for qc in q_grid:
                # If both p and q extremely tiny, model is nearly flat; allow but will lose vs data.
                for mc in m_grid:
                    yhat = predict_series(float(pc), float(qc), float(mc))
                    val = loss_fn(y, yhat)
                    if val < best["loss"]:
                        best = {"p": float(pc), "q": float(qc), "m": float(mc), "loss": float(val)}

        # refinement loops: shrink bounds around best
        p_c, q_c, m_c = best["p"], best["q"], best["m"]

        logp_lo_g, logp_hi_g = logp_lo, logp_hi
        logq_lo_g, logq_hi_g = logq_lo, logq_hi

        for _ in range(int(self.config.refine_steps)):
            shrink = float(self.config.refine_shrink)

            # spans
            logp_span = (logp_hi_g - logp_lo_g) * shrink
            logq_span = (logq_hi_g - logq_lo_g) * shrink
            m_span = (m_hi - m_lo) * shrink

            # centers
            logp_c = float(np.log(max(p_c, 1e-12)))
            logq_c = float(np.log(max(q_c + q_eps, 1e-12)))

            # local bounds in log space for p and q
            lp0 = logp_c - 0.5 * logp_span
            lp1 = logp_c + 0.5 * logp_span
            lq0 = logq_c - 0.5 * logq_span
            lq1 = logq_c + 0.5 * logq_span

            # clamp to global
            lp0 = max(lp0, logp_lo)
            lp1 = min(lp1, logp_hi)
            lq0 = max(lq0, logq_lo)
            lq1 = min(lq1, logq_hi)

            # local bounds for m
            m0 = max(m_lo, m_c - 0.5 * m_span)
            m1 = min(m_hi, m_c + 0.5 * m_span)
            m1 = max(m1, m0 * 1.0001)

            # smaller grids
            p_grid2 = np.exp(linspace(lp0, lp1, max(10, int(npg // 3))))
            q_grid2 = np.exp(linspace(lq0, lq1, max(12, int(nqg // 3)))) - q_eps
            q_grid2 = np.clip(q_grid2, 0.0, np.inf)
            m_grid2 = linspace(m0, m1, max(8, int(nmg // 3)))

            improved = False
            for pr in p_grid2:
                for qr in q_grid2:
                    for mr in m_grid2:
                        yhat = predict_series(float(pr), float(qr), float(mr))
                        val = loss_fn(y, yhat)
                        if val < best["loss"]:
                            best = {"p": float(pr), "q": float(qr), "m": float(mr), "loss": float(val)}
                            improved = True

            p_c, q_c, m_c = best["p"], best["q"], best["m"]
            if improved:
                # tighten search region for next iteration
                logp_lo_g, logp_hi_g = lp0, lp1
                logq_lo_g, logq_hi_g = lq0, lq1
                m_lo, m_hi = m0, m1

        params = BassParams(p=float(best["p"]), q=float(best["q"]), m=float(best["m"]))
        curve = BassCurve(params)

        info = {
            "loss": float(best["loss"]),
            "fit_to": fit_to_eff,
            "loss_type": loss_eff,
            "bounds_used": {
                "p_bounds": (float(p_bounds or self.config.p_bounds)[0], float(p_bounds or self.config.p_bounds)[1])
                if p_bounds is not None
                else (float(self.config.p_bounds[0]), float(self.config.p_bounds[1])),
                "q_bounds": (float(q_bounds or self.config.q_bounds)[0], float(q_bounds or self.config.q_bounds)[1])
                if q_bounds is not None
                else (float(self.config.q_bounds[0]), float(self.config.q_bounds[1])),
                "m_bounds": (float((m_bounds or (m_lo, m_hi))[0]), float((m_bounds or (m_lo, m_hi))[1])),
            },
            "y_summary": {"min": float(np.nanmin(y)), "max": float(np.nanmax(y)), "n": int(len(y))},
            "params": curve.to_dict(),
            "diagnostics": {
                "peak_time": curve.peak_time(),
                "peak_rate": float(np.max(curve.rate(t))),
                "end_cumulative": float(curve.cumulative(t)[-1]),
            },
        }
        return curve, info
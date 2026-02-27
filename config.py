"""
scurve.core.config

Configuration loading + lightweight validation for Agora Lycos S-Curve model.

Design goals:
- Deterministic, explicit behavior
- No hidden defaults beyond safe fallbacks
- Clear error messages for missing/invalid config keys
- Support YAML config files and snapshotting for reproducibility
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing required fields or is invalid."""


# -------------------------
# Basic file IO
# -------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config file into a dict and apply overlay mode toggles.

    Parameters
    ----------
    path : str | Path
        Path to YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ConfigError
        If YAML is empty or invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None or not isinstance(cfg, dict):
        raise ConfigError(f"Config file is empty or not a mapping: {p}")

    validate_config(cfg)
    cfg = apply_overlay_mode(cfg)  # overlay_mode becomes the source of truth
    return cfg


def save_config_snapshot(cfg: Dict[str, Any], out_path: str | Path) -> None:
    """
    Write a config snapshot (YAML) for run reproducibility.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# -------------------------
# Overlay mode routing
# -------------------------

def apply_overlay_mode(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce overlay_mode A/B/C/D by overriding overlays.*.enabled flags.

    Policy:
    - overlay_mode is the single source of truth
    - overlays.<block>.enabled values in YAML are treated as defaults only

    Modes:
      A: Revenue only
      B: Revenue + EPS revisions
      C: Revenue + EPS revisions + breakout confirmation
      D: Full charter overlays (same as C for now; you can add more later)
    """
    mode = str(cfg.get("overlay_mode", "A")).strip().upper()
    if mode not in {"A", "B", "C", "D"}:
        raise ConfigError(f"overlay_mode must be one of A,B,C,D; got '{mode}'.")

    overlays = cfg.setdefault("overlays", {})
    if not isinstance(overlays, dict):
        raise ConfigError("overlays must be a mapping if provided.")

    def _ensure_block(name: str) -> Dict[str, Any]:
        blk = overlays.setdefault(name, {})
        if not isinstance(blk, dict):
            raise ConfigError(f"overlays.{name} must be a mapping.")
        return blk

    eps_blk = _ensure_block("eps_revisions")
    brk_blk = _ensure_block("breakout")
    mkt_blk = _ensure_block("market_filters")

    # mode -> enabled flags
    eps_enabled = mode in {"B", "C", "D"}
    brk_enabled = mode in {"C", "D"}
    mkt_enabled = True  # typically always on for a stock screener

    eps_blk["enabled"] = bool(eps_enabled)
    brk_blk["enabled"] = bool(brk_enabled)
    mkt_blk["enabled"] = bool(mkt_enabled)

    # store normalized mode back
    cfg["overlay_mode"] = mode
    return cfg


# -------------------------
# Validation helpers
# -------------------------

def _require_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    sec = cfg.get(key)
    if not isinstance(sec, dict):
        raise ConfigError(f"Missing or invalid section '{key}' (expected mapping).")
    return sec


def _require_key(sec: Dict[str, Any], key: str, expected_type: type | tuple[type, ...]) -> Any:
    if key not in sec:
        raise ConfigError(f"Missing required key '{key}' in section.")
    val = sec[key]
    if not isinstance(val, expected_type):
        raise ConfigError(f"Invalid type for key '{key}': expected {expected_type}, got {type(val)}.")
    return val


def _optional_key(sec: Dict[str, Any], key: str, expected_type: type | tuple[type, ...], default: Any) -> Any:
    val = sec.get(key, default)
    if val is None:
        return val
    if not isinstance(val, expected_type):
        raise ConfigError(f"Invalid type for key '{key}': expected {expected_type}, got {type(val)}.")
    return val


def _require_positive(name: str, x: float) -> None:
    if float(x) <= 0.0:
        raise ConfigError(f"'{name}' must be > 0, got {x}.")


def _require_nonnegative(name: str, x: float) -> None:
    if float(x) < 0.0:
        raise ConfigError(f"'{name}' must be >= 0, got {x}.")


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate required structure + key constraints.

    Expected top-level sections:
      run, data, fit, gates, scoring, ranking, risk, backtest

    Optional:
      overlay_mode, overlays
    """
    run = _require_section(cfg, "run")
    data = _require_section(cfg, "data")
    fit = _require_section(cfg, "fit")
    gates = _require_section(cfg, "gates")
    scoring = _require_section(cfg, "scoring")
    ranking = _require_section(cfg, "ranking")
    risk = _require_section(cfg, "risk")
    backtest = _require_section(cfg, "backtest")

    # run
    seed = _optional_key(run, "seed", (int,), 42)
    _require_nonnegative("run.seed", seed)
    _optional_key(run, "output_root", (str,), "runs")
    _optional_key(run, "log_root", (str,), "logs")

    # data
    min_q = _require_key(data, "min_quarters", (int,))
    if int(min_q) < 4:
        raise ConfigError(f"data.min_quarters too small: {min_q} (expected >= 4, prefer 16+).")
    lag_days = _require_key(data, "lag_days", (int,))
    _require_nonnegative("data.lag_days", lag_days)

    price_min = _optional_key(data, "price_min", (int, float), 0.0)
    _require_nonnegative("data.price_min", price_min)

    mcap = _optional_key(data, "market_cap_min_usd", (int, float), 0.0)
    _require_nonnegative("data.market_cap_min_usd", mcap)

    adv = _optional_key(data, "adv_dollar_min_usd", (int, float), 0.0)
    _require_nonnegative("data.adv_dollar_min_usd", adv)

    # fit
    models = _require_key(fit, "models", (list,))
    if not models:
        raise ConfigError("fit.models must be a non-empty list (e.g., ['logistic','gompertz']).")
    for m in models:
        if m not in ("logistic", "gompertz", "bass"):
            raise ConfigError(f"fit.models contains unsupported model '{m}'.")
    _optional_key(fit, "loss", (str,), "huber")
    _optional_key(fit, "f_scale", (int, float), 1.0)
    max_nfev = _optional_key(fit, "max_nfev", (int,), 5000)
    _require_positive("fit.max_nfev", max_nfev)

    nrmse_max = _require_key(fit, "nrmse_max", (int, float))
    _require_positive("fit.nrmse_max", nrmse_max)

    k_mult = _require_key(fit, "k_bounds_multiplier", (list, tuple))
    if len(k_mult) != 2:
        raise ConfigError("fit.k_bounds_multiplier must be [low, high].")
    if float(k_mult[0]) <= 0.0 or float(k_mult[1]) <= float(k_mult[0]):
        raise ConfigError(f"Invalid fit.k_bounds_multiplier: {k_mult}")

    r_bounds = _require_key(fit, "r_bounds", (list, tuple))
    if len(r_bounds) != 2:
        raise ConfigError("fit.r_bounds must be [low, high].")
    if float(r_bounds[0]) <= 0.0 or float(r_bounds[1]) <= float(r_bounds[0]):
        raise ConfigError(f"Invalid fit.r_bounds: {r_bounds}")

    t0_pad = _optional_key(fit, "t0_pad_quarters", (int,), 2)
    _require_nonnegative("fit.t0_pad_quarters", t0_pad)

    _optional_key(fit, "k_bound_eps", (int, float), 1e-3)

    # gates
    _optional_key(gates, "require_fit_ok", (bool,), True)
    _optional_key(gates, "reject_if_k_on_upper_bound", (bool,), False)

    # scoring
    _require_key(scoring, "maturity_center", (int, float))
    mw = _require_key(scoring, "maturity_width", (int, float))
    _require_positive("scoring.maturity_width", mw)
    _optional_key(scoring, "stage_weight", (int, float), 0.55)
    _optional_key(scoring, "slope_weight", (int, float), 0.25)
    _optional_key(scoring, "accel_weight", (int, float), 0.20)
    _optional_key(scoring, "k_bound_penalty", (int, float), 0.70)

    wsum = float(scoring.get("stage_weight", 0.55)) + float(scoring.get("slope_weight", 0.25)) + float(scoring.get("accel_weight", 0.20))
    if abs(wsum - 1.0) > 1e-6:
        raise ConfigError(f"scoring weights must sum to 1.0 (got {wsum}).")

    # ranking
    tp = _require_key(ranking, "top_percentile", (int, float))
    if not (0.0 < float(tp) <= 1.0):
        raise ConfigError(f"ranking.top_percentile must be in (0,1], got {tp}.")

    # risk
    mpw = _optional_key(risk, "max_position_weight", (int, float), 0.05)
    if not (0.0 < float(mpw) <= 1.0):
        raise ConfigError(f"risk.max_position_weight must be in (0,1], got {mpw}.")
    sc = _optional_key(risk, "sector_cap", (int, float), 0.35)
    if not (0.0 < float(sc) <= 1.0):
        raise ConfigError(f"risk.sector_cap must be in (0,1], got {sc}.")
    _optional_key(risk, "use_vol_targeting", (bool,), False)
    vt = _optional_key(risk, "vol_target_annual", (int, float), 0.18)
    _require_positive("risk.vol_target_annual", vt)

    # backtest
    rb = _optional_key(backtest, "rebalance", (str,), "quarterly")
    if rb.lower() not in ("quarterly", "q"):
        raise ConfigError("backtest.rebalance must be 'quarterly' for now.")
    cost_bps = _optional_key(backtest, "cost_bps_per_trade", (int, float), 20.0)
    slip_bps = _optional_key(backtest, "slippage_bps_per_trade", (int, float), 5.0)
    _require_nonnegative("backtest.cost_bps_per_trade", cost_bps)
    _require_nonnegative("backtest.slippage_bps_per_trade", slip_bps)

    # overlays (optional but validated if present)
    mode = str(cfg.get("overlay_mode", "A")).strip().upper()
    if mode not in {"A", "B", "C", "D"}:
        raise ConfigError(f"overlay_mode must be one of A,B,C,D; got '{mode}'.")

    overlays = cfg.get("overlays", {})
    if overlays is not None and not isinstance(overlays, dict):
        raise ConfigError("overlays must be a mapping if provided.")


# -------------------------
# Convenience getters
# -------------------------

def get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Fetch nested key via dot path, e.g. "fit.nrmse_max".
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """
    Get a top-level section safely (raises if missing).
    """
    return _require_section(cfg, key)
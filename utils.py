"""
scurve.core.utils

Shared deterministic utilities for Agora Lycos S-Curve model.

Goals:
- Small, dependency-light helpers used across the package.
- Deterministic run directory creation (no random IDs).
- Date parsing / formatting helpers.
- Numeric guards (finite checks, clipping).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import math


def utc_now_iso(timespec: str = "seconds") -> str:
    """UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def parse_asof(asof: str) -> str:
    """
    Validate and normalize an as-of date string (YYYY-MM-DD).
    Returns the same string if valid.

    Raises ValueError if invalid.
    """
    asof = str(asof).strip()
    # strict parse
    datetime.strptime(asof, "%Y-%m-%d")
    return asof


def asof_compact(asof: str) -> str:
    """YYYY-MM-DD -> YYYYMMDD."""
    return parse_asof(asof).replace("-", "")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_dir(cfg: Dict[str, Any], *, asof: str) -> Path:
    """
    Create a deterministic run directory for the given as-of date.

    Default:
      runs/YYYYMMDD/

    You can override the root via cfg['run']['output_root'].

    Returns Path to created run directory.
    """
    asof = parse_asof(asof)
    root = Path(cfg.get("run", {}).get("output_root", "runs"))
    run_dir = root / asof_compact(asof)
    ensure_dir(run_dir)
    return run_dir


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    x = float(x)
    return float(max(lo, min(hi, x)))


def is_finite(x: Any) -> bool:
    """True if x is a finite float/int."""
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def safe_float(x: Any, default: float = float("nan")) -> float:
    """
    Convert to float safely; returns default on failure.
    """
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_div(numer: float, denom: float, default: float = float("nan")) -> float:
    """
    Safe division with default if denom is zero or non-finite.
    """
    numer = float(numer)
    denom = float(denom)
    if denom == 0.0 or not math.isfinite(denom) or not math.isfinite(numer):
        return default
    return float(numer / denom)


def pct_change(curr: float, prev: float, default: float = float("nan")) -> float:
    """
    Percent change: (curr/prev - 1). Returns default if prev invalid.
    """
    curr = float(curr)
    prev = float(prev)
    if prev == 0.0 or not math.isfinite(prev) or not math.isfinite(curr):
        return default
    return float(curr / prev - 1.0)


def rolling_pairs(items: Iterable[Any]) -> Iterable[Tuple[Any, Any]]:
    """
    Yield consecutive pairs: (x0,x1), (x1,x2), ...
    """
    it = iter(items)
    try:
        prev = next(it)
    except StopIteration:
        return
    for x in it:
        yield prev, x
        prev = x


def stable_sort_keys(d: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Return sorted keys of a dict as a stable tuple.
    Useful for deterministic output ordering.
    """
    return tuple(sorted(map(str, d.keys())))


def assert_keys_present(d: Dict[str, Any], keys: Iterable[str], *, where: str = "") -> None:
    """
    Raise KeyError if any keys missing in dict.
    """
    missing = [k for k in keys if k not in d]
    if missing:
        loc = f" in {where}" if where else ""
        raise KeyError(f"Missing keys{loc}: {missing}")


def normalize_weights(weights: Dict[str, float], *, long_only: bool = True) -> Dict[str, float]:
    """
    Normalize a weights dict to sum to 1.0.

    - Drops None values.
    - If long_only=True, drops negative weights.
    - Drops zero weights.
    """
    w = {str(k): safe_float(v) for k, v in weights.items() if v is not None}
    w = {k: float(v) for k, v in w.items() if is_finite(v) and v != 0.0}
    if long_only:
        w = {k: v for k, v in w.items() if v > 0.0}
    s = sum(w.values())
    if s <= 0.0 or not math.isfinite(s):
        return {}
    return {k: v / s for k, v in w.items()}


def top_n_by_value(scores: Dict[str, float], n: int) -> Dict[str, float]:
    """
    Return top-n items by value (descending) as a new dict.
    Deterministic tie-breaking by key sort.
    """
    if n <= 0:
        return {}
    items = [(str(k), safe_float(v)) for k, v in scores.items()]
    items = [(k, v) for k, v in items if is_finite(v)]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return dict(items[:n])


def percentile_cutoff(values: Iterable[float], q: float) -> float:
    """
    Compute percentile cutoff (q in [0,1]) using a deterministic method.

    This is a tiny helper. For large-scale work you'll likely use numpy/pandas,
    but keeping it here avoids cross-module imports.
    """
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if is_finite(v)]
    if not vals:
        return float("nan")
    vals.sort()
    q = clamp(q, 0.0, 1.0)
    # nearest-rank method
    idx = int(math.ceil(q * len(vals))) - 1
    idx = int(clamp(idx, 0, len(vals) - 1))
    return float(vals[idx])
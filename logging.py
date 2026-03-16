"""
scurve.core.logging

Structured logging utilities for Agora Lycos S-Curve model.

Goals:
- Deterministic, simple, and robust.
- Log to both console and file.
- Include run metadata (asof, config hash, optional git hash).
- Avoid third-party logging frameworks.

Usage:
    from scurve.core.logging import init_logger, write_run_metadata

    logger = init_logger(cfg, run_dir)
    write_run_metadata(cfg, run_dir, asof="2026-03-31", logger=logger)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    # Stable JSON encoding (sorted keys) to produce deterministic hash
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(payload)


def _try_git_commit_hash(cwd: Path) -> Optional[str]:
    """
    Best-effort retrieval of current git commit hash.
    Returns None if not in a git repo or git is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def init_logger(cfg: Dict[str, Any], run_dir: str | Path) -> logging.Logger:
    """
    Initialize a logger that writes to:
      - console (INFO)
      - file logs/scurve_{asof_or_runid}.log (INFO)

    The filename is handled in make_log_path(). This function will create a
    default log file name "scurve_run.log" if not provided later.

    Returns a configured logger instance.
    """
    run_dir = Path(run_dir)
    log_root = Path(cfg.get("run", {}).get("log_root", "logs"))
    _ensure_dir(log_root)

    # Default log path; can be replaced/renamed by caller if desired
    log_path = log_root / "scurve_run.log"

    logger = logging.getLogger("scurve")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if init_logger called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("logger_initialized path=%s", str(log_path))
    return logger


def make_log_path(cfg: Dict[str, Any], asof: str, prefix: str = "scurve_") -> Path:
    """
    Construct a standard log path for a given as-of date.
    """
    log_root = Path(cfg.get("run", {}).get("log_root", "logs"))
    _ensure_dir(log_root)
    safe_asof = asof.replace("-", "")
    return log_root / f"{prefix}{safe_asof}.log"


def switch_file_handler(logger: logging.Logger, new_path: Path) -> None:
    """
    Replace existing FileHandler with a new file path.

    Useful when init_logger() is called before knowing --asof.
    """
    new_path = Path(new_path)
    _ensure_dir(new_path.parent)

    # Remove old file handlers
    kept = []
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
        else:
            kept.append(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(new_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("file_handler_switched path=%s", str(new_path))


def write_run_metadata(
    cfg: Dict[str, Any],
    run_dir: str | Path,
    *,
    asof: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Write a machine-readable run metadata JSON file into the run directory.

    Writes:
      runs/{asof}/run_metadata.json

    Includes:
      - asof
      - utc timestamp
      - config hash (sha256 of stable json)
      - optional git commit hash
      - python version and platform basics (minimal)
    """
    run_dir = Path(run_dir)
    _ensure_dir(run_dir)

    config_hash = _sha256_json(cfg)

    # Try to detect repo root as parent of run_dir (best effort)
    cwd = Path.cwd()
    git_hash = _try_git_commit_hash(cwd)

    meta = {
        "asof": asof,
        "utc_timestamp": _utc_now_iso(),
        "config_sha256": config_hash,
        "git_commit": git_hash,
        "cwd": str(cwd),
    }

    out_path = run_dir / "run_metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    if logger is not None:
        logger.info("run_metadata_written path=%s config_sha256=%s git_commit=%s", str(out_path), config_hash, git_hash)

    return meta


def log_kv(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """
    Lightweight structured log line as JSON payload, but still readable.

    Example:
        log_kv(logger, logging.INFO, "fit_summary", n=842, pass_rate=0.78)

    Produces:
        ... | INFO | scurve | {"event":"fit_summary","n":842,"pass_rate":0.78}
    """
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))
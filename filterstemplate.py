"""
filterstemplate.py
------------------
Reference-only helpers for universe and filter selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


ETF_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "GLD",
    "SLV",
    "EEM",
    "EFA",
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLRE",
    "XLK",
    "XLU",
    "SMH",
)


def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if t.startswith("$"):
        t = t[1:]
    return t.replace(".", "-")


def ask_use_filters() -> bool:
    ans = input("Apply filters (CapBucket / Sector)? (y/n): ").strip().lower()
    return ans in ("y", "yes")


def ask_cap_filter() -> Optional[set[str]]:
    print("\nMarket cap filter (CapBucket):")
    print("  1) All")
    print("  2) Large")
    print("  3) Mid")
    print("  4) Small")
    print("  5) Micro")
    print("  6) Large+Mid")
    print("  7) Mid+Small")
    choice = input("Choose 1-7: ").strip()

    mapping = {
        "1": None,
        "2": {"Large"},
        "3": {"Mid"},
        "4": {"Small"},
        "5": {"Micro"},
        "6": {"Large", "Mid"},
        "7": {"Mid", "Small"},
    }
    if choice not in mapping:
        raise ValueError("Invalid cap filter choice.")
    return mapping[choice]


def ask_sector_filter(available_sectors: list[str]) -> Optional[set[str]]:
    print("\nSector filter:")
    print("  1) All")
    print("  2) Choose from list")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        return None
    if choice != "2":
        raise ValueError("Invalid sector filter choice.")

    if not available_sectors:
        print("No sector data available in this file. Sector filter skipped.")
        return None

    print("\nAvailable sectors:")
    for i, s in enumerate(available_sectors, 1):
        print(f"  {i}) {s}")

    raw = input("Enter sector numbers separated by commas (e.g. 1,3,7): ").strip()
    idx = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx.append(int(part))

    picked = {available_sectors[i - 1] for i in idx if 1 <= i <= len(available_sectors)}
    return picked if picked else None


def load_tickers_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "Ticker" in df.columns:
        series = df["Ticker"]
    elif "Symbol" in df.columns:
        series = df["Symbol"]
    else:
        series = df.iloc[:, 0]

    tickers = (
        series.astype(str)
        .map(normalize_ticker)
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(set(tickers))


def load_master_with_filters(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        df["Ticker"] = df.iloc[:, 0]

    df["Ticker"] = df["Ticker"].astype(str).map(normalize_ticker)
    df = df.dropna(subset=["Ticker"])

    if not ask_use_filters():
        tickers = sorted(set(df["Ticker"].tolist()))
        print(f"\nLoaded {len(tickers)} tickers (no filters)\n")
        return tickers

    cap_choice = None
    sector_choice = None

    if "CapBucket" in df.columns:
        df["CapBucket"] = df["CapBucket"].fillna("Unknown")
        cap_choice = ask_cap_filter()
    else:
        print("\nCapBucket column not found. Cap filter skipped.")

    if "Sector" in df.columns:
        df["Sector"] = df["Sector"].fillna("Unknown")
        sectors = sorted([s for s in df["Sector"].unique().tolist() if s and s != "Unknown"])
        sector_choice = ask_sector_filter(sectors)
    else:
        print("\nSector column not found. Sector filter skipped.")

    if cap_choice is not None:
        df = df[df.get("CapBucket", "Unknown").isin(cap_choice)]
    if sector_choice is not None:
        df = df[df.get("Sector", "Unknown").isin(sector_choice)]

    tickers = sorted(set(df["Ticker"].tolist()))
    print(f"\nFiltered universe size: {len(tickers)}\n")
    return tickers


def ask_universe() -> Tuple[str, list[str]]:
    print("\nSelect universe to scan:")
    print("  1) S&P 500")
    print("  2) Nasdaq")
    print("  3) ETFs (predefined list)")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        universe_name = "S&P 500"
        master = Path("sp500_master.csv")
        fallback = Path("sp500_companies.csv")
    elif choice == "2":
        universe_name = "Nasdaq"
        master = Path("nasdaq_master.csv")
        fallback = Path("nasdaq_tickers.csv")
    elif choice == "3":
        universe_name = "ETFs"
        tickers = list(ETF_TICKERS)
        print(f"\nUsing predefined ETF list ({len(tickers)} tickers).\n")
        return universe_name, tickers
    else:
        raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

    if master.exists():
        print(f"\nUsing master file: {master.name}")
        tickers = load_master_with_filters(master)
    else:
        if not fallback.exists():
            raise FileNotFoundError(
                f"Missing both {master.resolve()} and {fallback.resolve()}.\n"
                f"Create the master file (recommended) or place the fallback CSV in this folder."
            )
        print(f"\nMaster file not found. Using fallback list: {fallback.name} (no cap/sector filters)")
        tickers = load_tickers_from_csv(fallback)
        print(f"Loaded {len(tickers)} tickers.\n")

    return universe_name, tickers

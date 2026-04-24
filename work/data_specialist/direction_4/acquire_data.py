"""
Download raw data for Direction 4: Dual-Mode Intraday Volume Forecast.

Downloads:
1. Intraday 1-minute bars (EQUS.MINI) for a pilot universe of ~35 stocks.
2. Daily bars (EQUS.MINI) for the same universe (for ADV, universe
   ranking, and half-day detection).

Date range: 2024-01-02 to 2025-12-31 (2 full calendar years).

The pilot universe includes all Dow 30 components plus additional
high-volume and mid-cap names to provide diversity for per-stock model
fitting.
"""

import json
import sys
import time
from pathlib import Path

import bentoticks as bt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "direction_4"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pilot universe: Dow 30 + 5 additional high-volume / mid-cap names
# Dow 30 components (as of early 2025)
DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA",
    "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW",
    "TRV", "UNH", "V", "VZ", "WMT",
]

# Additional stocks for diversity (high-volume tech + mid-cap)
EXTRA = ["GOOG", "META", "TSLA", "AMD", "INTC"]

TICKERS = sorted(set(DOW_30 + EXTRA))

START = "2024-01-02"
END = "2025-12-31"
DATASET_INTRADAY = "EQUS.MINI"
DATASET_DAILY = "EQUS.MINI"  # EQUS.SUMMARY preferred but using MINI for consistency


def download_intraday() -> dict:
    """
    Download 1-minute intraday bars for all tickers.

    Returns
    -------
    Per-ticker result dict with status, row count, date range, file size.
    """
    results = {}
    for ticker in TICKERS:
        out_path = DATA_DIR / f"{ticker}_1m.parquet"
        print(f"[intraday] Downloading {ticker} 1m bars {START} to {END}...")
        t0 = time.time()
        try:
            df = bt.load_bars_intraday(
                ticker, START, END, barsz="1m", dataset=DATASET_INTRADAY
            )
            df.to_parquet(out_path)
            elapsed = time.time() - t0
            n_days = (
                df["date_event"].nunique()
                if "date_event" in df.columns
                else "N/A"
            )
            date_min = (
                str(df["date_event"].min())
                if "date_event" in df.columns
                else "N/A"
            )
            date_max = (
                str(df["date_event"].max())
                if "date_event" in df.columns
                else "N/A"
            )
            size_mb = round(out_path.stat().st_size / 1e6, 2)
            print(
                f"  -> {len(df)} rows, {n_days} days, "
                f"{size_mb} MB, {elapsed:.1f}s"
            )
            results[ticker] = {
                "status": "acquired",
                "rows": len(df),
                "days": int(n_days) if isinstance(n_days, (int, float)) else n_days,
                "size_mb": size_mb,
                "path": str(out_path.relative_to(DATA_DIR.parents[1])),
                "date_min": date_min,
                "date_max": date_max,
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> FAILED after {elapsed:.1f}s: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}
    return results


def download_daily() -> dict:
    """
    Download daily bars for all tickers.

    Returns
    -------
    Per-ticker result dict with status, row count, date range, file size.
    """
    results = {}
    for ticker in TICKERS:
        out_path = DATA_DIR / f"{ticker}_1d.parquet"
        print(f"[daily] Downloading {ticker} daily bars {START} to {END}...")
        t0 = time.time()
        try:
            df = bt.load_bars_daily(ticker, START, END, dataset=DATASET_DAILY)
            df.to_parquet(out_path)
            elapsed = time.time() - t0
            date_min = (
                str(df["date_event"].min())
                if "date_event" in df.columns
                else "N/A"
            )
            date_max = (
                str(df["date_event"].max())
                if "date_event" in df.columns
                else "N/A"
            )
            size_mb = round(out_path.stat().st_size / 1e6, 3)
            print(f"  -> {len(df)} rows, {size_mb} MB, {elapsed:.1f}s")
            results[ticker] = {
                "status": "acquired",
                "rows": len(df),
                "size_mb": size_mb,
                "path": str(out_path.relative_to(DATA_DIR.parents[1])),
                "date_min": date_min,
                "date_max": date_max,
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> FAILED after {elapsed:.1f}s: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Direction 4 — Raw Data Acquisition")
    print(f"Tickers: {len(TICKERS)} stocks")
    print(f"Date range: {START} to {END}")
    print(f"Dataset (intraday): {DATASET_INTRADAY}")
    print(f"Dataset (daily): {DATASET_DAILY}")
    print("=" * 60)

    print("\n--- Intraday 1-minute bars ---")
    intraday_results = download_intraday()

    print("\n--- Daily bars ---")
    daily_results = download_daily()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for label, res in [("Intraday", intraday_results), ("Daily", daily_results)]:
        acquired = sum(1 for v in res.values() if v["status"] == "acquired")
        failed = sum(1 for v in res.values() if v["status"] == "failed")
        print(f"{label}: {acquired} acquired, {failed} failed")
        for ticker, info in res.items():
            if info["status"] == "acquired":
                print(
                    f"  {ticker}: {info['rows']} rows, "
                    f"{info['size_mb']} MB, "
                    f"{info['date_min']} to {info['date_max']}"
                )
            else:
                print(f"  {ticker}: FAILED - {info['error']}")

    summary = {"intraday": intraday_results, "daily": daily_results}
    summary_path = DATA_DIR / "acquisition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

"""
Download raw data for Direction 7: Kalman Filter State-Space Model.

Downloads intraday 1-minute bars and daily bars from Databento via
bentoticks for 10 securities.

Date range: 2023-10-02 to 2026-03-31 (substitute for paper's
2014-01-02 to 2016-06-30, which predates Databento coverage).
"""

import json
import sys
import time
from pathlib import Path

import bentoticks as bt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "direction_7"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "DIA", "QQQ", "AAPL", "AMZN", "GOOG", "IBM", "JPM", "MSFT", "XOM"]
START = "2023-10-02"
END = "2026-03-31"
DATASET = "EQUS.MINI"


def download_intraday() -> dict:
    """Download 1-minute intraday bars for all tickers."""
    results = {}
    for ticker in TICKERS:
        out_path = DATA_DIR / f"{ticker}_1m.parquet"
        print(f"[intraday] Downloading {ticker} 1m bars {START} to {END}...")
        t0 = time.time()
        try:
            df = bt.load_bars_intraday(
                ticker, START, END, barsz="1m", dataset=DATASET
            )
            df.to_parquet(out_path)
            elapsed = time.time() - t0
            n_days = df["date_event"].nunique() if "date_event" in df.columns else "N/A"
            date_min = str(df["date_event"].min()) if "date_event" in df.columns else "N/A"
            date_max = str(df["date_event"].max()) if "date_event" in df.columns else "N/A"
            size_mb = round(out_path.stat().st_size / 1e6, 2)
            print(f"  -> {len(df)} rows, {n_days} days, {size_mb} MB, {elapsed:.1f}s")
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
    """Download daily bars for all tickers."""
    results = {}
    for ticker in TICKERS:
        out_path = DATA_DIR / f"{ticker}_1d.parquet"
        print(f"[daily] Downloading {ticker} daily bars {START} to {END}...")
        t0 = time.time()
        try:
            df = bt.load_bars_daily(ticker, START, END, dataset=DATASET)
            df.to_parquet(out_path)
            elapsed = time.time() - t0
            date_min = str(df["date_event"].min()) if "date_event" in df.columns else "N/A"
            date_max = str(df["date_event"].max()) if "date_event" in df.columns else "N/A"
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
    print("Direction 7 — Raw Data Acquisition")
    print(f"Date range: {START} to {END}")
    print(f"Dataset: {DATASET}")
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
                print(f"  {ticker}: {info['rows']} rows, {info['size_mb']} MB, {info['date_min']} to {info['date_max']}")
            else:
                print(f"  {ticker}: FAILED - {info['error']}")

    summary = {"intraday": intraday_results, "daily": daily_results}
    summary_path = DATA_DIR / "acquisition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

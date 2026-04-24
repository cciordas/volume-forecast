"""
Download 1-minute OHLCV bars for DJIA components from Databento EQUS.MINI.

Acquires data for Direction 2 (PCA Factor Decomposition / BDF) covering
2023-01-01 to 2025-12-31. Saves per-ticker Parquet files to data/direction_2/.
"""

import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# bentoticks is available in the finance conda environment
import bentoticks as bt

# Current DJIA components (30 stocks)
DJIA_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA",
    "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW",
    "TRV", "UNH", "V", "VZ", "WMT",
]

START_DATE = "2023-03-28"  # EQUS.MINI available from 2023-03-28
END_DATE = "2025-12-31"
DATASET = "EQUS.MINI"
BARSZ = "1m"

# Output directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "direction_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_ticker(ticker: str) -> dict:
    """
    Download 1-minute bars for a single ticker and save to Parquet.

    Parameters
    ----------
    ticker: Stock symbol.

    Returns
    -------
    Dict with status info for the acquisition report.
    """
    out_path = OUT_DIR / f"{ticker}_1m.parquet"
    result = {
        "ticker": ticker,
        "status": "failed",
        "path": str(out_path),
        "rows": 0,
        "date_min": None,
        "date_max": None,
        "size_mb": 0.0,
        "error": None,
    }

    try:
        print(f"[{ticker}] Downloading {BARSZ} bars from {START_DATE} to {END_DATE} ...")
        df = bt.load_bars_intraday(
            ticker,
            START_DATE,
            END_DATE,
            barsz=BARSZ,
            dataset=DATASET,
        )

        if df is None or len(df) == 0:
            result["error"] = "Empty DataFrame returned"
            print(f"[{ticker}] WARNING: empty result")
            return result

        # Save to Parquet
        df.to_parquet(out_path)

        result["status"] = "acquired"
        result["rows"] = len(df)
        result["size_mb"] = round(out_path.stat().st_size / (1024 * 1024), 2)

        # Date range from date_event column
        if "date_event" in df.columns:
            result["date_min"] = str(df["date_event"].min())
            result["date_max"] = str(df["date_event"].max())
        elif isinstance(df.index, pd.DatetimeIndex):
            result["date_min"] = str(df.index.min().date())
            result["date_max"] = str(df.index.max().date())

        print(f"[{ticker}] OK — {result['rows']:,} rows, "
              f"{result['date_min']} to {result['date_max']}, "
              f"{result['size_mb']} MB")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[{ticker}] FAILED — {result['error']}")
        traceback.print_exc()

    return result


def main() -> None:
    """
    Download all tickers and print summary.
    """
    print(f"Acquiring {BARSZ} OHLCV bars for {len(DJIA_TICKERS)} DJIA stocks")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Dataset: {DATASET}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    results = []
    for ticker in DJIA_TICKERS:
        r = download_ticker(ticker)
        results.append(r)

    # Summary
    acquired = [r for r in results if r["status"] == "acquired"]
    failed = [r for r in results if r["status"] == "failed"]

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(acquired)} acquired, {len(failed)} failed "
          f"out of {len(results)} total")

    if failed:
        print("\nFailed tickers:")
        for r in failed:
            print(f"  {r['ticker']}: {r['error']}")

    # Save results summary as JSON for report generation
    import json
    summary_path = OUT_DIR / "acquisition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

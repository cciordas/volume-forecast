"""
Prepare raw 1-minute OHLCV data into 15-minute bin volume arrays for Direction 4.

Transformations:
  1. Aggregate 1-minute bars into 26 x 15-minute bins per trading day (9:30-16:00 ET).
  2. Apply split adjustments for NVDA (10:1, 2024-06-10) and WMT (3:1, 2024-02-26).
  3. Exclude half-day trading sessions (< 300 1-minute bars).
  4. Build per-ticker volume_history arrays of shape (n_days, 26).
  5. Produce a combined volume_history dict and metadata file.

Output:
  data/direction_4/prepared/{TICKER}_15m_volume.parquet  — per-ticker (n_days x 26) volume
  data/direction_4/prepared/volume_history.parquet        — stacked panel (ticker, date, bin)
  data/direction_4/prepared/daily_stats.parquet           — daily OHLCV with dollar volume
  data/direction_4/prepared/metadata.json                 — tickers, dates, splits, exclusions
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/direction_4")
PREPARED_DIR = Path("data/direction_4/prepared")
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "AMD", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GOOG", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO",
    "MCD", "META", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV",
    "TSLA", "UNH", "V", "VZ", "WMT",
]

# 26 bins of 15 minutes each: 9:30-9:45, 9:45-10:00, ..., 15:45-16:00
I = 26
BIN_START_MINUTES = [570 + 15 * i for i in range(I)]  # minutes since midnight
# 570 = 9*60 + 30

# Known stock splits in the data window (pre-split -> post-split)
# Volume before split date must be divided by the split ratio to make it
# comparable to post-split volume.
SPLITS = {
    "NVDA": {"date": datetime.date(2024, 6, 10), "ratio": 10},
    "WMT":  {"date": datetime.date(2024, 2, 26), "ratio": 3},
}

# Half-day detection: a session ending before 14:30 ET (minute 870) is a half-day.
# Normal close is 16:00 (minute 960). Half-days close at 13:00 (minute 780).
# We use 14:30 as a generous cutoff to catch any half-day variant.
HALF_DAY_LAST_BAR_CUTOFF = datetime.time(14, 30)


def minute_of_day(t):
    """
    Convert a datetime.time to minutes since midnight.

    Parameters
    ----------
    t: Time value to convert.

    Returns
    -------
    Integer minutes since midnight.
    """
    return t.hour * 60 + t.minute


def assign_bin(minute):
    """
    Map a minute-of-day value to a 15-minute bin index (0-based, 0..25).

    Parameters
    ----------
    minute: Minutes since midnight for the bar start time.

    Returns
    -------
    Bin index (0-25), or -1 if outside regular trading hours.
    """
    if minute < 570 or minute >= 960:  # before 9:30 or at/after 16:00
        return -1
    return (minute - 570) // 15


def aggregate_1m_to_15m(df_1m, ticker):
    """
    Aggregate 1-minute bars to 15-minute bin volumes for one ticker.

    Parameters
    ----------
    df_1m:   Raw 1-minute dataframe with columns [date_event, ts_event, volume].
    ticker:  Ticker symbol (for split adjustment lookup).

    Returns
    -------
    DataFrame with columns [date, bin_1, bin_2, ..., bin_26] where each bin
    column contains the total volume for that 15-minute interval. Only
    full trading days (>= HALF_DAY_BAR_THRESHOLD bars) are included.
    """
    df = df_1m.copy()

    # Compute minute of day and bin assignment
    df["minute"] = df["ts_event"].apply(minute_of_day)
    df["bin"] = df["minute"].apply(assign_bin)

    # Drop bars outside RTH (bin == -1)
    df = df[df["bin"] >= 0].copy()

    # Identify half-days by last bar time (half-days end before 14:30)
    last_bar_per_day = df.groupby("date_event")["ts_event"].max()
    half_days = set(last_bar_per_day[last_bar_per_day < HALF_DAY_LAST_BAR_CUTOFF].index)

    # Exclude half-days
    df = df[~df["date_event"].isin(half_days)].copy()

    # Apply split adjustment: divide pre-split volume by ratio
    # Cast volume to float64 first to avoid incompatible dtype warning
    if ticker in SPLITS:
        split_info = SPLITS[ticker]
        df["volume"] = df["volume"].astype(np.float64)
        pre_split_mask = df["date_event"] < split_info["date"]
        df.loc[pre_split_mask, "volume"] = (
            df.loc[pre_split_mask, "volume"] / split_info["ratio"]
        )

    # Aggregate volume per (date, bin)
    agg = df.groupby(["date_event", "bin"])["volume"].sum().reset_index()

    # Pivot to wide format: one row per date, 26 bin columns
    pivot = agg.pivot(index="date_event", columns="bin", values="volume")

    # Ensure all 26 bins are present (fill missing with 0)
    for b in range(I):
        if b not in pivot.columns:
            pivot[b] = 0
    pivot = pivot[list(range(I))].fillna(0)

    # Rename columns to bin_1 .. bin_26 (1-indexed as per impl spec)
    pivot.columns = [f"bin_{i+1}" for i in range(I)]
    pivot.index.name = "date"
    pivot = pivot.sort_index()

    return pivot, half_days


def prepare_daily_stats(ticker):
    """
    Load daily OHLCV and compute dollar volume for universe ranking.

    Parameters
    ----------
    ticker: Ticker symbol.

    Returns
    -------
    DataFrame with columns [date, open, high, low, close, volume, dollar_volume].
    """
    df = pd.read_parquet(RAW_DIR / f"{ticker}_1d.parquet")
    df = df.rename(columns={"date_event": "date"})

    # Apply split adjustment to price and volume for consistency
    if ticker in SPLITS:
        split_info = SPLITS[ticker]
        pre_mask = df["date"] < split_info["date"]
        ratio = split_info["ratio"]
        df.loc[pre_mask, "open"] = df.loc[pre_mask, "open"] / ratio
        df.loc[pre_mask, "high"] = df.loc[pre_mask, "high"] / ratio
        df.loc[pre_mask, "low"] = df.loc[pre_mask, "low"] / ratio
        df.loc[pre_mask, "close"] = df.loc[pre_mask, "close"] / ratio
        df.loc[pre_mask, "volume"] = df.loc[pre_mask, "volume"] * ratio

    df["dollar_volume"] = df["close"] * df["volume"]
    df["ticker"] = ticker
    return df


def main():
    """
    Run the full data preparation pipeline.
    """
    all_volumes = {}
    all_daily = []
    all_half_days = {}
    metadata = {
        "tickers": TICKERS,
        "bins_per_day": I,
        "bin_width_minutes": 15,
        "trading_hours": "09:30-16:00 ET",
        "splits_applied": {},
        "half_days_excluded": {},
        "dates": {},
    }

    print(f"Preparing data for {len(TICKERS)} tickers...")
    print(f"Output directory: {PREPARED_DIR}")
    print()

    for ticker in TICKERS:
        print(f"Processing {ticker}...", end=" ")
        sys.stdout.flush()

        # --- 15-minute volume bins ---
        df_1m = pd.read_parquet(RAW_DIR / f"{ticker}_1m.parquet")
        vol_15m, half_days = aggregate_1m_to_15m(df_1m, ticker)

        # Save per-ticker
        out_path = PREPARED_DIR / f"{ticker}_15m_volume.parquet"
        vol_15m.to_parquet(out_path)
        all_volumes[ticker] = vol_15m
        all_half_days[ticker] = [str(d) for d in sorted(half_days)]

        # --- Daily stats ---
        daily = prepare_daily_stats(ticker)
        all_daily.append(daily)

        # Log split info
        if ticker in SPLITS:
            s = SPLITS[ticker]
            metadata["splits_applied"][ticker] = {
                "date": str(s["date"]),
                "ratio": s["ratio"],
                "description": f"{s['ratio']}:1 split",
            }

        metadata["half_days_excluded"][ticker] = all_half_days[ticker]
        metadata["dates"][ticker] = {
            "first": str(vol_15m.index.min()),
            "last": str(vol_15m.index.max()),
            "n_days": len(vol_15m),
        }

        print(
            f"{len(vol_15m)} days, "
            f"{len(half_days)} half-days excluded, "
            f"volume range [{vol_15m.values.min():.0f}, {vol_15m.values.max():.0f}]"
        )

    # --- Build stacked panel: (ticker, date, bin_1..bin_26) ---
    print("\nBuilding stacked volume panel...")
    panels = []
    for ticker, vol_df in all_volumes.items():
        tmp = vol_df.copy()
        tmp["ticker"] = ticker
        tmp = tmp.reset_index()
        panels.append(tmp)

    volume_panel = pd.concat(panels, ignore_index=True)
    volume_panel = volume_panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    volume_panel.to_parquet(PREPARED_DIR / "volume_history.parquet")

    # --- Combined daily stats ---
    print("Building combined daily stats...")
    daily_all = pd.concat(all_daily, ignore_index=True)
    daily_all = daily_all.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily_all.to_parquet(PREPARED_DIR / "daily_stats.parquet")

    # --- Metadata ---
    metadata["total_tickers"] = len(TICKERS)
    metadata["total_trading_days"] = int(volume_panel.groupby("ticker")["date"].count().median())

    with open(PREPARED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # --- Summary ---
    print("\n=== Preparation Summary ===")
    print(f"Tickers processed: {len(TICKERS)}")
    print(f"Volume panel shape: {volume_panel.shape}")
    print(f"Daily stats shape: {daily_all.shape}")
    print(f"Trading days (median across tickers): {metadata['total_trading_days']}")
    print(f"Splits applied: {list(metadata['splits_applied'].keys()) or 'None'}")
    n_half = sum(len(v) for v in all_half_days.values())
    print(f"Half-day sessions excluded: {n_half // len(TICKERS)} unique dates")
    print(f"\nOutput files:")
    for p in sorted(PREPARED_DIR.glob("*")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()

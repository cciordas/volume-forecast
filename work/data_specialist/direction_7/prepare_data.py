"""
Prepare raw 1-minute and daily bar data for the Kalman filter intraday volume model.

Transformations:
1. Identify and exclude half-day sessions (< 390 1-min bars in regular hours).
2. Aggregate 1-minute bars to 15-minute bins (I=26 per full session).
3. Compute rolling 60-day average daily volume (ADV-60) as normalization factor.
4. Normalize bin volume by ADV-60.
5. Log-transform normalized volume; mark zero-volume bins as missing.
6. Build per-ticker volume matrices y[t, i] and a combined panel.
7. Derive and save trading calendar.
8. Save daily OHLCV cross-check dataset.

Output files are written to data/direction_7/prepared/.
"""

import json
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "direction_7"
PREP_DIR = PROJECT_ROOT / "data" / "direction_7" / "prepared"
PREP_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "DIA", "QQQ", "AAPL", "AMZN", "GOOG", "IBM", "JPM", "MSFT", "XOM"]

# 15-minute bins covering the regular NYSE session 09:30-16:00
# Bin i spans [09:30 + 15*i, 09:30 + 15*(i+1)), i = 0..25
BINS_PER_DAY = 26
BIN_WIDTH_MIN = 15
SESSION_START = time(9, 30)
SESSION_END = time(16, 0)
FULL_DAY_1M_BARS = 390  # 6.5 hours * 60 min

# ADV lookback for normalization (workaround for missing shares outstanding)
ADV_LOOKBACK = 60

# Threshold: if a day has fewer than this many 1-min bars in regular hours,
# treat it as a half-day session and exclude it.
# We use 385 to allow for a few missing bars on otherwise full days.
HALF_DAY_THRESHOLD = 385


def load_1m(ticker: str) -> pd.DataFrame:
    """
    Load raw 1-minute bar data for a ticker.

    Parameters
    ----------
    ticker: Ticker symbol.

    Returns
    -------
    DataFrame with date_event (date), ts_event (time), and volume columns,
    filtered to regular session hours only.
    """
    path = RAW_DIR / f"{ticker}_1m.parquet"
    df = pd.read_parquet(path, columns=["date_event", "ts_event", "volume"])

    # Ensure correct types
    if not isinstance(df["date_event"].iloc[0], pd.Timestamp):
        df["date_event"] = pd.to_datetime(df["date_event"])
    df["date_event"] = df["date_event"].dt.date

    if isinstance(df["ts_event"].iloc[0], str):
        df["ts_event"] = pd.to_timedelta(df["ts_event"])
    elif hasattr(df["ts_event"].iloc[0], "hour"):
        # Already time objects -- convert to timedelta for arithmetic
        df["ts_event"] = pd.to_timedelta(df["ts_event"].astype(str))

    # Filter to regular session: 09:30:00 <= ts_event < 16:00:00
    session_start_td = pd.Timedelta(hours=9, minutes=30)
    session_end_td = pd.Timedelta(hours=16)
    mask = (df["ts_event"] >= session_start_td) & (df["ts_event"] < session_end_td)
    df = df[mask].copy()

    return df


def load_daily(ticker: str) -> pd.DataFrame:
    """
    Load raw daily bar data for a ticker.

    Parameters
    ----------
    ticker: Ticker symbol.

    Returns
    -------
    DataFrame with date_event (date) and volume columns.
    """
    path = RAW_DIR / f"{ticker}_1d.parquet"
    df = pd.read_parquet(path, columns=["date_event", "volume", "close"])
    if not isinstance(df["date_event"].iloc[0], pd.Timestamp):
        df["date_event"] = pd.to_datetime(df["date_event"])
    df["date_event"] = df["date_event"].dt.date
    return df


def identify_half_days_from_spy(spy_1m: pd.DataFrame) -> tuple[set, set]:
    """
    Identify half-day sessions using SPY (most liquid ticker).

    Half-day sessions are detected by checking whether SPY has significantly
    fewer than 390 regular-session bars. This is more reliable than checking
    per-ticker bar counts, because less liquid tickers (DIA, IBM) routinely
    have missing 1-minute bars even on full trading days.

    Parameters
    ----------
    spy_1m: SPY 1-minute bar DataFrame with date_event and ts_event columns.

    Returns
    -------
    Tuple of (full_days, half_days) as sets of date objects.
    """
    bars_per_day = spy_1m.groupby("date_event").size()
    all_days = set(bars_per_day.index)
    half_days = set(bars_per_day[bars_per_day < HALF_DAY_THRESHOLD].index)
    full_days = all_days - half_days
    return full_days, half_days


def aggregate_to_15min(df_1m: pd.DataFrame, full_days: set) -> pd.DataFrame:
    """
    Aggregate 1-minute bars to 15-minute bins for full days only.

    Creates a complete grid of 26 bins per day. For less liquid tickers
    where some 1-minute bars are missing, the sum of available bars within
    each 15-minute window is used (which may be zero if all bars are missing).

    Parameters
    ----------
    df_1m:      1-minute bar DataFrame.
    full_days:  Set of dates to include (full trading days).

    Returns
    -------
    DataFrame with columns [date, bin_idx, volume] where bin_idx is 0..25.
    """
    # Filter to full days
    df = df_1m[df_1m["date_event"].isin(full_days)].copy()

    # Compute bin index from time: bin_idx = (minutes_since_930) // 15
    session_start_td = pd.Timedelta(hours=9, minutes=30)
    minutes_since_open = (df["ts_event"] - session_start_td).dt.total_seconds() / 60
    df["bin_idx"] = (minutes_since_open // BIN_WIDTH_MIN).astype(int)

    # Only keep bins 0..25 (regular session)
    df = df[(df["bin_idx"] >= 0) & (df["bin_idx"] < BINS_PER_DAY)]

    # Sum volume per (date, bin)
    agg = df.groupby(["date_event", "bin_idx"])["volume"].sum().reset_index()
    agg.rename(columns={"date_event": "date"}, inplace=True)

    # Create complete grid to ensure all 26 bins present for every day
    all_dates = sorted(full_days)
    grid = pd.DataFrame(
        [(d, i) for d in all_dates for i in range(BINS_PER_DAY)],
        columns=["date", "bin_idx"],
    )
    result = grid.merge(agg, on=["date", "bin_idx"], how="left")
    result["volume"] = result["volume"].fillna(0).astype(np.uint64)

    return result


def compute_adv(df_15min: pd.DataFrame, lookback: int = ADV_LOOKBACK) -> pd.DataFrame:
    """
    Compute rolling average daily volume (ADV) from 15-minute bin data.

    ADV for day t is the mean of total daily volume over the preceding
    `lookback` trading days (not including day t itself).

    Parameters
    ----------
    df_15min: 15-minute bin DataFrame with [date, bin_idx, volume].
    lookback: Number of trailing days for the rolling average.

    Returns
    -------
    DataFrame with columns [date, adv] where adv is the trailing average
    daily volume. Days without enough history get NaN.
    """
    daily_vol = df_15min.groupby("date")["volume"].sum().reset_index()
    daily_vol.rename(columns={"volume": "daily_volume"}, inplace=True)
    daily_vol = daily_vol.sort_values("date").reset_index(drop=True)

    # Rolling mean of the *previous* lookback days (shift by 1 to exclude current day)
    daily_vol["adv"] = (
        daily_vol["daily_volume"]
        .shift(1)
        .rolling(window=lookback, min_periods=lookback)
        .mean()
    )

    return daily_vol[["date", "adv"]]


def prepare_ticker(ticker: str, common_full_days: set) -> dict:
    """
    Prepare all datasets for a single ticker.

    Parameters
    ----------
    ticker:           Ticker symbol.
    common_full_days: Set of dates that are full days across all tickers.

    Returns
    -------
    Dict with summary statistics for the preparation report.
    """
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")

    # Load and inspect
    df_1m = load_1m(ticker)
    excluded_days = sorted(set(df_1m["date_event"].unique()) - common_full_days)

    print(f"  Raw 1m bars: {len(df_1m):,}")
    print(f"  Full days used: {len(common_full_days)}")
    print(f"  Excluded days: {len(excluded_days)}")

    # Aggregate to 15-minute bins (using common full days)
    df_15min = aggregate_to_15min(df_1m, common_full_days)
    print(f"  15-min bins: {len(df_15min):,} ({len(df_15min) // BINS_PER_DAY} days x {BINS_PER_DAY} bins)")

    # Compute ADV
    adv_df = compute_adv(df_15min)
    df_15min = df_15min.merge(adv_df, on="date", how="left")

    # Days with valid ADV (enough history)
    valid_adv_mask = df_15min["adv"].notna()
    n_days_no_adv = df_15min.loc[~valid_adv_mask, "date"].nunique()
    print(f"  Days dropped (no ADV-{ADV_LOOKBACK} history): {n_days_no_adv}")

    # Keep only rows with valid ADV
    df_valid = df_15min[valid_adv_mask].copy()

    # Normalize volume
    df_valid["vol_norm"] = df_valid["volume"] / df_valid["adv"]

    # Mark zero-volume bins
    df_valid["is_observed"] = df_valid["volume"] > 0

    # Log-transform (observed bins only; set missing to NaN)
    df_valid["log_vol"] = np.where(
        df_valid["is_observed"],
        np.log(df_valid["vol_norm"].clip(lower=1e-20)),  # clip to avoid log(0)
        np.nan,
    )

    zero_bins = (~df_valid["is_observed"]).sum()
    total_bins = len(df_valid)
    if total_bins > 0:
        print(f"  Zero-volume bins: {zero_bins} ({100*zero_bins/total_bins:.2f}%)")
    else:
        print(f"  WARNING: no valid bins after ADV filtering!")
        return {
            "ticker": ticker, "raw_1m_bars": len(df_1m), "n_days": 0,
            "date_range": "N/A", "shape": "0 x 0", "zero_bins": 0,
            "zero_pct": 0.0, "days_dropped_no_adv": n_days_no_adv,
            "excluded_half_days": len(excluded_days),
        }

    # Build the y[t, i] matrix (days x bins)
    dates_sorted = sorted(df_valid["date"].unique())
    n_days = len(dates_sorted)

    y_matrix = df_valid.pivot(index="date", columns="bin_idx", values="log_vol")
    y_matrix = y_matrix.sort_index()
    y_matrix.columns = [f"bin_{i:02d}" for i in range(BINS_PER_DAY)]

    observed_matrix = df_valid.pivot(index="date", columns="bin_idx", values="is_observed")
    observed_matrix = observed_matrix.sort_index()
    observed_matrix.columns = [f"bin_{i:02d}" for i in range(BINS_PER_DAY)]

    raw_vol_matrix = df_valid.pivot(index="date", columns="bin_idx", values="volume")
    raw_vol_matrix = raw_vol_matrix.sort_index().astype(np.float64)
    raw_vol_matrix.columns = [f"bin_{i:02d}" for i in range(BINS_PER_DAY)]

    print(f"  Output matrix shape: {y_matrix.shape} (days x bins)")
    print(f"  Date range: {dates_sorted[0]} to {dates_sorted[-1]}")

    # Save per-ticker datasets
    y_matrix.to_parquet(PREP_DIR / f"{ticker}_log_volume.parquet")
    observed_matrix.to_parquet(PREP_DIR / f"{ticker}_observed.parquet")
    raw_vol_matrix.to_parquet(PREP_DIR / f"{ticker}_raw_volume_15min.parquet")

    # Save the long-form data too (useful for debugging)
    df_valid_out = df_valid[["date", "bin_idx", "volume", "adv", "vol_norm", "is_observed", "log_vol"]].copy()
    df_valid_out.to_parquet(PREP_DIR / f"{ticker}_long.parquet", index=False)

    return {
        "ticker": ticker,
        "raw_1m_bars": len(df_1m),
        "n_days": n_days,
        "date_range": f"{dates_sorted[0]} to {dates_sorted[-1]}",
        "shape": f"{y_matrix.shape[0]} x {y_matrix.shape[1]}",
        "zero_bins": int(zero_bins),
        "zero_pct": round(100 * zero_bins / total_bins, 2),
        "days_dropped_no_adv": n_days_no_adv,
        "excluded_half_days": len(excluded_days),
    }


def prepare_trading_calendar(
    all_tickers_1m: dict, common_full_days: set, valid_dates: list
) -> pd.DataFrame:
    """
    Derive and save the trading calendar.

    Parameters
    ----------
    all_tickers_1m:  Dict mapping ticker to raw 1-minute DataFrame.
    common_full_days: Set of full trading days common to all tickers.
    valid_dates:     List of dates that passed ADV filtering.

    Returns
    -------
    Calendar DataFrame.
    """
    # Use SPY to get all trading days
    spy_dates = sorted(all_tickers_1m["SPY"]["date_event"].unique())
    spy_bars_per_day = all_tickers_1m["SPY"].groupby("date_event").size()

    cal = pd.DataFrame({"date": spy_dates})
    cal["bars_1m"] = cal["date"].map(spy_bars_per_day).fillna(0).astype(int)
    cal["is_full_day"] = cal["date"].isin(common_full_days)
    cal["is_half_day"] = (~cal["is_full_day"]) & (cal["bars_1m"] > 0)
    cal["in_prepared_data"] = cal["date"].isin(valid_dates)

    cal.to_parquet(PREP_DIR / "trading_calendar.parquet", index=False)
    return cal


def prepare_daily_crosscheck() -> pd.DataFrame:
    """
    Save combined daily OHLCV data for validation cross-checks.

    Returns
    -------
    Combined daily DataFrame.
    """
    frames = []
    for ticker in TICKERS:
        df = load_daily(ticker)
        df["ticker"] = ticker
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(PREP_DIR / "daily_ohlcv.parquet", index=False)
    return combined


def main() -> None:
    """
    Run all data preparation steps and print summary.
    """
    print("=" * 60)
    print("Data Preparation: Direction 7 — Kalman Filter Intraday Volume")
    print("=" * 60)

    # Step 1: Identify full days using SPY as reference
    # SPY is the most liquid ticker; missing bars for DIA/IBM reflect low
    # liquidity, not early closes. Only SPY-detected short days are excluded.
    print("\n--- Identifying full trading days (via SPY) ---")
    all_tickers_1m = {}
    for ticker in TICKERS:
        all_tickers_1m[ticker] = load_1m(ticker)
        print(f"  {ticker}: {all_tickers_1m[ticker]['date_event'].nunique()} trading days, {len(all_tickers_1m[ticker]):,} bars")

    full_days, half_days = identify_half_days_from_spy(all_tickers_1m["SPY"])
    print(f"\n  Full trading days (from SPY): {len(full_days)}")
    print(f"  Half-day sessions excluded: {len(half_days)}")
    for d in sorted(half_days):
        spy_count = all_tickers_1m["SPY"].groupby("date_event").size().get(d, 0)
        print(f"    {d}: SPY has {spy_count} bars")
    common_full_days = full_days

    # Step 2: Process each ticker
    summaries = []
    for ticker in TICKERS:
        summary = prepare_ticker(ticker, common_full_days)
        summaries.append(summary)

    # Step 3: Trading calendar
    print("\n--- Preparing trading calendar ---")
    # Get valid dates from the first ticker (all tickers share the same date grid)
    first_ticker_log = pd.read_parquet(PREP_DIR / f"{TICKERS[0]}_log_volume.parquet")
    valid_dates = list(first_ticker_log.index)
    cal = prepare_trading_calendar(all_tickers_1m, common_full_days, valid_dates)
    print(f"  Total trading days: {len(cal)}")
    print(f"  Full days: {cal['is_full_day'].sum()}")
    print(f"  Half days: {cal['is_half_day'].sum()}")
    print(f"  In prepared data: {cal['in_prepared_data'].sum()}")

    # Step 4: Daily cross-check
    print("\n--- Preparing daily OHLCV cross-check ---")
    daily = prepare_daily_crosscheck()
    print(f"  Combined daily rows: {len(daily)}")

    # Step 5: Save metadata
    metadata = {
        "bins_per_day": BINS_PER_DAY,
        "bin_width_minutes": BIN_WIDTH_MIN,
        "session_start": "09:30",
        "session_end": "16:00",
        "adv_lookback": ADV_LOOKBACK,
        "normalization": "ADV-60 (rolling 60-day average daily volume, lagged by 1 day)",
        "log_transform": "natural log of (volume / ADV-60)",
        "half_day_threshold": HALF_DAY_THRESHOLD,
        "tickers": TICKERS,
        "n_common_full_days": len(common_full_days),
        "excluded_dates": [str(d) for d in sorted(half_days)],
        "date_range_paper": "2014-01-02 to 2016-06-30",
        "date_range_substitute": f"{min(common_full_days)} to {max(common_full_days)}",
        "ticker_summaries": summaries,
    }
    with open(PREP_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in summaries:
        print(f"  {s['ticker']}: {s['shape']}, zero bins: {s['zero_bins']} ({s['zero_pct']}%)")
    print(f"\n  Output directory: {PREP_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()

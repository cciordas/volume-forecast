"""
Prepare data for Direction 2: PCA Factor Decomposition (BDF) for Intraday Volume.

Transforms raw 1-minute OHLCV bars into the datasets needed by the BDF algorithm:
1. 15-minute volume bins (k=26 per regular trading day)
2. ADV-normalized turnover (since TSO is unavailable)
3. Turnover matrix X of shape (days * k, N) for each period
4. Train/validate/test splits with VWAP reference prices

Functions
---------
get_bin_edges
    Generate 15-minute bin edges for regular trading hours.
load_and_validate_ticker
    Load raw 1-minute data for a single ticker.
aggregate_to_15min_bins
    Aggregate 1-minute bars into 15-minute volume bins.
compute_adv_normalized_turnover
    Normalize bin volume by trailing average daily volume.
build_turnover_matrix
    Build the (P, N) turnover matrix for a date range.
compute_vwap
    Compute VWAP from 1-minute bars for execution cost evaluation.
"""

import json
import os
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────

RAW_DIR = Path("data/direction_2")
PREPARED_DIR = Path("data/direction_2/prepared")
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ",
    "WMT",
]
N_STOCKS = len(TICKERS)

# BDF parameters
K_BINS = 26  # 15-minute bins per regular day (390 min / 15 = 26)
BIN_MINUTES = 15
L_WINDOW = 20  # rolling estimation window in trading days
ADV_WINDOW = 60  # trailing days for ADV normalization

# Regular trading hours: 09:30 - 15:59 (390 minutes)
REGULAR_MINUTES = 390

# Half-days identified from the raw data (fewer than 390 1-min bars)
HALF_DAYS = {
    _date(2023, 7, 3), _date(2023, 11, 24),
    _date(2024, 7, 3), _date(2024, 11, 29), _date(2024, 12, 24),
    _date(2025, 7, 3), _date(2025, 11, 28), _date(2025, 12, 24),
}

# Date splits (from acquisition report, adjusted for EQUS.MINI availability)
TRAIN_START = _date(2023, 3, 28)
TRAIN_END = _date(2024, 6, 30)
VAL_START = _date(2024, 7, 1)
VAL_END = _date(2024, 12, 31)
TEST_START = _date(2025, 1, 1)
TEST_END = _date(2025, 12, 31)


def get_bin_edges() -> list[str]:
    """
    Generate 15-minute bin start times for regular trading hours.

    Returns
    -------
    List of time strings like ['09:30', '09:45', ..., '15:45'].
    """
    edges = []
    hour, minute = 9, 30
    for _ in range(K_BINS):
        edges.append(f"{hour:02d}:{minute:02d}")
        minute += BIN_MINUTES
        if minute >= 60:
            hour += 1
            minute -= 60
    return edges


def load_and_validate_ticker(ticker: str) -> pd.DataFrame:
    """
    Load raw 1-minute data for a single ticker and validate structure.

    Parameters
    ----------
    ticker: Stock ticker symbol.

    Returns
    -------
    DataFrame with columns [date_event, ts_event, open, high, low, close, volume].
    """
    path = RAW_DIR / f"{ticker}_1m.parquet"
    df = pd.read_parquet(path)

    required_cols = {"date_event", "ts_event", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    df["volume"] = df["volume"].astype(np.float64)
    return df


def aggregate_to_15min_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute bars into 15-minute volume bins.

    Each bin spans 15 minutes. Bin 0 = 09:30-09:44, bin 1 = 09:45-09:59, ...,
    bin 25 = 15:45-15:59.

    Parameters
    ----------
    df: Raw 1-minute DataFrame for a single ticker (already filtered to full days).

    Returns
    -------
    DataFrame with columns [date, bin_idx, volume, vwap_price] where bin_idx
    ranges from 0 to K_BINS-1.
    """
    # Create a combined datetime for bin assignment
    df = df.copy()
    # ts_event is datetime.time objects
    df["hour"] = df["ts_event"].apply(lambda t: t.hour)
    df["minute"] = df["ts_event"].apply(lambda t: t.minute)

    # Minutes since 09:30
    df["min_since_open"] = (df["hour"] - 9) * 60 + df["minute"] - 30
    df["bin_idx"] = df["min_since_open"] // BIN_MINUTES

    # Compute dollar volume at the minute level for VWAP calculation
    df["dollar_volume"] = df["close"] * df["volume"]

    # Aggregate volume per bin
    agg = df.groupby(["date_event", "bin_idx"]).agg(
        volume=("volume", "sum"),
        dollar_volume=("dollar_volume", "sum"),
    ).reset_index()

    agg["vwap_price"] = agg["dollar_volume"] / agg["volume"].replace(0, np.nan)
    agg.drop(columns=["dollar_volume"], inplace=True)
    agg.rename(columns={"date_event": "date"}, inplace=True)

    return agg


def build_full_day_grid(dates: list[str]) -> pd.DataFrame:
    """
    Build a complete grid of (date, bin_idx) for all regular trading days.

    Parameters
    ----------
    dates: List of date strings for regular trading days.

    Returns
    -------
    DataFrame with columns [date, bin_idx] covering all date x bin combinations.
    """
    grid = pd.DataFrame(
        [(d, b) for d in dates for b in range(K_BINS)],
        columns=["date", "bin_idx"],
    )
    return grid


def prepare_15min_volume() -> tuple[pd.DataFrame, list[str]]:
    """
    Load all tickers, exclude half-days, aggregate to 15-min bins.

    Returns
    -------
    Tuple of (panel DataFrame with [date, bin_idx, ticker, volume, vwap_price],
    list of regular trading dates).
    """
    all_frames = []
    regular_dates = None

    for ticker in TICKERS:
        print(f"  Loading {ticker}...")
        df = load_and_validate_ticker(ticker)

        # Exclude half-days
        df = df[~df["date_event"].isin(HALF_DAYS)]

        # Identify regular trading days for this ticker (should be ~686)
        if regular_dates is None:
            day_counts = df.groupby("date_event").size()
            regular_dates = sorted(day_counts[day_counts >= REGULAR_MINUTES - 5].index.tolist())

        agg = aggregate_to_15min_bins(df)
        agg["ticker"] = ticker

        # Ensure complete grid (fill missing bins with 0 volume)
        grid = build_full_day_grid(regular_dates)
        grid["ticker"] = ticker
        agg = grid.merge(agg, on=["date", "bin_idx", "ticker"], how="left")
        agg["volume"] = agg["volume"].fillna(0.0)

        all_frames.append(agg)

    panel = pd.concat(all_frames, ignore_index=True)
    return panel, regular_dates


def compute_daily_volume(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total daily volume per ticker from 15-min binned data.

    Parameters
    ----------
    panel: Panel DataFrame with [date, bin_idx, ticker, volume].

    Returns
    -------
    DataFrame with columns [date, ticker, daily_volume].
    """
    daily = panel.groupby(["date", "ticker"])["volume"].sum().reset_index()
    daily.rename(columns={"volume": "daily_volume"}, inplace=True)
    return daily


def compute_adv(daily_vol: pd.DataFrame, window: int = ADV_WINDOW) -> pd.DataFrame:
    """
    Compute trailing average daily volume for each ticker.

    Uses a rolling mean of the past `window` trading days. The first
    `window` days use an expanding mean to avoid NaN.

    Parameters
    ----------
    daily_vol: DataFrame with [date, ticker, daily_volume].
    window:    Number of trailing days for the rolling average.

    Returns
    -------
    DataFrame with columns [date, ticker, adv].
    """
    daily_vol = daily_vol.sort_values(["ticker", "date"])
    daily_vol["adv"] = daily_vol.groupby("ticker")["daily_volume"].transform(
        lambda s: s.rolling(window, min_periods=1).mean().shift(1)
    )
    # First day has no prior data; use the day's own volume as fallback
    mask_null = daily_vol["adv"].isna()
    daily_vol.loc[mask_null, "adv"] = daily_vol.loc[mask_null, "daily_volume"]

    return daily_vol[["date", "ticker", "adv"]]


def compute_turnover(panel: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ADV-normalized turnover for each bin.

    turnover = bin_volume / adv (trailing average daily volume)

    This is the ADV workaround for missing TSO data. It produces a
    normalization functionally equivalent to volume/shares_outstanding
    for the purpose of cross-sectional PCA.

    Parameters
    ----------
    panel: Panel DataFrame with [date, bin_idx, ticker, volume].
    adv:   DataFrame with [date, ticker, adv].

    Returns
    -------
    Panel with added 'turnover' column.
    """
    panel = panel.merge(adv, on=["date", "ticker"], how="left")
    panel["turnover"] = panel["volume"] / panel["adv"].replace(0, np.nan)
    # Fill NaN turnover (from zero ADV) with 0
    panel["turnover"] = panel["turnover"].fillna(0.0)
    return panel


def build_turnover_matrix(
    panel: pd.DataFrame,
    dates: list[str],
    tickers: list[str],
) -> np.ndarray:
    """
    Build the turnover matrix X of shape (P, N).

    P = len(dates) * K_BINS, N = len(tickers).
    Rows are time-ordered: day 1 bin 0, day 1 bin 1, ..., day 1 bin K-1,
    day 2 bin 0, ..., day L bin K-1.
    Columns are stocks in ticker order.

    Parameters
    ----------
    panel:   Panel DataFrame with [date, bin_idx, ticker, turnover].
    dates:   Ordered list of trading dates.
    tickers: Ordered list of ticker symbols.

    Returns
    -------
    NumPy array of shape (P, N) with float64 turnover values.
    """
    P = len(dates) * K_BINS
    N = len(tickers)
    X = np.zeros((P, N), dtype=np.float64)

    # Filter panel to requested dates
    panel_sub = panel[panel["date"].isin(set(dates))].copy()

    for j, ticker in enumerate(tickers):
        tk_data = panel_sub[panel_sub["ticker"] == ticker].copy()
        tk_data = tk_data.sort_values(["date", "bin_idx"])
        vals = tk_data["turnover"].values
        if len(vals) == P:
            X[:, j] = vals
        else:
            # Shouldn't happen with complete grid, but handle gracefully
            print(f"  WARNING: {ticker} has {len(vals)} rows, expected {P}")
            X[:len(vals), j] = vals[:P]

    return X


def build_vwap_reference(panel: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """
    Compute daily VWAP for each ticker from 15-min bin data.

    This serves as the benchmark for VWAP execution cost evaluation.

    Parameters
    ----------
    panel: Panel DataFrame with [date, bin_idx, ticker, volume, vwap_price].
    dates: Ordered list of trading dates.

    Returns
    -------
    DataFrame with columns [date, ticker, daily_vwap].
    """
    sub = panel[panel["date"].isin(set(dates))].copy()
    sub["dollar_vol"] = sub["volume"] * sub["vwap_price"].fillna(0)
    daily = sub.groupby(["date", "ticker"]).agg(
        total_volume=("volume", "sum"),
        total_dollar_vol=("dollar_vol", "sum"),
    ).reset_index()
    daily["daily_vwap"] = daily["total_dollar_vol"] / daily["total_volume"].replace(0, np.nan)
    return daily[["date", "ticker", "daily_vwap"]]


def split_dates(
    regular_dates: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Split regular trading dates into train/validate/test periods.

    Parameters
    ----------
    regular_dates: Sorted list of all regular trading date strings.

    Returns
    -------
    Tuple of (train_dates, val_dates, test_dates).
    """
    train = [d for d in regular_dates if TRAIN_START <= d <= TRAIN_END]
    val = [d for d in regular_dates if VAL_START <= d <= VAL_END]
    test = [d for d in regular_dates if TEST_START <= d <= TEST_END]
    return train, val, test


def save_summary(
    regular_dates: list[str],
    train_dates: list[str],
    val_dates: list[str],
    test_dates: list[str],
    panel: pd.DataFrame,
) -> None:
    """
    Save a JSON summary of the prepared datasets.

    Parameters
    ----------
    regular_dates: All regular trading dates.
    train_dates:   Training period dates.
    val_dates:     Validation period dates.
    test_dates:    Test period dates.
    panel:         Full panel DataFrame.
    """
    def _d(d):
        return str(d) if d is not None else None

    summary = {
        "tickers": TICKERS,
        "n_stocks": N_STOCKS,
        "k_bins": K_BINS,
        "bin_minutes": BIN_MINUTES,
        "l_window": L_WINDOW,
        "adv_window": ADV_WINDOW,
        "half_days_excluded": [str(d) for d in sorted(HALF_DAYS)],
        "total_regular_days": len(regular_dates),
        "date_range": {
            "start": _d(regular_dates[0]),
            "end": _d(regular_dates[-1]),
        },
        "splits": {
            "train": {
                "start": _d(train_dates[0]) if train_dates else None,
                "end": _d(train_dates[-1]) if train_dates else None,
                "n_days": len(train_dates),
            },
            "validate": {
                "start": _d(val_dates[0]) if val_dates else None,
                "end": _d(val_dates[-1]) if val_dates else None,
                "n_days": len(val_dates),
            },
            "test": {
                "start": _d(test_dates[0]) if test_dates else None,
                "end": _d(test_dates[-1]) if test_dates else None,
                "n_days": len(test_dates),
            },
        },
        "normalization": "ADV (trailing 60-day average daily volume)",
        "turnover_stats": {
            "mean": float(panel["turnover"].mean()),
            "std": float(panel["turnover"].std()),
            "min": float(panel["turnover"].min()),
            "max": float(panel["turnover"].max()),
            "q05": float(panel["turnover"].quantile(0.05)),
            "q95": float(panel["turnover"].quantile(0.95)),
        },
    }
    with open(PREPARED_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    """
    Run the full data preparation pipeline.
    """
    print("=" * 70)
    print("Step 3: Data Preparation — Direction 2 (BDF PCA Volume)")
    print("=" * 70)

    # ── 1. Load and aggregate ──────────────────────────────────────────
    print("\n[1/6] Loading raw 1-min data and aggregating to 15-min bins...")
    panel, regular_dates = prepare_15min_volume()
    print(f"  Regular trading days: {len(regular_dates)}")
    print(f"  Panel rows: {len(panel):,}")

    # ── 2. Compute daily volume and ADV ────────────────────────────────
    print("\n[2/6] Computing daily volume and trailing ADV...")
    daily_vol = compute_daily_volume(panel)
    adv = compute_adv(daily_vol)
    print(f"  ADV window: {ADV_WINDOW} days (trailing, shifted by 1 day)")

    # ── 3. Compute ADV-normalized turnover ─────────────────────────────
    print("\n[3/6] Computing ADV-normalized turnover...")
    panel = compute_turnover(panel, adv)
    turnover_stats = panel["turnover"].describe()
    print(f"  Turnover stats:\n{turnover_stats}")

    # ── 4. Split dates ─────────────────────────────────────────────────
    print("\n[4/6] Splitting into train/validate/test periods...")
    train_dates, val_dates, test_dates = split_dates(regular_dates)
    print(f"  Train:    {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"  Validate: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
    print(f"  Test:     {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # ── 5. Build and save turnover matrices ────────────────────────────
    print("\n[5/6] Building and saving turnover matrices...")

    for split_name, split_dates_list in [
        ("train", train_dates),
        ("validate", val_dates),
        ("test", test_dates),
    ]:
        X = build_turnover_matrix(panel, split_dates_list, TICKERS)
        expected_P = len(split_dates_list) * K_BINS
        assert X.shape == (expected_P, N_STOCKS), (
            f"{split_name}: shape {X.shape} != ({expected_P}, {N_STOCKS})"
        )
        out_path = PREPARED_DIR / f"turnover_matrix_{split_name}.npy"
        np.save(out_path, X)
        print(f"  {split_name}: shape {X.shape}, saved to {out_path}")

    # Save full turnover matrix (all dates) for rolling-window usage
    X_full = build_turnover_matrix(panel, regular_dates, TICKERS)
    np.save(PREPARED_DIR / "turnover_matrix_full.npy", X_full)
    print(f"  full: shape {X_full.shape}, saved to turnover_matrix_full.npy")

    # ── 6. Save panel, metadata, and VWAP reference ────────────────────
    print("\n[6/6] Saving panel data, VWAP reference, and metadata...")

    # Save the full panel (for the developer to use in rolling-window mode)
    panel_save_cols = ["date", "bin_idx", "ticker", "volume", "turnover", "vwap_price"]
    panel[panel_save_cols].to_parquet(PREPARED_DIR / "panel_15min.parquet", index=False)
    print(f"  Panel saved: {len(panel):,} rows")

    # Save daily volume and ADV for reference
    daily_vol_merged = daily_vol.merge(adv, on=["date", "ticker"], how="left")
    daily_vol_merged.to_parquet(PREPARED_DIR / "daily_volume_adv.parquet", index=False)
    print(f"  Daily volume + ADV saved: {len(daily_vol_merged):,} rows")

    # Save VWAP reference prices for execution cost evaluation
    vwap_ref = build_vwap_reference(panel, regular_dates)
    vwap_ref.to_parquet(PREPARED_DIR / "vwap_reference.parquet", index=False)
    print(f"  VWAP reference saved: {len(vwap_ref):,} rows")

    # Save ordered dates and tickers for matrix indexing
    np.save(PREPARED_DIR / "tickers.npy", np.array(TICKERS))
    np.save(PREPARED_DIR / "regular_dates.npy", np.array([str(d) for d in regular_dates]))
    np.save(PREPARED_DIR / "train_dates.npy", np.array([str(d) for d in train_dates]))
    np.save(PREPARED_DIR / "val_dates.npy", np.array([str(d) for d in val_dates]))
    np.save(PREPARED_DIR / "test_dates.npy", np.array([str(d) for d in test_dates]))

    # Save summary JSON
    save_summary(regular_dates, train_dates, val_dates, test_dates, panel)

    print("\n" + "=" * 70)
    print("Data preparation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

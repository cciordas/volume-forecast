# Preparation Report: Direction 7 — Kalman Filter State-Space Model for Intraday Volume

## Status: all prepared

## Overview

Raw 1-minute bar data was aggregated to 15-minute bins (I=26 per session), normalized
by rolling 60-day average daily volume (ADV-60), and log-transformed. All 10 tickers
produced clean 559 x 26 matrices with zero missing observations.

### Key Decisions

- **Half-day detection:** Used SPY (most liquid) to identify 7 half-day sessions. Less
  liquid tickers (DIA, IBM) have missing 1-minute bars on normal days due to low
  liquidity, not early closes. Per-ticker bar-count thresholds would incorrectly
  exclude most days for DIA.
- **Normalization:** Used ADV-60 (rolling 60-day trailing average daily volume, lagged
  by 1 day) instead of shares outstanding. This is the recommended workaround from the
  data requirements (Gap 1). The daily component eta_t absorbs remaining day-to-day
  scale changes.
- **ADV warm-up:** The first 60 full trading days are dropped because there is
  insufficient history to compute ADV-60. This reduces the dataset from 619 to 559 days.
- **Date range substitution:** Uses 2023-12-28 to 2026-03-31 instead of the paper's
  2014-01-02 to 2016-06-30 (Databento coverage limitation, documented in acquisition
  report).

## Datasets

### 1. Log-Volume Matrices (Primary Model Input)
- **Source requirement:** Requirement 1 (Intraday Volume Bars)
- **Raw data:** `data/direction_7/{TICKER}_1m.parquet`
- **Prepared data:** `data/direction_7/prepared/{TICKER}_log_volume.parquet`
- **Transformations applied:**
  1. Filtered to regular session hours (09:30-16:00)
  2. Excluded 7 half-day sessions identified via SPY
  3. Aggregated 1-minute bars to 15-minute bins by summing volume
  4. Computed ADV-60 from daily volume totals (lagged by 1 day)
  5. Normalized: vol_norm = bin_volume / ADV-60
  6. Log-transformed: log_vol = ln(vol_norm)
  7. Dropped first 60 days (insufficient ADV history)
- **Shape:** 559 x 26 (days x bins) per ticker
- **Date range:** 2023-12-28 to 2026-03-31
- **Columns:** bin_00 through bin_25 (15-min bins from 09:30 to 15:45)
- **Index:** date (Python date objects)
- **Status:** ready
- **Notes:** All bins observed (no NaN values). The log-volume values show the expected
  U-shaped intraday pattern (higher at open/close, lower midday). The Developer should
  use this as y[t, i] in the Kalman filter. To build the unified time series y[tau],
  flatten row-by-row: tau = (t-1)*26 + i.

### 2. Observation Masks
- **Source requirement:** Requirement 1
- **Prepared data:** `data/direction_7/prepared/{TICKER}_observed.parquet`
- **Transformations applied:** Boolean matrix, True where bin volume > 0
- **Shape:** 559 x 26 per ticker
- **Status:** ready
- **Notes:** All values are True for all tickers (no zero-volume 15-min bins). The
  Developer can use this as is_observed[tau]. If future datasets include illiquid
  securities, this mask enables the missing-data handling in the Kalman filter.

### 3. Raw 15-Minute Volume (for Debugging and VWAP)
- **Source requirement:** Requirement 1
- **Prepared data:** `data/direction_7/prepared/{TICKER}_raw_volume_15min.parquet`
- **Transformations applied:** Same aggregation as Dataset 1, but without normalization
  or log transform. Integer share counts.
- **Shape:** 559 x 26 per ticker
- **Status:** ready
- **Notes:** Useful for converting log-volume forecasts back to share counts for VWAP
  weight computation (Step 6 in the impl spec: bias-corrected exponentiation).

### 4. Long-Form Detail (for Debugging)
- **Source requirement:** Requirement 1
- **Prepared data:** `data/direction_7/prepared/{TICKER}_long.parquet`
- **Transformations applied:** All intermediate values preserved in long format.
- **Columns:** date, bin_idx, volume, adv, vol_norm, is_observed, log_vol
- **Shape:** 14,534 rows per ticker (559 days x 26 bins)
- **Status:** ready
- **Notes:** Useful for debugging normalization or inspecting individual bins. Not
  needed by the model directly.

### 5. Trading Calendar
- **Source requirement:** Requirement 3 (Trading Calendar)
- **Raw data:** Derived from SPY 1-minute bars
- **Prepared data:** `data/direction_7/prepared/trading_calendar.parquet`
- **Transformations applied:** Classified each trading day as full/half-day, marked
  which days appear in the prepared dataset.
- **Shape:** 626 rows (all trading days in raw data)
- **Columns:** date, bars_1m, is_full_day, is_half_day, in_prepared_data
- **Status:** ready
- **Notes:** 626 total trading days, 619 full days, 7 half-day sessions excluded,
  559 days in prepared data (after ADV-60 warm-up). The Developer can use
  `in_prepared_data == True` to get the exact date list for the model.

### 6. Daily OHLCV Cross-Check
- **Source requirement:** Requirement 4 (Daily OHLCV)
- **Raw data:** `data/direction_7/{TICKER}_1d.parquet`
- **Prepared data:** `data/direction_7/prepared/daily_ohlcv.parquet`
- **Transformations applied:** Combined all 10 tickers into a single file with a
  ticker column. No other transformations.
- **Shape:** 6,260 rows (626 days x 10 tickers)
- **Columns:** date_event, volume, close, ticker
- **Status:** ready
- **Notes:** Cross-check: sum of intraday 15-min volume / daily bar volume has mean
  ratio 0.967 for SPY (expected <1.0 because daily bars include pre/post-market volume).
  The rolling mean baseline (Sanity Check 5 in impl spec) can be computed from the
  log-volume matrices directly.

### 7. Metadata
- **Prepared data:** `data/direction_7/prepared/metadata.json`
- **Contents:** All configuration parameters (bins_per_day=26, bin_width=15min,
  ADV lookback=60, session hours, tickers, excluded dates, per-ticker summaries).
- **Status:** ready

## Per-Ticker Summary

| Ticker | Shape | Zero Bins | Log-Vol Range | Notes |
|---|---|---|---|---|
| SPY | 559 x 26 | 0 | [-6.25, 0.76] | Most liquid; reference ticker |
| DIA | 559 x 26 | 0 | [-9.37, -0.31] | Lower volume than others |
| QQQ | 559 x 26 | 0 | [-5.79, 0.64] | |
| AAPL | 559 x 26 | 0 | [-5.92, 0.66] | |
| AMZN | 559 x 26 | 0 | [-5.88, 0.85] | |
| GOOG | 559 x 26 | 0 | [-6.70, 0.19] | |
| IBM | 559 x 26 | 0 | [-8.75, 0.11] | Lower volume; many missing 1m bars but 15m aggregation fills in |
| JPM | 559 x 26 | 0 | [-6.45, 0.43] | |
| MSFT | 559 x 26 | 0 | [-5.98, 0.72] | |
| XOM | 559 x 26 | 0 | [-6.89, 0.24] | |

## Issues

None — all datasets ready for implementation.

### Deviations from Paper

1. **Normalization:** ADV-60 instead of shares outstanding (documented in data
   requirements Gap 1). Effect: absolute log-volume scale differs from the paper,
   but the model's relative structure (daily AR, intraday AR, seasonal pattern) is
   unaffected. The daily component eta_t absorbs the normalizer's slow drift.

2. **Date range:** 2023-12-28 to 2026-03-31 instead of paper's 2014-2016 (Databento
   coverage limitation). The algorithm is date-agnostic.

3. **ADV warm-up:** 60 days dropped at the start. The paper's normalization by shares
   outstanding does not require warm-up, so the paper uses the full training window.
   559 days is still ample for the model's maximum 24-month training window (the paper
   tests up to T_train=504 days).

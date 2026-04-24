# Preparation Report: Direction 2 — PCA Factor Decomposition (BDF) for Intraday Volume

## Status: all prepared

## Datasets

### 1. Turnover Matrices (train / validate / test / full)
- **Source requirement:** Requirement 1 (Intraday Volume Bars) + Requirement 2 workaround (ADV normalization)
- **Raw data:** `data/direction_2/{TICKER}_1m.parquet` (30 tickers)
- **Prepared data:**
  - `data/direction_2/prepared/turnover_matrix_train.npy` — shape (8164, 30)
  - `data/direction_2/prepared/turnover_matrix_validate.npy` — shape (3250, 30)
  - `data/direction_2/prepared/turnover_matrix_test.npy` — shape (6422, 30)
  - `data/direction_2/prepared/turnover_matrix_full.npy` — shape (17836, 30)
- **Transformations applied:**
  1. Excluded 8 half-days (Jul 3, Nov Black Friday, Dec 24 across 2023-2025).
  2. Aggregated 1-minute bars into 15-minute bins (k=26 per regular day, 390 min / 15 = 26).
  3. Computed trailing 60-day average daily volume (ADV) per ticker, shifted by 1 day (no lookahead).
  4. Normalized each bin's volume by ADV to produce turnover: `turnover = bin_volume / adv_60`.
  5. Built turnover matrix X of shape (P, N) where P = days * 26 bins, N = 30 stocks. Rows are time-ordered (day 1 bin 0, day 1 bin 1, ..., day L bin 25). Columns are stocks in ticker order. No demeaning applied (per BDF 2008 / Bai 2003).
- **Shape:** See per-split shapes above. Full matrix: 17836 x 30.
- **Date range:**
  - Train: 2023-03-28 to 2024-06-28 (314 days)
  - Validate: 2024-07-01 to 2024-12-31 (125 days)
  - Test: 2025-01-02 to 2025-12-31 (247 days)
  - Full: 2023-03-28 to 2025-12-31 (686 days)
- **Columns:** N = 30 stocks in order: AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GS, HD, HON, IBM, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, NVDA, PG, SHW, TRV, UNH, V, VZ, WMT.
- **Status:** ready
- **Notes:**
  - The Developer should use the full matrix with a rolling L=20 day window for the daily pipeline, extracting the trailing 20 * 26 = 520 rows as the estimation window X for each forecast day.
  - Turnover values are float64. Mean ~0.039, std ~0.043, range [1.6e-5, 2.36]. The max of 2.36 represents extreme volume spikes (likely earnings/news days) — the Developer should be aware these may affect PCA factor extraction.
  - No NaNs, no negative values, no zero rows. Zero-volume 1-minute bars were filled with 0 before aggregation.

### 2. Panel Data (15-minute bins)
- **Source requirement:** Requirements 1 and 2 (combined)
- **Raw data:** `data/direction_2/{TICKER}_1m.parquet`
- **Prepared data:** `data/direction_2/prepared/panel_15min.parquet`
- **Transformations applied:** Same as Dataset 1, but stored in long panel format (date x bin_idx x ticker) for flexible querying.
- **Shape:** 535,080 rows x 6 columns
- **Date range:** 2023-03-28 to 2025-12-31 (686 regular trading days)
- **Columns:** date, bin_idx, ticker, volume (raw bin volume), turnover (ADV-normalized), vwap_price (volume-weighted close price within the 15-min bin)
- **Status:** ready
- **Notes:** This is the same data as the turnover matrices but in a more convenient format for ad-hoc analysis, plotting, and the dynamic VWAP execution evaluation (which needs per-bin prices). The `vwap_price` column uses the 1-minute close price as a proxy for trade price — it is the dollar-volume-weighted average close within each 15-min bin.

### 3. Daily Volume and ADV
- **Source requirement:** Requirement 1 (derived)
- **Raw data:** `data/direction_2/{TICKER}_1m.parquet`
- **Prepared data:** `data/direction_2/prepared/daily_volume_adv.parquet`
- **Transformations applied:** Summed 1-minute volumes to daily totals, computed trailing 60-day rolling mean (shifted by 1 day).
- **Shape:** 20,580 rows x 4 columns
- **Date range:** 2023-03-28 to 2025-12-31
- **Columns:** date, ticker, daily_volume, adv
- **Status:** ready
- **Notes:** Useful for the Developer if they want to convert turnover forecasts back to raw volume (multiply by ADV). Also serves as a reference for the ADV normalization used.

### 4. VWAP Reference Prices
- **Source requirement:** Requirement 1 (derived)
- **Raw data:** `data/direction_2/{TICKER}_1m.parquet`
- **Prepared data:** `data/direction_2/prepared/vwap_reference.parquet`
- **Transformations applied:** Computed daily VWAP as sum(close * volume) / sum(volume) across all 15-min bins.
- **Shape:** 20,580 rows x 3 columns
- **Date range:** 2023-03-28 to 2025-12-31
- **Columns:** date, ticker, daily_vwap
- **Status:** ready
- **Notes:** The VWAP execution cost metric (BDF 2008, Section 4.3) requires comparing the achieved execution price against the true end-of-day VWAP. This dataset provides the benchmark VWAP for each stock on each day. Note: VWAP is computed from 1-minute close prices as a proxy (true tick-level VWAP would require trade data, which is not available in OHLCV bars).

### 5. Metadata Files
- **Prepared data:**
  - `data/direction_2/prepared/tickers.npy` — ordered array of 30 ticker symbols
  - `data/direction_2/prepared/regular_dates.npy` — all 686 regular trading dates
  - `data/direction_2/prepared/train_dates.npy` — 314 training dates
  - `data/direction_2/prepared/val_dates.npy` — 125 validation dates
  - `data/direction_2/prepared/test_dates.npy` — 247 test dates
  - `data/direction_2/prepared/summary.json` — configuration and turnover statistics
- **Status:** ready
- **Notes:** The Developer should load `tickers.npy` and `regular_dates.npy` to index into the turnover matrices. Column j of the matrix corresponds to `tickers[j]`. Row i corresponds to `(regular_dates[i // 26], i % 26)` — i.e., date index `i // 26` and bin index `i % 26`.

## Issues

None — all datasets ready for implementation.

## Notes for the Developer

1. **ADV normalization instead of TSO:** Since shares outstanding data is unavailable (see data_requirements.md), turnover is computed as `bin_volume / trailing_60day_ADV` rather than `volume / shares_outstanding`. This achieves the same cross-sectional normalization effect needed for PCA. If TSO data becomes available, substitute `turnover = volume / tso` and re-run preparation.

2. **Rolling window usage:** The full turnover matrix (`turnover_matrix_full.npy`) is designed for the rolling-window daily pipeline. For forecast day d (0-indexed into `regular_dates`), the estimation window is rows `(d-L)*26` through `d*26` (exclusive), giving a (L*26, 30) submatrix. Ensure d >= L (= 20) before producing forecasts.

3. **Half-day exclusion:** 8 half-days were excluded entirely. The estimation window will seamlessly skip these days. The Developer does not need to handle them.

4. **Turnover extremes:** A few bins have turnover > 1.0 (max = 2.36), indicating that a single 15-minute bin's volume exceeded the 60-day average daily volume. These are likely earnings/news days. The impl spec does not call for outlier removal, but the Developer should be aware that these may influence PCA factor extraction during the relevant windows.

5. **Bin-to-time mapping:** Bin 0 = 09:30-09:44, bin 1 = 09:45-09:59, ..., bin 25 = 15:45-15:59. Each bin spans exactly 15 minutes. k = 26.

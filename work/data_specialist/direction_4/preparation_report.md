# Preparation Report: Direction 4 — Dual-Mode Intraday Volume Forecast

## Status: all prepared

## Datasets

### 1. Per-Ticker 15-Minute Volume Arrays
- **Source requirement:** Requirement 1 (Intraday Volume Bars)
- **Raw data:** `data/direction_4/{TICKER}_1m.parquet` (35 files)
- **Prepared data:** `data/direction_4/prepared/{TICKER}_15m_volume.parquet` (35 files)
- **Transformations applied:**
  - 1-minute bars aggregated to 26 x 15-minute bins per day (bins 1-26, covering 9:30-16:00 ET). Each bin sums all 1-minute volume bars whose start time falls within the 15-minute window.
  - Bars outside regular trading hours (before 9:30 or at/after 16:00) excluded.
  - Split adjustment applied: NVDA pre-2024-06-10 volume divided by 10 (10:1 split); WMT pre-2024-02-26 volume divided by 3 (3:1 split).
  - Half-day trading sessions excluded (sessions ending before 14:30 ET). Six dates identified across the data window: 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, 2025-12-24. Not all tickers had data on all half-days.
  - Missing bins (minutes with no trade activity) filled with 0 volume.
- **Shape:** (496 days, 26 bins) median across tickers. Range: 496-499 days.
- **Date range:** 2024-01-02 to 2025-12-31
- **Columns:** `bin_1` through `bin_26` (1-indexed as per impl spec). Index is `date` (datetime.date).
- **Status:** ready
- **Notes:** Volume is float64 (fractional after split adjustment for pre-split NVDA/WMT). The impl spec's `volume_history[stock, day, bin]` maps directly to these arrays: `vol_df.loc[date, f"bin_{i}"]` gives `volume_history[stock, date, i]`. Zero-volume bins occur in ~0.02% of cells, concentrated in less liquid tickers (TRV, SHW, GS) during midday bins. The impl spec's min-nonzero floor in Functions 1/1a handles these.

### 2. Stacked Volume History Panel
- **Source requirement:** Requirement 1 (convenience format combining all tickers)
- **Raw data:** All per-ticker 15-minute files
- **Prepared data:** `data/direction_4/prepared/volume_history.parquet`
- **Transformations applied:** Vertical concatenation of all per-ticker 15-minute arrays with a `ticker` column added. Sorted by (ticker, date).
- **Shape:** (17376 rows, 28 columns) — 35 tickers x ~496 days, columns: date, bin_1..bin_26, ticker
- **Date range:** 2024-01-02 to 2025-12-31
- **Columns:** `date`, `bin_1` through `bin_26`, `ticker`
- **Status:** ready
- **Notes:** Alternative to loading per-ticker files individually. Filter by ticker to get a single stock's history: `vol[vol['ticker'] == 'AAPL']`.

### 3. Daily Stats (for Universe Selection and ADV)
- **Source requirement:** Requirement 2 (Daily OHLCV Bars)
- **Raw data:** `data/direction_4/{TICKER}_1d.parquet` (35 files)
- **Prepared data:** `data/direction_4/prepared/daily_stats.parquet`
- **Transformations applied:**
  - Split adjustment for NVDA and WMT: pre-split prices divided by ratio, pre-split volumes multiplied by ratio (standard adjustment convention for daily bars).
  - Dollar volume computed as `close * volume`.
  - Ticker column added; all tickers concatenated and sorted by (ticker, date).
- **Shape:** (17570 rows, 8 columns)
- **Date range:** 2024-01-02 to 2025-12-31
- **Columns:** `date`, `open`, `high`, `low`, `close`, `volume`, `dollar_volume`, `ticker`
- **Status:** ready
- **Notes:** Includes all 502 trading days (no half-day exclusion for daily bars). Dollar volume can be used for universe ranking per the impl spec ("top 500 by dollar volume"). ADV-60 or other rolling averages can be computed from the volume column.

### 4. Metadata
- **Source requirement:** All requirements (reference file)
- **Prepared data:** `data/direction_4/prepared/metadata.json`
- **Contents:** Ticker list, bin configuration, split details (dates and ratios), half-day exclusions per ticker, date ranges and day counts per ticker, total summary statistics.
- **Status:** ready

## Issues

None — all datasets ready for implementation.

**Developer notes:**
- The primary input for the model is the per-ticker 15-minute volume files (Dataset 1). Each file's row index is a `datetime.date` and columns `bin_1`..`bin_26` correspond to the impl spec's `volume_history[stock, day, bin]` with 1-indexed bins.
- Volume values are float64. For non-split-adjusted tickers, values are exact integers stored as float. For NVDA and WMT, pre-split values are fractional (divided by 10 and 3 respectively).
- The model requires 126 trading days of warm-up (N_seasonal). With ~496 days available, the first model can be trained starting around day 127, leaving ~370 days for training + validation + out-of-sample.
- Half-day sessions (6 dates) have been excluded. If the Developer needs to handle them differently, the raw 1-minute data is still available in `data/direction_4/`.
- NVDA shows a large volume regime shift post-split (from ~85K to ~6-9M shares/day). The per-stock model fitting handles this naturally since all lookback windows use the same split-adjusted volume, but the Developer should be aware that NVDA's volume characteristics change dramatically mid-dataset.

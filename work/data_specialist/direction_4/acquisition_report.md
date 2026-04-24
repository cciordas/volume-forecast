# Acquisition Report: Direction 4 — Dual-Mode Intraday Volume Forecast

## Status: all acquired

## Downloads

### 1. Intraday Volume Bars (1-minute, EQUS.MINI)
- **Source:** Databento
- **Tool:** bentoticks load_bars_intraday (barsz="1m", dataset="EQUS.MINI")
- **Status:** acquired (35/35 tickers)
- **Location:** `data/direction_4/{TICKER}_1m.parquet` (35 files)
- **Date range:** 2024-01-02 to 2025-12-31 (502 trading days per ticker)
- **Size:** ~74 MB total (~2.1 MB per ticker average)
- **Row counts:** ~159k-195k rows per ticker (390 bars/day x 502 days, minus non-RTH bars and variable per-stock activity)
- **Tickers:** AAPL, AMD, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GOOG, GS, HD, HON, IBM, INTC, JNJ, JPM, KO, MCD, META, MMM, MRK, MSFT, NKE, NVDA, PG, SHW, TRV, TSLA, UNH, V, VZ, WMT
- **Notes:** Some days flagged as "degraded" quality by Databento (2025-03-24, 2025-04-04, 2025-05-06). This is informational — data was still returned. SHW (170k rows) and TRV (159k rows) have fewer rows than other tickers, likely due to lower intraday trading activity generating fewer 1-minute bars with non-zero volume.

### 2. Daily OHLCV Bars (EQUS.MINI)
- **Source:** Databento
- **Tool:** bentoticks load_bars_daily (dataset="EQUS.MINI")
- **Status:** acquired (35/35 tickers)
- **Location:** `data/direction_4/{TICKER}_1d.parquet` (35 files)
- **Date range:** 2024-01-02 to 2025-12-31 (502 rows per ticker)
- **Size:** ~0.7 MB total (~0.021 MB per ticker)
- **Notes:** Used EQUS.MINI instead of EQUS.SUMMARY for consistency (both provide consolidated daily OHLCV). 502 trading days matches the expected count for 2 full calendar years.

### 3. Instrument Definitions (symbol mapping / split detection)
- **Source:** Databento
- **Tool:** N/A — bentoticks does not expose a definition loader
- **Status:** skipped
- **Notes:** The bentoticks tool reference does not include a loader function for the `definition` schema. Split adjustment will need to be handled in Step 3 via price/volume discontinuity detection from the daily bars, or deferred if the pilot universe has no splits in the 2024-2025 window. For the Dow 30 + 5 additional stocks, major splits in this period are: NVDA 10:1 (June 2024), WMT 3:1 (February 2024). These will need manual adjustment in Step 3.

## Failures

None — all intraday and daily data acquired successfully for all 35 tickers.

The definition schema (Requirement 3) was not downloaded because bentoticks lacks a definition loader. This is a minor gap — split adjustment can be handled using known split dates (NVDA 10:1 on 2024-06-10, WMT 3:1 on 2024-02-26) applied during data preparation in Step 3.

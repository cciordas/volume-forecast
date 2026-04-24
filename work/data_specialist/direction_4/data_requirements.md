# Data Requirements: Direction 4 — Dual-Mode Intraday Volume Forecast

## Status: all matched

## Requirements

### 1. Intraday Volume Bars (15-minute bins, training + warm-up)

- **Needed for:** All model components. The core input is `volume_history[stock, day, bin]` — split-adjusted share counts per 15-minute bin. Used by: Function 1 (seasonal factors, 126-day window), Function 1a (historical average, 21-day window), Function 2 (inter-day ARMA, 63-day window), Function 3 (intraday ARMA, 21-day window), Function 4 (regime classifier, 63-day window), Function 5 (weight optimization, 63-day window + 21-day pre-context), Function 7 (percentage model, 63-day window), Function 9 (training orchestration), and Function 10 (daily evaluation).
- **Data type:** OHLCV bars (only the volume field is used; OHLC are not required but come with the schema)
- **Granularity:** 15-minute bars (I = 26 bins per trading day, 9:30-16:00 ET). The source provides 1-minute bars as the finest standard interval; aggregation to 15-minute bins happens in Step 3.
- **Instruments:** A representative universe of liquid U.S. equities. The paper uses the "top 500 by dollar volume" (impl spec line 777, line 1626). For initial development and validation, a smaller pilot set (e.g., 20-50 stocks spanning large-cap, mid-cap, and high-variance names including Dow 30 components) is sufficient. The model is fitted per-stock independently.
- **Date range:** The longest lookback window is N_seasonal = 126 trading days. Adding the training window (N_weight_train = 63 days), the validation split (21 days), and a reasonable out-of-sample evaluation period (63 days), the total data need is approximately 126 + 63 + 21 + 63 = 273 trading days (~13 calendar months). A practical request: **2024-01-02 through 2025-12-31** (2 full calendar years, ~504 trading days). This provides ample data for warm-up, training, validation, and out-of-sample testing, with room for multiple re-estimation cycles.
- **Fields needed:** volume (split-adjusted share count per bin). Timestamp for bin alignment. OHLC fields are included by the schema but unused.
- **Source:** Databento
- **Dataset:** `EQUS.MINI` — composite across all venues, no exchange license fees. Provides consolidated volume across all 16 exchanges and TRFs. This is the cheapest option that meets the requirement (L1/trade-level data is unnecessary; aggregated bars suffice).
- **Schema:** `ohlcv-1m` — 1-minute bars. Will be aggregated to 15-minute bins in Step 3. The `ohlcv-1h` schema is too coarse (1 hour > 15 minutes). There is no native 15-minute bar schema; 1-minute is the correct choice.
- **Notes:** The impl spec requires split-adjusted volume. Databento provides raw (unadjusted) volume. Split adjustment must be applied in Step 3 using corporate action data (see Requirement 2) or by using a provider that handles adjustments. If split adjustment data is unavailable, the impact is limited to stocks that split during the data window — for most stocks, raw volume equals split-adjusted volume. Half-day trading sessions (13 bins instead of 26) should be identified and excluded per the impl spec (edge case 4).

### 2. Daily OHLCV Bars (for stock universe selection and ADV computation)

- **Needed for:** Selecting the stock universe ("top 500 by dollar volume" or a development subset), computing average daily volume (ADV) for potential normalization, and identifying half-day trading sessions (sessions with significantly lower total volume or shorter trading hours).
- **Data type:** OHLCV daily bars
- **Granularity:** 1 day
- **Instruments:** All NMS-listed U.S. equities (broad pull to enable universe selection by ranking on dollar volume)
- **Date range:** **2024-01-02 through 2025-12-31** (matching the intraday data window)
- **Fields needed:** close price, volume, and ideally dollar volume (close * volume) for universe ranking. Symbol/ticker for identification.
- **Source:** Databento
- **Dataset:** `EQUS.SUMMARY` — consolidated daily volume and end-of-day OHLCV across all venues. This is the cheapest dataset for daily bars. It avoids per-exchange data and license fees.
- **Schema:** `ohlcv-1d` — daily bars
- **Notes:** Used primarily for universe construction. Once the stock list is determined, the per-stock intraday data (Requirement 1) is the primary model input. If `EQUS.SUMMARY` does not include dollar volume directly, it can be computed as close * volume.

### 3. Instrument Definitions (for symbol mapping and split detection)

- **Needed for:** Mapping symbols across the data window (ticker changes, symbology), detecting stock splits and corporate actions that affect volume (impl spec edge case 6: "Volume data must be split-adjusted. A 2:1 split doubles apparent volume."), and identifying listing exchange for potential stratification.
- **Data type:** Instrument reference / definition data
- **Granularity:** Point-in-time snapshots
- **Instruments:** All instruments in the selected universe
- **Date range:** **2024-01-02 through 2025-12-31**
- **Fields needed:** Symbol, instrument ID, split factor / corporate action indicator, listing exchange, lot size
- **Source:** Databento
- **Dataset:** `EQUS.MINI` (definition schema is available on this dataset)
- **Schema:** `definition`
- **Notes:** The definition schema provides point-in-time instrument reference data. For split adjustment, the key information is whether any splits occurred during the data window. If Databento's definition schema does not include explicit split factors, an alternative approach is to detect splits from daily data (sudden 2x or 0.5x volume change with corresponding inverse price change) and adjust manually. This requirement is lower priority if the development universe is restricted to stocks without splits in the data window.

## Gaps

None — all requirements matched.

The primary data need is straightforward: 1-minute OHLCV bars from `EQUS.MINI` covering the target stock universe over 2 years. Daily bars from `EQUS.SUMMARY` support universe selection. The only potential complication is split adjustment, which can be handled via the definition schema or manual detection from price/volume discontinuities.

**Cost considerations:** `EQUS.MINI` with `ohlcv-1m` is the most cost-effective choice. It avoids exchange license fees (unlike direct feeds like `XNAS.ITCH` or `XNYS.PILLAR`) and provides consolidated volume across all venues. Higher-granularity schemas (trades, MBP-10, MBO) are unnecessary — the model operates on 15-minute bin volumes, not tick-level data.

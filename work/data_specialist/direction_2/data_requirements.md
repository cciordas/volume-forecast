# Data Requirements: Direction 2 — PCA Factor Decomposition (BDF) for Intraday Volume

## Status: partially matched

## Requirements

### 1. Intraday Volume Bars
- **Needed for:** Phase 1 (Data Preparation) — constructing the turnover matrix X of shape (P, N) where P = L * k. Volume is aggregated into k intraday bins per day across L rolling estimation days for N stocks. Used by all subsequent phases (PCA extraction, component forecasting, dynamic VWAP execution).
- **Data type:** OHLCV bars (only the volume field is needed, but OHLC prices are useful for validation and benchmarking VWAP execution cost)
- **Granularity:** 1-minute bars. The algorithm operates on 15-minute bins (k=26 per day), but 1-minute is the finest bar granularity available and will be aggregated to 15-minute in Step 3. This also allows experimenting with other bin sizes (k=13 for 30-min, k=78 for 5-min) without re-downloading.
- **Instruments:** 30-40 liquid US equities. The spec recommends N >= 30 for reliable PCA factor extraction (Bai 2003). Szucs 2017 used 33 DJIA components; BDF 2008 used 39 CAC40 stocks. Recommended universe: current DJIA components or a comparable set of highly liquid large-cap stocks.
- **Date range:** Minimum 1 year for meaningful train/validate/test splits. Recommended: 2023-01-01 to 2025-12-31 (3 years). This provides:
  - Training: 2023-01-01 to 2024-06-30 (~375 trading days)
  - Validation: 2024-07-01 to 2024-12-31 (~125 trading days)
  - Out-of-sample test: 2025-01-01 to 2025-12-31 (~250 trading days)
  - The 20-day rolling estimation window (L=20) consumes the first 20 days of each period for warm-up.
- **Fields needed:** timestamp, volume (required); open, high, low, close (useful for VWAP execution cost computation)
- **Source:** Databento
- **Dataset:** EQUS.MINI — composite across all venues, no exchange license fees. This is the cheapest option that provides OHLCV bars with consolidated volume across all exchanges and TRFs.
- **Schema:** ohlcv-1m (1-minute OHLCV bars)
- **Notes:** EQUS.MINI provides consolidated volume across all 16 exchanges and 30+ ATSs, which is what the algorithm needs (total market volume per bin, not per-venue volume). Individual exchange feeds (e.g., XNAS.ITCH) would only give single-venue volume and would need to be summed — more expensive and complex. Half-days (Dec 24, Jul 3, etc.) should be identified and excluded per BDF 2008 Section 3.1.

### 2. Shares Outstanding (Total Shares Outstanding / TSO)
- **Needed for:** Phase 1 (Data Preparation) — computing turnover as volume / shares_outstanding for each stock. Turnover normalization is essential for making the cross-sectional PCA meaningful, since raw volume varies by orders of magnitude across stocks.
- **Data type:** Reference / fundamental data — point-in-time total shares outstanding per stock
- **Granularity:** Daily (or less frequent). TSO changes infrequently (corporate actions: stock splits, buybacks, secondary offerings). A daily or monthly series is sufficient.
- **Instruments:** Same universe as Requirement 1 (30-40 liquid US equities)
- **Date range:** Same as Requirement 1 (2023-01-01 to 2025-12-31)
- **Fields needed:** date, ticker/symbol, total shares outstanding
- **Source:** UNFULFILLED — Databento provides instrument reference data via the Definition schema, but this covers symbology, tick size, and lot size — not shares outstanding or float. TSO is a fundamental data field typically sourced from:
  - Bloomberg (Szucs 2017 used Bloomberg TSO)
  - SEC EDGAR filings (quarterly, with lag)
  - Commercial fundamental data providers (Sharadar, Polygon, etc.)
- **Dataset:** N/A
- **Schema:** N/A
- **Notes:** Without TSO, we cannot compute turnover directly. Workarounds:
  1. **Use average daily volume (ADV) normalization instead of TSO.** Normalize each stock's bin volume by its trailing N-day average daily volume. This produces a similar cross-sectional normalization effect. Not exactly what BDF/Szucs specify, but achieves the same purpose of making stocks comparable for PCA.
  2. **Use a static TSO snapshot.** For large-cap DJIA stocks, TSO changes slowly. A single recent TSO value per stock (obtainable from public sources like Yahoo Finance or SEC filings) could be used as an approximation across the full sample period. Accuracy degrades around stock splits and major buybacks.
  3. **Source TSO from a free/cheap provider.** If a fundamental data source becomes available, this requirement can be fulfilled directly.
  The ADV workaround (option 1) is recommended as the primary approach since it can be computed entirely from the volume data in Requirement 1, requires no additional data source, and produces a normalization that is functionally equivalent for the PCA decomposition.

## Gaps

### Shares Outstanding (TSO)
Databento does not provide shares outstanding or float data. This is fundamental/reference data outside the scope of market data providers. The impl spec explicitly calls for turnover = volume / shares_outstanding (BDF 2008 uses float; Szucs 2017 uses TSO from Bloomberg).

**Impact:** Without TSO, turnover cannot be computed exactly as specified. However, the ADV normalization workaround described above is a practical substitute that preserves the model's cross-sectional comparability requirement. The PCA factor structure should be qualitatively identical since ADV normalization and TSO normalization both remove the stock-level scale differences that would otherwise dominate the first principal component.

**Recommendation:** Proceed with ADV normalization for initial implementation. If a fundamental data source is added to the registry in the future, TSO-based turnover can be substituted as a direct improvement.

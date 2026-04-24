# Data Requirements: Direction 7 — Kalman Filter State-Space Model for Intraday Volume

## Status: partially matched

## Requirements

### 1. Intraday Volume Bars (Primary Training and Prediction Data)
- **Needed for:** Core model input. The Kalman filter operates on log-transformed, normalized intraday volume bins. Used for EM parameter estimation (training), dynamic/static prediction (out-of-sample), and VWAP weight computation.
- **Data type:** OHLCV bars (only the volume field is used)
- **Granularity:** 15-minute bars (I=26 bins per 6.5-hour NYSE session). The paper uses 15-minute bins explicitly (Section 4.1, Table 2). Other bin widths could work but 15-min is the reference.
- **Instruments:** Liquid US equities and ETFs. The paper uses 10 securities: SPY, DIA, QQQ, AAPL, AMZN, GOOG, IBM, JPM, MSFT, XOM (Table 2). For replication, use at minimum SPY, DIA, and IBM (the three used in the robustness simulation, Table 1). For full replication, all 10.
- **Date range:** The paper uses June 2014 to June 2016 (approximately 2 years). Cross-validation period: January 2015 to May 2015 (5 months). Out-of-sample: June 2015 to June 2016 (250 trading days). Training windows of up to 24 months are tested, so data from at least January 2014 is needed to support the longest window for the earliest CV day. Recommended: 2014-01-01 to 2016-06-30.
- **Fields needed:** timestamp, volume. Open/high/low/close are available but not used by the model.
- **Source:** Databento
- **Dataset:** EQUS.MINI (composite across all venues, no exchange license fees; adequate since only volume aggregates are needed, not venue-level detail)
- **Schema:** ohlcv-1m (1-minute bars). The model needs 15-minute bars, but downloading 1-minute bars and aggregating to 15 minutes in Step 3 provides flexibility to test other bin widths without re-downloading. 1-minute is the finest OHLCV granularity that is practical for multi-year downloads.
- **Notes:** EQUS.MINI is preferred over individual exchange feeds because the model uses consolidated volume across all venues. The 1-minute granularity is downloaded rather than ohlcv-1h because 1-hour bars cannot be aggregated down to 15-minute bins. Half-day sessions (e.g., day before Thanksgiving, Christmas Eve) should be identified during preparation and excluded per the paper's methodology.

### 2. Shares Outstanding (for Volume Normalization)
- **Needed for:** Step 1 of the algorithm (Data Preprocessing). Raw volume is normalized by shares outstanding before log transformation: vol_norm[t,i] = raw_volume[t,i] / shares_outstanding[t]. This handles stock splits and cross-security scale differences.
- **Data type:** Daily reference data (shares outstanding per security per day)
- **Granularity:** Daily (one value per security per trading day; constant within a day)
- **Instruments:** Same as Requirement 1 (SPY, DIA, QQQ, AAPL, AMZN, GOOG, IBM, JPM, MSFT, XOM)
- **Date range:** 2014-01-01 to 2016-06-30
- **Fields needed:** date, symbol, shares outstanding
- **Source:** UNFULFILLED
- **Dataset:** N/A
- **Schema:** N/A
- **Notes:** Databento does not provide shares outstanding or fundamental reference data. The Definition schema provides instrument reference data (tick size, lot size, listing exchange) but not shares outstanding. This is a gap. Possible alternatives: (a) Use a fundamental data provider (e.g., WRDS/CRSP, Sharadar, Polygon.io). (b) As a practical workaround for replication, use average daily volume (ADV) as a normalizer instead of shares outstanding -- this changes the interpretation slightly but achieves the same scale normalization. (c) For ETFs like SPY/DIA/QQQ, shares outstanding changes with creation/redemption and is available from fund providers. For the 2014-2016 period, this data may be available from free sources or could be approximated.

### 3. Trading Calendar and Session Hours
- **Needed for:** Identifying trading days, half-day sessions (to exclude), and confirming session length (6.5 hours = 26 bins at 15-min). Also needed for the rolling window logic to correctly count T training days.
- **Data type:** Calendar/reference data
- **Granularity:** Daily
- **Instruments:** NYSE/Nasdaq calendar (covers all 10 securities)
- **Date range:** 2014-01-01 to 2016-06-30
- **Fields needed:** trading dates, early close dates
- **Source:** UNFULFILLED (as standalone data)
- **Dataset:** N/A
- **Schema:** N/A
- **Notes:** This is implicitly available from the volume data itself -- days with no bars are non-trading days, and days with fewer than 26 fifteen-minute bins (or fewer than 390 one-minute bars in regular hours) are half-day sessions. No separate download is needed; this will be derived from the volume data during Step 3 preparation. Alternatively, the pandas_market_calendars or exchange_calendars Python package provides this directly.

### 4. Daily OHLCV (for Validation Benchmarks)
- **Needed for:** Computing the rolling mean baseline that the Kalman filter is compared against (Sanity Check 5: "average of same bin over the last T_train days"). Also useful for sanity-checking total daily volume against the sum of intraday bins.
- **Data type:** OHLCV daily bars
- **Granularity:** Daily (1d)
- **Instruments:** Same as Requirement 1
- **Date range:** 2014-01-01 to 2016-06-30
- **Fields needed:** date, symbol, volume, close (close useful for context)
- **Source:** Databento
- **Dataset:** EQUS.MINI
- **Schema:** ohlcv-1d
- **Notes:** This is a nice-to-have for validation, not strictly required. The daily volume can also be computed by summing intraday bars from Requirement 1. Low cost and small data volume make it worth downloading as a cross-check.

## Gaps

### Gap 1: Shares Outstanding Data (Requirement 2)

Databento does not provide shares outstanding. This is needed for the volume normalization step specified in the impl spec (vol_norm = raw_volume / shares_outstanding).

**Impact:** Without shares outstanding, the normalization step cannot be performed exactly as specified. This affects the absolute scale of log-volume but does not affect the model's ability to capture intraday patterns, since shares outstanding is constant within a day and changes slowly across days.

**Workarounds (in order of preference):**
1. **Use average daily volume (ADV) as normalizer.** Replace shares_outstanding[t] with a rolling average of total daily volume (e.g., 60-day trailing average). This achieves the same goal of removing slow scale changes and is commonly used in practice. The daily component eta_t absorbs the remaining day-to-day variation.
2. **Use a constant normalizer per security.** Divide by the median daily volume over the full sample. This is the simplest approach but does not handle splits during the sample period.
3. **Source shares outstanding from an external provider.** If WRDS/CRSP access is available, CRSP daily stock files include shares outstanding (SHROUT field). Polygon.io also provides this via their reference data API.
4. **Skip normalization entirely for ETFs.** For SPY, DIA, QQQ, the volume scale is relatively stable over the 2014-2016 period. The daily component eta_t can absorb gradual level changes without explicit normalization. This may slightly slow EM convergence but the model is robust to this (Paper Figure 4 shows convergence from various initializations).

**Recommendation:** Use workaround 1 (ADV normalization) for the initial implementation. This is fully derivable from the volume data in Requirement 1 and requires no additional data source. Note this deviation from the paper in the preparation report.

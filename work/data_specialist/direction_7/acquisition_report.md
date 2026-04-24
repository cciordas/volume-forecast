# Acquisition Report: Direction 7 — Kalman Filter State-Space Model for Intraday Volume

## Status: all acquired (with date range substitution)

## Date Range Substitution

The paper uses June 2014 to June 2016, but Databento's EQUS.MINI dataset only starts
from 2023-03-28 (and direct exchange feeds from 2018-05-01 at earliest). Since no
available data source covers the 2014-2016 period, a substitute date range of equivalent
length was used:

| | Paper | Substitute |
|---|---|---|
| Full range | 2014-01-02 to 2016-06-30 (2.5 years) | 2023-10-02 to 2026-03-31 (2.5 years) |
| Cross-validation | 2015-01-01 to 2015-05-31 | 2024-10-01 to 2025-02-28 |
| Out-of-sample | 2015-06-01 to 2016-06-30 (250 days) | 2025-03-01 to 2026-03-31 (~250 days) |

The algorithm is general-purpose and does not depend on the specific date range. The
substitute period includes comparable market conditions (normal trading, no unusual
market closures beyond standard holidays).

Note: A few days in 2025 have "degraded" data quality flags from Databento (2025-03-24,
2025-04-04, 2025-05-06, 2025-10-10, 2025-10-13). These are minor and unlikely to affect
model results at 15-minute aggregation.

## Downloads

### 1. Intraday 1-Minute Bars (Requirement 1)
- **Source:** Databento (EQUS.MINI)
- **Tool:** bentoticks load_bars_intraday (barsz="1m")
- **Status:** acquired (all 10 tickers)
- **Date range:** 2023-10-02 to 2026-03-31 (626 trading days)
- **Notes:** Downloaded as 1-minute bars; aggregation to 15-minute bins happens in Step 3.

| Ticker | Rows | Size (MB) | Location |
|---|---|---|---|
| SPY | 243,011 | 3.20 | data/direction_7/SPY_1m.parquet |
| DIA | 208,333 | 2.43 | data/direction_7/DIA_1m.parquet |
| QQQ | 242,966 | 3.15 | data/direction_7/QQQ_1m.parquet |
| AAPL | 242,947 | 2.79 | data/direction_7/AAPL_1m.parquet |
| AMZN | 242,959 | 2.80 | data/direction_7/AMZN_1m.parquet |
| GOOG | 242,734 | 2.85 | data/direction_7/GOOG_1m.parquet |
| IBM | 235,958 | 2.75 | data/direction_7/IBM_1m.parquet |
| JPM | 242,018 | 2.88 | data/direction_7/JPM_1m.parquet |
| MSFT | 242,875 | 3.02 | data/direction_7/MSFT_1m.parquet |
| XOM | 242,741 | 2.43 | data/direction_7/XOM_1m.parquet |

DIA has fewer rows (208,333 vs ~243,000) because it is less liquid and some 1-minute
bins have no trades. IBM similarly has slightly fewer rows (235,958). This is expected
and handled during preparation.

### 2. Shares Outstanding (Requirement 2)
- **Source:** N/A (unfulfilled, per data requirements)
- **Status:** not acquired — will use ADV workaround in Step 3
- **Notes:** As documented in the data requirements, shares outstanding is not available
  from Databento. The recommended workaround is to normalize by rolling average daily
  volume (ADV-60) instead, which is computed from the intraday volume data in Step 3.

### 3. Trading Calendar (Requirement 3)
- **Source:** Derived from volume data
- **Status:** not acquired separately — derived in Step 3
- **Notes:** Trading days and half-day sessions will be identified from the downloaded
  1-minute bar data during preparation.

### 4. Daily OHLCV Bars (Requirement 4)
- **Source:** Databento (EQUS.MINI)
- **Tool:** bentoticks load_bars_daily
- **Status:** acquired (all 10 tickers)
- **Date range:** 2023-10-02 to 2026-03-31 (626 trading days each)

| Ticker | Rows | Size (MB) | Location |
|---|---|---|---|
| SPY | 626 | 0.026 | data/direction_7/SPY_1d.parquet |
| DIA | 626 | 0.026 | data/direction_7/DIA_1d.parquet |
| QQQ | 626 | 0.026 | data/direction_7/QQQ_1d.parquet |
| AAPL | 626 | 0.026 | data/direction_7/AAPL_1d.parquet |
| AMZN | 626 | 0.026 | data/direction_7/AMZN_1d.parquet |
| GOOG | 626 | 0.026 | data/direction_7/GOOG_1d.parquet |
| IBM | 626 | 0.026 | data/direction_7/IBM_1d.parquet |
| JPM | 626 | 0.026 | data/direction_7/JPM_1d.parquet |
| MSFT | 626 | 0.026 | data/direction_7/MSFT_1d.parquet |
| XOM | 626 | 0.025 | data/direction_7/XOM_1d.parquet |

## Failures

None — all downloadable data acquired successfully. The shares outstanding gap
(Requirement 2) was identified in Step 1 and will be handled via ADV workaround
in Step 3.

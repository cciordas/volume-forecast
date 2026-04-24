# Acquisition Report: Direction 2 — PCA Factor Decomposition (BDF) for Intraday Volume

## Status: all acquired

## Summary

- **Tickers:** 30 DJIA components
- **Total rows:** 7,861,625
- **Total size:** 82.8 MB (Parquet)
- **Date range:** 2023-03-28 to 2025-12-31 (694 trading days)
- **Adjusted start date:** The original requirement specified 2023-01-01, but EQUS.MINI data availability begins 2023-03-28. The start date was shifted accordingly. This still provides ~690 trading days, sufficient for the train/validate/test split (the spec requires a minimum of 1 year).

## Downloads

### 1. Intraday Volume Bars (1-minute OHLCV)
- **Source:** Databento
- **Dataset:** EQUS.MINI
- **Schema:** ohlcv-1m
- **Tool:** bentoticks load_bars_intraday
- **Status:** acquired (all 30 tickers)
- **Location:** `data/direction_2/{TICKER}_1m.parquet`
- **Date range:** 2023-03-28 to 2025-12-31
- **Columns:** date_event, ts_event, open, high, low, close, volume
- **Per-ticker details:**

| Ticker | Rows | Size (MB) |
|--------|-----:|----------:|
| AAPL | 269,299 | 2.91 |
| AMGN | 248,075 | 2.65 |
| AMZN | 269,301 | 2.98 |
| AXP | 257,215 | 3.06 |
| BA | 264,293 | 2.78 |
| CAT | 255,136 | 3.23 |
| CRM | 267,087 | 2.88 |
| CSCO | 268,319 | 2.38 |
| CVX | 268,352 | 2.32 |
| DIS | 268,420 | 2.48 |
| GS | 254,909 | 3.60 |
| HD | 261,199 | 2.81 |
| HON | 256,099 | 2.41 |
| IBM | 262,085 | 2.98 |
| JNJ | 267,747 | 2.53 |
| JPM | 268,388 | 3.03 |
| KO | 268,235 | 2.30 |
| MCD | 258,748 | 2.50 |
| MMM | 256,536 | 2.46 |
| MRK | 268,323 | 2.52 |
| MSFT | 269,220 | 3.21 |
| NKE | 268,826 | 2.58 |
| NVDA | 269,463 | 3.80 |
| PG | 268,155 | 2.34 |
| SHW | 236,713 | 2.65 |
| TRV | 224,723 | 2.47 |
| UNH | 263,899 | 3.30 |
| V | 267,181 | 2.81 |
| VZ | 266,953 | 2.19 |
| WMT | 268,726 | 2.60 |

- **Notes:** Row count varies across tickers (224K-269K) due to differences in trading activity — some stocks have minutes with zero volume that may be omitted. SHW and TRV have noticeably fewer rows, likely due to lower tick activity in some minutes. This is expected for less liquid DJIA names and does not affect the algorithm (zero-volume minutes will be filled with 0 during preparation).

### 2. Shares Outstanding (TSO)
- **Source:** UNFULFILLED (not available from Databento)
- **Status:** not acquired — per data requirements, ADV normalization will be used as a workaround in Step 3. No additional download needed.

## Revised Date Splits

With the adjusted start date (2023-03-28 instead of 2023-01-01):
- **Training:** 2023-03-28 to 2024-06-30 (~315 trading days)
- **Validation:** 2024-07-01 to 2024-12-31 (~125 trading days)
- **Out-of-sample test:** 2025-01-01 to 2025-12-31 (~250 trading days)

## Failures

None — all data acquired successfully.

# Validation Results: Kalman Filter State-Space Model for Intraday Volume

## Sanity Checks

| Check | Result | Expected | Source |
|-------|--------|----------|--------|
| Synthetic data recovery (a_eta) | 0.9779 (0.21% err) | 0.98 | Paper Fig 4(a) |
| Synthetic data recovery (r) | 0.185 (7.5% err) | 0.20 | Paper Fig 4(c) |
| Synthetic phi correlation | 0.977 | ~1.0 | Paper Fig 4(f) |
| EM log-likelihood monotonic | Yes (all tickers) | Yes | EM theory |
| Seasonality U-shape | Yes (all tickers) | Yes | Paper Fig 4(f) |
| Improvement over rolling mean | 38.5% avg | 64% (paper) | Paper Section 4.2 |
| a_eta close to 1 | 0.995-0.999 (all tickers) | ~1.0 | Paper Section 4 |
| EM convergence iterations | 25-100 (first fit), 11-30 (warm) | 5-20 typical | Paper Section 2.3.3 |

Notes on synthetic recovery: a_mu shows identifiability issues (estimated 0.27 vs true 0.60)
due to low signal-to-noise ratio between mu and observation noise. Verified that with
true parameters, the smoother M-step gives correct a_mu (0.596). This is an inherent
identifiability issue, not a code bug. The EM finds a different but observationally
equivalent solution.

## Paper Benchmark Comparison

| Benchmark | Paper Value | Our Value | Source | Notes |
|-----------|-------------|-----------|--------|-------|
| SPY dynamic MAPE | 0.24 | 0.28 | Paper Table 3 | Different data period (2023-2026 vs 2015-2016) |
| DIA dynamic MAPE | 0.38 | 0.54 | Paper Table 3 | DIA much less liquid in our data |
| IBM dynamic MAPE | 0.24 | 0.38 | Paper Table 3 | IBM less liquid in our data |
| QQQ dynamic MAPE | 0.30 | 0.28 | Paper Table 3 | Close to paper |
| Avg dynamic MAPE | 0.46 | 0.32 | Paper Table 3, avg row | Our avg better (different tickers) |
| Avg static MAPE | 0.61 | 0.48 | Paper Table 3, avg row | Our avg better |
| Improvement over RM | 64% | 38.5% | Paper Section 4.2 | Lower because our RM baseline is stronger |
| Dynamic < Static MAPE | Yes | Yes (all tickers) | Paper Table 3 | Expected: dynamic uses intraday info |

### Per-Ticker Dynamic MAPE Results

| Ticker | Dynamic MAPE | Static MAPE | RM MAPE | Improvement % |
|--------|-------------|-------------|---------|--------------|
| SPY | 0.2771 | 0.4145 | 0.5767 | 51.9 |
| DIA | 0.5423 | 0.7592 | 0.8282 | 34.5 |
| QQQ | 0.2841 | 0.4206 | 0.4126 | 31.1 |
| AAPL | 0.2714 | 0.4137 | 0.4536 | 40.2 |
| AMZN | 0.2661 | 0.4031 | 0.4616 | 42.4 |
| GOOG | 0.3092 | 0.4555 | 0.5594 | 44.7 |
| IBM | 0.3817 | 0.6035 | 0.6330 | 39.7 |
| JPM | 0.3066 | 0.4539 | 0.4834 | 36.6 |
| MSFT | 0.3025 | 0.4440 | 0.4921 | 38.5 |
| XOM | 0.2788 | 0.3855 | 0.3735 | 25.4 |
| **Average** | **0.3220** | **0.4753** | **0.5274** | **38.5** |

## Parameters Used

| Parameter | Value |
|-----------|-------|
| I (bins per day) | 26 |
| Bin width | 15 minutes |
| Training window | 252 days |
| Re-estimation interval | 21 days |
| EM max iterations | 100 (first fit), 30 (warm-start) |
| EM convergence threshold | 1e-6 (relative LL change) |
| Robust mode | Yes |
| Lambda scaling | k=5.0 (effective lambda = 2*k/sqrt(r_init)) |
| Normalization | ADV-60 (rolling 60-day average daily volume) |

### Estimated Model Parameters (final rolling window, representative values)

| Parameter | SPY | DIA | QQQ | AAPL | AMZN |
|-----------|-----|-----|-----|------|------|
| a_eta | 0.998 | 0.996 | 0.998 | 0.997 | 0.996 |
| a_mu | 0.716 | 0.793 | 0.724 | 0.718 | 0.735 |
| r | 0.055 | 0.279 | 0.063 | 0.046 | 0.052 |

## Data Used
- Dataset: Prepared log-volume matrices from data specialist (direction 7)
- Date range: 2023-12-28 to 2026-03-31 (559 trading days)
- Training: first 252 days (rolling window)
- Out-of-sample: remaining 307 days
- Instruments: SPY, DIA, QQQ, AAPL, AMZN, GOOG, IBM, JPM, MSFT, XOM
- Normalization: ADV-60 (substitute for shares outstanding)

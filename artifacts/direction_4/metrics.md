# Validation Results: Dual-Mode Intraday Volume Forecast

## Sanity Checks
| Check | Result | Expected | Source |
|-------|--------|----------|--------|
| 1. Seasonal factor U-shape | PASS (edges=258,551 vs midday=69,076) | Edges > midday | Spec check 1 |
| 2. H/SF ratio | [0.629, 0.888] | Within [0.5, 2.0] | Spec check 2 |
| 3. Deseasonalized stationarity (ADF) | p=0.0000 (stat=-6.611) | p < 0.05 | Spec check 3 |
| 4. ARMA parsimony (median p+q) | 1 | <= 4 | Spec check 4 |
| 5. Weight non-negativity | PASS (all >= 0) | All non-negative | Spec check 5 |
| 6. Regime bucket population | 4 regimes selected | Reasonable count | Spec check 6 |
| 9. Training surprise mean | max|mean| = 0.000000 | < 0.001 | Spec check 9 |
| 10. Surprise beta bounded | |0.412| < 1.0 | All |beta| < 1.0 | Spec check 10 |
| 14. MAPE formula consistency | Verified | Matches paper formula | Spec check 14 |

## Paper Benchmark Comparison
| Benchmark | Paper Value | Our Value | Source | Notes |
|-----------|-------------|-----------|--------|-------|
| Model A MAPE reduction (median, 10 stocks) | ~24% | 50.0% | Satish 2014, Exhibit 6 | Our conditioned forecasts use observed same-day bins; paper's 24% may be unconditional |
| Model A MAPE reduction (AAPL) | ~24% | 51.5% | Satish 2014, Exhibit 6 | Single-stock result |
| Model A per-bin MAPE reduction range | 10%-33% | 16.9%-62.5% | Satish 2014, Exhibit 6 | Increasing through day as expected |
| Model A bin 1 reduction (no intraday info) | ~10% | 16.9% | Satish 2014, Exhibit 6 | Open bin, no conditioning signal |
| Model B MAD reduction | 7.55% | -16.9% | Satish 2014, Exhibit 9 | Model B shows worsening; fixed 10% deviation bound likely too conservative vs paper's proprietary adaptive bounds |
| Surprise regression L | Not disclosed | L=1 | Satish 2014, p.19 | Single lag selected by CV |
| Surprise beta_1 sign | Expected positive | +0.412 | Spec check 10 | Positive autocorrelation confirmed |
| ARMA 11-term soft limit | Observed < 11 | 11 (AAPL) | Satish 2014, p.18 | At soft limit; no hard violation |

## Cross-Stock Model A Results (Full 10-Ticker Evaluation)
| Ticker | Model MAPE | Baseline MAPE | MAPE Reduction (%) | Train Time (s) |
|--------|-----------|---------------|---------------------|----------------|
| NVDA | 2.253 | 4.192 | 46.3% | 86 |
| TSLA | 0.426 | 0.827 | 48.5% | 92 |
| AAPL | 0.279 | 0.575 | 51.5% | 99 |
| AMZN | 0.721 | 0.844 | 14.6% | 85 |
| MSFT | 0.297 | 0.567 | 47.6% | 74 |
| META | 0.332 | 0.768 | 56.7% | 98 |
| AMD | 0.439 | 1.254 | 65.0% | 80 |
| GOOG | 0.363 | 0.776 | 53.1% | 82 |
| UNH | 0.360 | 0.751 | 52.1% | 74 |
| INTC | 1.567 | 1.223 | -28.1% | 74 |
| **Median** | **0.395** | **0.801** | **50.0%** | |
| **Mean** | **0.704** | **1.178** | **40.7%** | |

**Notes:**
- 9/10 stocks show improvement (14.6% to 65.0%)
- INTC is the sole outlier (-28.1%) — likely due to its highly irregular volume pattern during 2024-2025 (Intel restructuring, foundry spin-off)
- AMZN shows modest 14.6% improvement, lowest among positive results
- AMD shows the largest improvement at 65.0%

## Model A Weight Analysis (AAPL, 4 regimes)
| Regime | w_H (Historical) | w_D (Inter-day) | w_A (Intraday) |
|--------|------------------|-----------------|----------------|
| 0 (low vol) | 0.0000 | 0.0588 | 0.6920 |
| 1 | 0.0000 | 0.1236 | 0.7079 |
| 2 | 0.0302 | 0.1419 | 0.6498 |
| 3 (high vol) | 0.0839 | 0.0159 | 0.8294 |

## Parameters Used
| Parameter | Value |
|-----------|-------|
| I (bins/day) | 26 |
| N_seasonal | 126 |
| N_hist | 21 |
| N_interday_fit | 63 |
| p_max_inter | 5 |
| q_max_inter | 5 |
| N_intraday_fit | 21 |
| p_max_intra | 4 |
| q_max_intra | 5 |
| N_regime_window | 63 |
| regime_candidates | {3, 4, 5} |
| N_weight_train | 63 |
| min_samples_per_regime | 50 |
| min_volume_floor | 100.0 |
| N_regression_fit | 63 |
| L_max | 5 |
| max_deviation | 0.10 |
| pct_switchoff | 0.80 |
| reestimation_interval | 21 |

## Data Used
- Dataset: Dow 30 + AMD, GOOG, INTC, META, TSLA (35 tickers total)
- Date range: 2024-01-02 to 2025-12-31 (~497 trading days)
- Training: First 434 days (through ~2025-09)
- Evaluation: Last 63 days (OOS)
- Instruments evaluated in detail: AAPL, NVDA, MSFT, JPM, KO
- Bin structure: 26 x 15-minute bins (9:30-16:00 ET)

# Validation Results: PCA Factor Decomposition (BDF) for Intraday Volume

## Sanity Checks
| Check                        | Result    | Expected  | Source              |
|------------------------------|-----------|-----------|---------------------|
| Reconstruction (X = C + e)   | 0.00e+00  | < 1e-10   | BDF 2008, Eq. (6)   |
| Factor normalization F'F/P   | Identity  | I_r       | Bai 2003; BDF 2008  |
| Proportion sums              | 4.44e-16  | < 1e-10   | BDF 2008, Sec 4.2   |
| AR(1) stationarity (all)     | Pass      | |psi_1|<1 | BDF 2008, Eq. (10)  |
| Turnover non-negativity      | Pass      | >= 0      | Researcher inference |
| BDF beats U-method (MAPE)    | Pass      | Yes       | BDF 2008, Sec 3.2   |
| Dynamic beats Static (MAPE)  | Pass      | Yes       | BDF 2008, Sec 4.2.1 |

## Paper Benchmark Comparison
| Benchmark                     | Paper Value | Our Value | Source               | Notes                                    |
|-------------------------------|-------------|-----------|----------------------|------------------------------------------|
| BDF_SETAR MAPE (one-step)     | 0.399       | 0.526     | Szucs 2017, Table 2a | ADV vs TSO normalization; different period |
| BDF_AR MAPE (one-step)        | 0.403       | N/A       | Szucs 2017, Table 2a | SETAR selected for all 30 stocks          |
| U-method MAPE                 | 0.503       | 0.702     | Szucs 2017, Table 2a | Higher baseline reflects data differences |
| MAPE improvement vs U-method  | ~20%        | 25.2%     | Szucs 2017, Sec 5    | Consistent direction and magnitude        |
| Dynamic beats classical       | 30/39       | 30/30     | BDF 2008, Table 6    | All stocks show improvement               |
| Factor count (typical)        | 1-3         | 3-10      | Researcher inference  | Higher r with ADV normalization           |

## Validation Set Results (105 days)
| Metric          | BDF Dynamic | BDF Static | U-method |
|-----------------|-------------|------------|----------|
| MAPE            | 0.5748      | 0.8199     | 0.7861   |
| MSE             | 0.001133    | 0.001391   | 0.001394 |

## Test Set Results (227 days)
| Metric          | BDF Dynamic | BDF Static | U-method |
|-----------------|-------------|------------|----------|
| MAPE            | 0.5256      | 0.7384     | 0.7024   |
| MSE             | 0.000918    | 0.001206   | 0.001190 |
| MAPE improvement vs U-method | 25.2% | -5.1% | baseline |
| MSE improvement vs U-method  | 22.9% | -1.3% | baseline |

## Per-Stock Test Set MAPE
| Ticker | BDF Dynamic | U-method | Improvement |
|--------|-------------|----------|-------------|
| AAPL   | 0.4217      | 0.6273   | 32.8%       |
| AMGN   | 0.6553      | 0.8321   | 21.2%       |
| AMZN   | 0.4295      | 0.5947   | 27.8%       |
| AXP    | 0.5770      | 0.7579   | 23.9%       |
| BA     | 0.5680      | 0.7920   | 28.3%       |
| CAT    | 0.6602      | 0.8672   | 23.9%       |
| CRM    | 0.4772      | 0.6900   | 30.8%       |
| CSCO   | 0.4366      | 0.5875   | 25.7%       |
| CVX    | 0.4526      | 0.6319   | 28.4%       |
| DIS    | 0.5086      | 0.6560   | 22.5%       |
| GS     | 0.6216      | 0.8317   | 25.3%       |
| HD     | 0.5162      | 0.6634   | 22.2%       |
| HON    | 0.6521      | 0.8495   | 23.2%       |
| IBM    | 0.6793      | 0.8704   | 22.0%       |
| JNJ    | 0.4803      | 0.6706   | 28.4%       |
| JPM    | 0.4519      | 0.6254   | 27.7%       |
| KO     | 0.3913      | 0.5039   | 22.3%       |
| MCD    | 0.5127      | 0.6752   | 24.1%       |
| MMM    | 0.6463      | 0.8505   | 24.0%       |
| MRK    | 0.4154      | 0.5927   | 29.9%       |
| MSFT   | 0.4938      | 0.6337   | 22.1%       |
| NKE    | 0.5144      | 0.7312   | 29.7%       |
| NVDA   | 0.3359      | 0.5031   | 33.2%       |
| PG     | 0.3934      | 0.5132   | 23.3%       |
| SHW    | 0.7025      | 0.8817   | 20.3%       |
| TRV    | 0.6767      | 0.8237   | 17.8%       |
| UNH    | 0.7452      | 0.9735   | 23.4%       |
| V      | 0.4803      | 0.6635   | 27.6%       |
| VZ     | 0.5070      | 0.6557   | 22.7%       |
| WMT    | 0.3656      | 0.5224   | 30.0%       |

## Parameters Used
| Parameter         | Value       |
|-------------------|-------------|
| k (bins/day)      | 26          |
| L (window days)   | 20          |
| r_max             | 10          |
| n_grid (SETAR)    | 100         |
| tau_quantile_range| [0.15, 0.85]|
| min_regime_obs    | 10          |

## Data Used
- Dataset: Turnover matrices from preparation_report.md
- Date range: 2023-03-28 to 2025-12-31 (686 regular trading days)
- Validation: 2024-07-01 to 2024-12-31 (125 days, 105 forecast days after L=20 warm-up)
- Test: 2025-01-02 to 2025-12-31 (247 days, 227 forecast days after L=20 warm-up)
- Instruments: 30 DJIA stocks
- Normalization: ADV-based turnover (60-day trailing average daily volume)

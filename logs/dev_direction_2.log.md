# Developer Log: Direction 2 — PCA Factor Decomposition (BDF) for Intraday Volume

## Status: done

## Current Step: Complete

### [2026-04-14 09:30] Specification Review
- Read impl_spec.md and preparation_report.md in full.
- Spec is clear and complete. All pseudocode steps can be directly translated.
- No ambiguities found. Proceeding to implementation.
- Data: 30 DJIA stocks, 686 trading days, 26 bins/day (15-min), turnover matrices ready.
- Key parameters: L=20 (window), k=26 (bins), r_max=10, n_grid=100.

### [2026-04-14 09:35] Implementation
- Wrote bdf_model.py with BDFModel class implementing:
  - extract_factors(): truncated SVD with IC_p2 factor selection
  - forecast_common(): time-of-day averaging of common component
  - fit_ar1() / fit_setar(): specific component modeling
  - run_daily_pipeline(): full end-to-end daily pipeline
  - dynamic_vwap_execution(): bin-by-bin execution simulation
  - dynamic_one_step_ahead(): evaluation mode matching Szucs 2017
  - compute_u_method(), compute_mape(), compute_mse(): benchmarks and metrics
- Wrote exploration.ipynb with 8 sections covering data exploration through evaluation.

### [2026-04-14 09:45] Notebook Validation
- Executed exploration.ipynb via nbconvert — completed successfully, no errors.

### [2026-04-14 09:45] Execution and Validation Results
- **Sanity Checks — ALL PASS:**
  - Reconstruction error: 0.00e+00 (< 1e-10)
  - Factor normalization F'F/P = I_r: verified (identity matrix)
  - Proportion sums: max deviation 4.44e-16 (machine precision)
  - AR(1) stationarity: all stocks have |psi_1| < 1
  - SETAR selected for all 30 stocks (as expected per BDF 2008 — SETAR outperforms)
  - All turnover forecasts non-negative after flooring

- **Validation Set (105 days):**
  - BDF Dynamic MAPE: 0.5748, MSE: 0.001133
  - BDF Static MAPE: 0.8199, MSE: 0.001391
  - U-method MAPE: 0.7861, MSE: 0.001394
  - Dynamic beats U-method by ~27% MAPE

- **Test Set (227 days):**
  - BDF Dynamic MAPE: 0.5256, MSE: 0.000918
  - BDF Static MAPE: 0.7384, MSE: 0.001206
  - U-method MAPE: 0.7024, MSE: 0.001190
  - MAPE improvement over U-method: 25.2%
  - MSE improvement over U-method: 22.9%
  - All 30 stocks show improvement (17.8% to 33.2%)

- **Factor count:** IC_p2 selects r=3-10, averaging ~7.6 on validation and varying on test.
  Higher than the "typically 1-3" researcher inference, likely because our ADV-normalized
  turnover has more cross-sectional variation than TSO-normalized turnover.

- **Paper benchmark comparison:**
  - Szucs 2017 reports BDF_SETAR MAPE 0.399 on DJIA. Our 0.526 is higher, but we use
    ADV normalization instead of TSO and different data period (2023-2025 vs 2000-2010).
  - The ~25% MAPE improvement over U-method matches the ~20% reported by Szucs 2017.
  - BDF Dynamic clearly beats BDF Static, consistent with BDF 2008 Section 4.2.1.

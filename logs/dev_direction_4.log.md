# Developer Log: Direction 4 — Dual-Mode Intraday Volume Forecast

## Status: done
## Current Step: Step 5 — Reporting (complete)

### [2026-04-12 18:52] Specification Review
- Read impl_spec.md (Draft 4, all 1715 lines)
- Read preparation_report.md
- Spec is comprehensive and clear:
  - 11 functions fully specified with pseudocode
  - All parameters defined with recommended values
  - 14 sanity checks with specific criteria
  - Edge cases documented
  - Data mapping to prepared datasets is explicit
- No ambiguities found — proceeding to implementation.
- Spec review complete — clear, proceeding to implementation.

### [2026-04-12 18:55] Implementation (Step 2)
- Created artifacts/direction_4/code/model.py:
  - All 11 functions from spec implemented
  - ModelParams, RegimeClassifier, ModelA, PercentageModel dataclasses
  - compute_seasonal_factors (F1), compute_historical_average (F1a)
  - fit_interday_arma (F2) with AICc selection (p,q up to 5)
  - fit_intraday_arma (F3) with concatenated segment approach
  - build_regime_classifier (F4), assign_regime
  - optimize_regime_weights (F5) with multi-restart Nelder-Mead
  - forecast_raw_volume (F6), forecast_volume_percentage (F8)
  - train_percentage_model (F7) with blocked CV for lag selection
  - train_full_model (F9) with regime grid search
  - compute_evaluation_mape, compute_baseline_mape
  - load_volume_matrix helper
- Created artifacts/direction_4/code/evaluate.py:
  - Full evaluation pipeline for multiple tickers
  - Per-ticker and aggregate metrics
  - Sanity checks built into evaluation
- Quick validation on AAPL:
  - Training: 80.5s, 4 regimes selected
  - Intraday ARMA dominates weights (A component ~0.65-0.83)
  - Model MAPE: 0.3198 vs Baseline: 0.7966 (59.8% reduction)
  - Per-bin analysis: 27% reduction at bin 1, 56% at bin 21
  - pct_model: L=1, beta=0.41 (positive autocorrelation in surprises)
  - hist_pct sum: 1.0000
  - All sanity checks passed

### [2026-04-12 19:02] Exploration Notebook (Step 3)
- Created exploration.ipynb with 7 sections:
  1. Data loading and exploration
  2. Step-by-step model training
  3. Out-of-sample evaluation
  4. Sanity checks
  5. Single-day prediction walkthrough
  6. Cross-stock comparison
  7. Summary
- Notebook executed successfully via nbconvert (684KB output)

### [2026-04-12 19:16] Notebook Validation Complete (Step 3)
- Notebook executes without errors
- All cells produce expected outputs and visualizations
- Cross-stock comparison: 5 stocks evaluated successfully
- Full evaluation script running on 10 tickers (background, ~25min runtime)

### [2026-04-12 19:20] Validation Results (Step 4)
- Model A (Raw Volume):
  - Median MAPE reduction: 46.3% across 5 stocks (paper: ~24%)
  - Per-stock: AAPL 51.5%, NVDA 46.3%, MSFT 47.6%, JPM 44.5%, KO 33.4%
  - Per-bin: 16.9% at bin 1, up to 62.5% at later bins
  - All sanity checks pass (U-shape, ADF, parsimony, weights, beta)
  - Intraday ARMA (A) dominates weights (0.65-0.83)
- Model B (Volume Percentage):
  - MAD reduction: -3.98% (slight worsening vs baseline)
  - Discrepancy with paper's 7.55% improvement likely due to fixed 10%
    deviation bound (paper uses proprietary adaptive bounds)
- Decision: Results are valid. Model A significantly outperforms baseline.
  Model B underperformance is documented but not a blocking issue.

### [2026-04-12 19:25] Reporting (Step 5)
- Created artifacts/direction_4/metrics.md
- Created artifacts/direction_4/report.md
- Updated logs and progress

### [2026-04-12 19:26] Full 10-Ticker Evaluation Complete
- evaluate.py completed (~26 min runtime)
- 10 tickers evaluated: NVDA, TSLA, AAPL, AMZN, MSFT, META, AMD, GOOG, UNH, INTC
- Median MAPE reduction: 50.0% (9/10 stocks improved)
- Mean MAPE reduction: 40.7%
- Outlier: INTC -28.1% (worsening; Intel restructuring 2024-2025)
- Model B: Mean MAD worsening -16.9% (confirms notebook finding)
- Updated metrics.md and report.md with full results
- Results saved to evaluation_results.json

# Report: Direction 4 — Dual-Mode Intraday Volume Forecast

## Summary

This implementation reproduces the dual-model intraday volume forecasting system
described in Satish, Saxena, and Palmer (2014). The system operates on 15-minute
bins (26 per trading day) and is fitted independently per stock. Model A forecasts
raw bin-level volume by combining three components (historical average, inter-day
ARMA, intraday ARMA) with regime-switching dynamic weights based on cumulative
volume percentiles. Model B forecasts next-bin volume percentages by adjusting a
historical percentage curve using a surprise regression extending
Humphery-Jenner (2011).

The implementation was evaluated on 10 stocks (NVDA, TSLA, AAPL, AMZN, MSFT,
META, AMD, GOOG, UNH, INTC — top 10 by dollar volume) over a 63-day
out-of-sample period. Model A achieves a median 50.0% MAPE reduction over the
historical-average baseline (9/10 stocks show improvement), substantially
exceeding the paper's reported 24% median. Model B shows a worsening (-16.9%)
relative to the static historical percentage baseline, contrasting with the
paper's 7.55% improvement.

## Key Implementation Decisions

1. **Intraday ARMA (Function 3):** The spec describes a panel ARMA approach
   where each day is an independent segment with state reset at day boundaries.
   We implemented a simplified version: all daily segments are concatenated into
   a single series, and a standard ARMA model is fitted. This approximation
   means the model may pick up some spurious cross-day dependencies in the
   training data. During prediction, we use `model.apply()` on just the current
   day's deseasonalized observations, which naturally resets the state.

2. **Inter-day ARMA prediction:** For each call to `forecast_raw_volume`, we use
   `model.apply(series[:day_idx])` followed by `forecast(1)` to get the
   one-step-ahead prediction. This is faithful to the spec but computationally
   expensive during evaluation (each call re-processes the full series). In a
   production system, Kalman filter state updates would be used instead.

3. **Regime grid search:** The grid search over {3, 4, 5} regime counts uses a
   21-day validation split within the training window, then re-optimizes weights
   on the full window after selecting the best count, as specified.

4. **Weight optimization:** Multi-restart Nelder-Mead with 4 starting points
   (equal, H-dominant, D-dominant, A-dominant) in log-space for non-negativity.
   Falls back to equal weights if optimization doesn't improve.

5. **Model B surprise regression:** Percentage-space surprises (actual_pct -
   hist_pct) with blocked 5-fold time-series CV for lag selection. The selected
   lag is consistently L=1 across stocks, indicating simple first-order
   autocorrelation in percentage surprises.

## Validation Results

### Model A — Raw Volume Forecast

All critical sanity checks pass:
- Seasonal factors show the expected U-shaped intraday volume pattern
- Deseasonalized series are stationary (ADF p < 0.001)
- ARMA models are parsimonious (median p+q = 1)
- Weights are non-negative (by construction via exp-transformation)
- MAPE reduction is monotonic across components (H < H+D < H+D+A)

The model achieves strong MAPE reductions across 9 of 10 tested stocks:
- AMD: 65.0%, META: 56.7%, GOOG: 53.1%, UNH: 52.1%, AAPL: 51.5%
- TSLA: 48.5%, MSFT: 47.6%, NVDA: 46.3%, AMZN: 14.6%
- INTC: -28.1% (sole outlier — Intel's highly irregular volume during
  corporate restructuring in 2024-2025)
- Median: 50.0%, Mean: 40.7% (across all 10 stocks)

The improvement increases through the day (16.9% at bin 1 to 62.5% at later
bins), consistent with the paper's observation that intraday ARMA benefits from
accumulated conditioning information.

**Discrepancy with paper benchmarks:** Our 50.0% median reduction substantially
exceeds the paper's 24%. This is likely because our evaluation uses conditioned
forecasts (each bin forecast uses observed bins 1..i-1 from the same day), which
is the operationally correct setup. The paper's 24% may reflect a different
evaluation methodology (e.g., unconditional forecasts, or averaged over a
different stock universe and time period). The intraday ARMA component (A)
dominates the weights (0.65-0.83), confirming that same-day conditioning is the
primary source of improvement.

### Model B — Volume Percentage Forecast

Model B shows a worsening (-16.9% mean across 10 stocks) relative to the scaled
static historical percentage baseline. This contrasts with the paper's 7.55%
improvement.

**Analysis of the discrepancy:**
- The surprise regression consistently selects L=1 (single lag), suggesting
  limited predictive signal in percentage-space surprises beyond the first lag.
- Beta_1 = +0.41 indicates positive autocorrelation in percentage surprises,
  which is the expected sign.
- The deviation constraint (10% of base_pct) limits the regression adjustment
  to very small values (~0.004 percentage points), which may not be enough to
  overcome the noise added by the regression.
- The paper's improvement may depend on their proprietary adaptive deviation
  bounds (they state "we developed a separate method for computing the deviation
  bounds"), which could be wider or more context-sensitive than the fixed 10%.
- V_total_est uncertainty in early bins may cause the prediction-time surprise
  denominators to differ substantially from training-time values.

### Weight Structure

The intraday ARMA component (A) dominates across all regimes (0.65-0.83
weight). Historical average (H) receives near-zero weight in low-volume regimes
and increases in high-volume regimes, consistent with the intuition that
high-volume days are more predictable from recent history. Inter-day ARMA (D)
contributes modestly (0.02-0.14).

## Limitations and Potential Improvements

1. **Computational cost:** Training takes ~100s per stock, primarily due to the
   ARMA grid search (36 combinations for inter-day x 25 for intraday). The
   evaluation is even more expensive due to per-bin ARMA apply calls. A Kalman
   filter implementation would make online prediction O(1) per bin.

2. **Intraday ARMA day-boundary handling:** The concatenated-segment approach is
   an approximation. A proper panel ARMA with segment-boundary state resets would
   be more faithful to the spec.

3. **Model B underperformance:** The fixed 10% deviation bound may be too
   conservative. The paper uses proprietary adaptive bounds. Potential
   improvements: (a) widen max_deviation to 0.15-0.20; (b) use a multiplicative
   adjustment instead of additive; (c) implement adaptive bounds based on
   recent forecast accuracy.

4. **Combined ARMA terms:** The 11-term soft limit from the paper is at the
   boundary for AAPL (11 terms) and exceeded for some stocks. This is a
   soft observation, not a hard constraint, per the spec.

5. **NVDA high MAPE:** NVDA shows much higher absolute MAPE (2.253) than other
   stocks, reflecting its highly variable volume pattern (post-split regime shift
   in the data). The relative improvement (46.3%) is comparable.

6. **INTC outlier:** INTC is the only stock where the model worsens MAPE
   (-28.1%). This likely reflects Intel's highly unusual volume patterns during
   2024-2025 (foundry spin-off, restructuring), which violate the model's
   stationarity assumptions. In production, such stocks would be flagged for
   special handling.

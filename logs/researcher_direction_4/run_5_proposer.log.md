## Status: done
## Current Step: Draft 5 complete

### Log

[2026-04-12 00:17] Started proposer for direction 4, run 5. No prior drafts exist in run_5 directory — producing initial spec.

[2026-04-12 00:17] Read paper summary: Satish, Saxena, Palmer (2014). Key insights:
- Dual-model system: raw volume (3 components + regime weights) and volume percentage (surprise regression).
- 26 bins per day (15-min), trailing 6-month seasonal factors, inter-day ARMA per bin, intraday ARMA per symbol.
- Regime switching via cumulative volume percentile cutoffs.
- Volume percentage extends Humphery-Jenner (2011) dynamic VWAP with surprise-based adjustments.
- Key results: 24% median MAPE reduction (raw), 7.55% MAD reduction (percentage), 9.1% VWAP tracking error reduction.

[2026-04-12 00:18] Read full paper PDF (pages 1-9). Verified key algorithmic details:
- p.16: 26 bins, 15-min intervals, VWAP tracking error definition.
- p.17: MAPE formula, MAD formula, historical window average (N tunable), inter-day ARMA with AICc, deseasonalization by trailing 6-month bin average.
- p.18: Exhibit 1 data flow, intraday ARMA (AR < 5, dual model < 11 terms), regime switching with volume percentile cutoffs, dynamic weight overlay, custom curves for special days.
- p.18-19: Volume percentage methodology extending Humphery-Jenner, optimal model terms, separate deviation bounds, regressions without constant.
- p.19: No-intercept regressions for validation.
- pp.20-23: Exhibits 2-10 with validation results.

[2026-04-12 00:18] Read paper summary: Chen, Feng, Palomar (2016). Benchmark comparison target only.

[2026-04-12 00:19] Producing implementation specification draft 1.

[2026-04-12 00:26] Draft 1 complete. Key design decisions in this run:
- 10 functions organized for clear separation of concerns: 5 for Model A, 3 for Model B, 2 for orchestration.
- Explicit DailyOrchestration function (Function 9) showing complete intraday workflow — addresses a known gap from prior work.
- UpdateInterDayState (in Function 9) with explicit buffer update logic for daily state updates vs weekly re-estimation.
- Intraday ARMA conditioning specified as pure reset-and-reprocess (stateless) design with forecast purity guarantee.
- Weight optimization uses MSE/SLSQP as primary (simplex constraint: w1+w2+w3=1, non-negative) rather than MAPE/Nelder-Mead — avoids Nelder-Mead bounds issue.
- 11-term observation treated as soft guardrail (warning only), not hard constraint.
- N_interday_fit = 126 (aligns with seasonal window).
- Time-series cross-validation (blocked by day) for surprise lag selection — avoids temporal leakage from random K-fold.
- Surprises computed in percentage space (common V_total_est denominator).
- Train/predict denominator mismatch explicitly documented as edge case.
- All researcher inferences clearly marked with reasoning.

[2026-04-12 00:33] Critique 1 received: 2 major, 4 moderate, 5 minor issues. All valid.
- M1 (major): Surprise baseline mismatch — training uses dynamic forecasts, prediction uses unconditional. Will adopt option (a): both unconditional. Paper's language ("departures from a naive volume forecast model," p.18) supports this.
- M2 (major): CV implementation is a placeholder. Will add complete pseudocode.
- Mo1: Missing helper functions (BuildCumVolDistribution, EvaluateWeights, compute_hist_pct). Will add.
- Mo2: 11-term constraint should be hard, not soft. Will implement as hard constraint with fallback.
- Mo3: N_hist for H vs hist_pct ambiguity. Will explicitly share and document reasoning.
- Mo4: Humphery-Jenner not in assigned papers. Will add provenance flags throughout Model B.
- m1-m5: All minor issues accepted and will be addressed.

[2026-04-12 00:34] Re-read paper pp.16-24. Key verifications:
- p.18: "departures from a historical average approach, or a naive volume forecast model" — supports unconditional baseline for surprises.
- p.18: "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms" — after re-reading, this follows from AR < 5, reads as an outcome that is effectively a constraint.
- p.19: "we perform both regressions without the inclusion of a constant term" — confirmed this is about VWAP-error regressions, not surprise regression.
- p.24: Humphery-Jenner details are second-hand through Satish et al.
- Exhibit 6: Median reduction line is not monotonically increasing — has notable bin-to-bin variation.

[2026-04-12 00:34] Producing draft 2 with all revisions.

[2026-04-12 00:42] Draft 2 complete. All 11 critique issues addressed:
- M1: Unified surprise baseline to unconditional forecasts in both Function 7 and Function 8.
- M2: Full expanding-window CV pseudocode in Function 8 (min_train_days=21, 1-day step, MAE metric).
- Mo1: Added BuildCumVolDistribution (per-bin CDFs), EvaluateWeights (out-of-sample MAPE), compute_hist_pct (per-bin fraction averages).
- Mo2: Changed 11-term constraint from soft guardrail to hard constraint with reduced-order re-fit fallback in Function 3.
- Mo3: Explicitly shared N_hist between H and hist_pct with rationale in compute_hist_pct and Parameters section.
- Mo4: Added provenance flags throughout Model B (Overview, Function 7, Function 8, Parameters, Known Limitations).
- m1: Added information leakage acknowledgment note in Function 5.
- m2: Replaced ClassifyRegime_Training with direct ClassifyRegime call using training-period distribution.
- m3: Revised Exhibit 6 description to reflect non-monotonic median profile with bin-to-bin variation.
- m4: Clarified per-segment likelihood as correct approach; concatenation as pragmatic alternative.
- m5: No-intercept for surprise regression marked as Researcher inference with self-justifying reasoning.
- Added sanity check 18 (train/predict surprise baseline consistency).
- Added edge case 13 (combined term constraint activation).
- Added known limitation 10 (Model B provenance gap).
- Added max_combined_terms and min_cv_train_days to parameter table.

[2026-04-12 00:49] Critique 2 received: 0 major, 2 moderate, 4 minor issues. All valid and straightforward to address.
- Mo1 (moderate): ClassifyRegime called with bins 1..j in training but 1..j-1 in prediction. Fix: use j-1 in training to match.
- Mo2 (moderate): Dead H_array parameter in EvaluateWeights. Fix: remove from signature and callers.
- m1: Intraday ARMA "post-training state" language is misleading. Fix: clarify as zeros.
- m2: Exhibit 9 30-min arithmetic inconsistency in paper itself. Fix: add footnote.
- m3: MAPE variant pseudo-code missing. Fix: simplify to MSE only, note MAPE as future enhancement.
- m4: predict_interday buffer state not managed across training days. Fix: add explicit buffer reconstruction.

[2026-04-12 00:50] Producing draft 3 with all 6 fixes applied:
- Function 5: Changed observed_up_to_j to observed_before_j (bins 1..j-1) and ClassifyRegime arg from j to j-1.
- EvaluateWeights: Same causal fix. Also removed H_array from signature.
- OptimizeRegimeWeights: Removed H from signature (was only passed through to EvaluateWeights).
- Function 10 calling code: Removed model_a.H from OptimizeRegimeWeights call.
- Function 6: Replaced "post-training state" with "zeros" in conditioning comment.
- Validation: Added arithmetic inconsistency footnote for Exhibit 9 30-min row.
- Function 5: Simplified MSE/MAPE design decision to MSE-only with future enhancement note.
- predict_interday: Added explicit buffer reconstruction from historical data for training-time usage.

[2026-04-12 00:53] Draft 3 complete. Verified all 6 changes:
- No remaining references to observed_up_to_j (all changed to observed_before_j).
- No remaining H_array in code (only in revision notes describing the fix).
- No remaining "post-training state" in code (only in revision notes).
- predict_interday now has full buffer reconstruction pseudocode.
- Exhibit 9 footnote added.
- MSE/MAPE design decision simplified.
All changes are internally consistent. Spec is implementation-ready.

[2026-04-12 01:02] Critique 3 received: 0 major, 1 moderate, 2 minor issues. All valid and straightforward.
- Mo1 (moderate): predict_interday MA buffer reconstruction uses undefined function and has circular dependency. Will replace with sequential burn-in approach.
- m1: Calibration section MAPE variant text contradicts Function 5's future-enhancement treatment. Will align.
- m2: Day-index convention (calendar vs trading-day) never clarified. Will add convention statement.

[2026-04-12 01:03] Producing draft 4 with all 3 fixes:
- predict_interday: Replaced `model.predict_from_params()` and simple q-day MA loop with sequential burn-in processing. Starts from (day_d - burn_in - q) with MA buffer initialized to zeros, processes forward computing AR+MA predictions and residuals at each step. burn_in = max(2*q, 10) ensures convergence for invertible MA processes.
- Calibration section line: Changed "Variant: MAPE minimization..." to "Variant (future enhancement): MAPE minimization..." and "Switch to" to "If implemented, switch to". Now consistent with Function 5 pseudocode.
- Added "Date arithmetic convention" paragraph after Outputs in Model Description, clarifying all date indices are trading-day indices and is_full_trading_day() filters half-days only.

[2026-04-12 01:05] Draft 4 complete. Verified all 3 changes:
- No remaining references to predict_from_params (removed).
- predict_interday MA buffer now uses sequential burn-in with explicit AR+MA loop.
- Calibration section and Function 5 are now consistent (both treat MAPE as future enhancement).
- Date arithmetic convention added once in Model Description, applies globally.
All changes are internally consistent. Spec is implementation-ready.

[2026-04-12 01:08] Critique 4 received: 0 major, 0 moderate, 1 minor issue. Spec declared implementation-ready.
- m1 (minor): Function 5 docstring still says "both MSE and MAPE objectives" — inconsistent with rest of spec. One-line text fix.

[2026-04-12 01:09] Producing draft 5 with single cosmetic fix:
- Function 5 docstring: Changed "both MSE and MAPE objectives" to "MSE as the primary objective; a MAPE-based alternative is noted as a future enhancement."

[2026-04-12 01:09] Draft 5 complete. All text now consistently treats MAPE as a future enhancement throughout the spec (Function 5 docstring, Function 5 pseudocode, Calibration section). Spec is finalized.

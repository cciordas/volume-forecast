## Status: done
## Current Step: Draft 4 complete

### [2026-04-12 11:05] Session started
- Role: Proposer, Direction 4, Run 6
- No prior drafts or critiques exist in run_6 directory
- Will produce impl_spec_draft_1.md from scratch based solely on the papers

### [2026-04-12 11:05] Paper reading: Satish, Saxena, Palmer (2014)
- Read full PDF (10 pages) and structured summary
- Key insights extracted:
  - Model A: 4-component raw volume forecast (historical avg, inter-day ARMA, intraday ARMA, dynamic weight overlay)
  - Model B: Volume percentage forecast based on Humphery-Jenner (2011) dynamic VWAP
  - Bin structure: 26 bins x 15 minutes (9:30-16:00 ET)
  - Exhibit 1: Historical window uses "Prior 21 days", ARMA Daily uses "Prior 5 days", ARMA Intraday uses "Current Bin" and "4 Bins Prior to Current Bin"
  - AICc for ARMA model selection, p,q in {0,...,5}
  - Intraday ARMA: deseasonalized by 6-month trailing average, fitted on rolling 1-month window
  - "fewer than 11 terms" constraint on combined dual ARMA
  - Regime switching via historical volume percentile cutoffs
  - Model B: rolling regression on volume surprises, 10% deviation limit, 80% switch-off
  - Performance: 24% median MAPE reduction (raw), 7.55% median error reduction (percentage, 15-min)
  - VWAP tracking error: 9.1% reduction (9.62 -> 8.74 bps)
  - Exhibit 9: detailed results by bin size
  - Exhibit 10: VWAP simulation results

### [2026-04-12 11:06] Ambiguities identified
1. N (historical window): "a variable that we shall call N" — exact value not disclosed; Exhibit 1 shows 21 days
2. Number of regime buckets: not disclosed
3. Weight optimization method: not disclosed (only "minimizes the error on in-sample data")
4. Humphery-Jenner regression specifics: optimal number of terms not disclosed for U.S. equities
5. Deviation bounds computation: "separate method" developed but not specified
6. Inter-day ARMA fitting window: "Prior 5 days" in Exhibit 1 but text doesn't confirm exact window

### [2026-04-12 11:06] Ambiguity resolutions
1. N_hist = 21 days based on Exhibit 1's "Prior 21 days" label
2. N_regimes selected via cross-validation over {1..5} (Researcher inference)
3. MAPE minimization via Nelder-Mead with exp-transform (Researcher inference)
4. N_surprise_lags selected via cross-validation over {1..10} (Researcher inference)
5. Deviation bounds: used the paper's stated 10% limit (Satish p.24, Humphery-Jenner 2011)
6. N_interday_fit = 252 days (~1 year) to give ARMA enough data; separate from "Prior 5 days" which may refer to a different concept in Exhibit 1

### [2026-04-12 11:10] Draft 1 complete
- Produced impl_spec_draft_1.md with 12 functions:
  - Functions 1-6: Model A (seasonal factors, historical average, inter-day ARMA, intraday ARMA, regime weights, raw prediction)
  - Functions 7-9: Model B (historical percentages, surprise regression, percentage prediction)
  - Functions 10-12: Orchestration (full training, daily prediction, evaluation metrics)
- 21 parameters documented with recommended values and sensitivity
- 13 sanity checks
- 11 edge cases
- 8 known limitations
- 17 explicitly marked Researcher inference items
- Complete paper reference table
- Ready for critic review

### [2026-04-12 11:17] Critique 1 received — assessment
- 5 major issues, 9 minor issues
- Severity breakdown:
  - M1 (MAJOR): Joint "fewer than 11 terms" constraint misapplied — was per-model, should be joint. Valid.
  - M2 (MAJOR): No-intercept citation wrong — p.19 refers to VWAP validation regressions, not surprise regression. Valid.
  - M3 (MAJOR): Weight normalization unspecified — exp-transform allows arbitrary scale. Valid.
  - M4 (MAJOR): Self-updating deviation limits from Humphery-Jenner not addressed. Valid.
  - M5 (MAJOR): Model A-based surprise variant (sophisticated) never specified in pseudocode. Valid.
  - m1-m9: All minor issues valid and actionable.
- All 14 issues will be addressed in draft 2.

### [2026-04-12 11:27] Draft 2 complete — changes summary
- **M1 resolved:** Function 3 now returns interday_term_counts. Function 4 computes
  intraday_budget = max_dual_arma_terms - max(interday_term_counts) and enforces
  joint constraint. New edge case #12 for budget exhaustion. New sanity check #15.
- **M2 resolved:** Removed false p.19 citation from Function 8. No-intercept choice
  now marked as Researcher inference (#19) justified by mean-zero-surprise property.
  Detailed comment explains what p.19 actually refers to (VWAP validation regressions).
- **M3 resolved:** minimize_mape now uses softmax parameterization (w_j = exp(theta_j) / sum)
  instead of plain exp-transform. Weights guaranteed to sum to 1 and be non-negative.
  Default fallback weights [1/3, 1/3, 1/3] now consistent with optimized weights.
  Sanity check #11 updated. Shapes table updated (regime_weights: "sum=1").
- **M4 resolved:** Added Function 9a (calibrate_adaptive_limits) with per-stock adaptive
  calibration. Deviation limit set as 95th percentile of historical surprise magnitudes,
  clamped to [0.5x, 2.0x] base. Switch-off threshold set from median cumulative volume.
  Added to training pipeline (Function 10 Step 8). Parameters renamed to base_deviation_limit
  and base_switchoff_threshold. Added N_calibration parameter. Known limitation #9 added
  noting Humphery-Jenner (2011) should be consulted.
- **M5 resolved:** Function 8 now accepts optional model_a_forecasts parameter for
  sophisticated variant. Function 9 accepts optional baseline_pct for Model A-based
  surprises. Function 10 includes generate_model_a_training_forecasts helper.
  Function 11 computes Model A-based baseline on the fly during live prediction.
  Added use_model_a_baseline parameter. Variants section updated.
- **m1 resolved:** Overview updated to acknowledge "four components" per p.17, clarifying
  that three are signal sources and the fourth is the combination mechanism.
- **m2 resolved:** Added detailed "Note on Exhibit 1 labels" in Model Description explaining
  "Prior 5 days" refers to p_max = 5, not fitting window.
- **m3 resolved:** Same note explains "4 Bins Prior to Current Bin" as p_max_intra = 4
  upper bound for AICc selection.
- **m4 resolved:** Split into two separate sanity checks: #1 for Model A [1,0,0] weights
  (verify V_hat = H, no absolute benchmark), #2 for Model B zero coefficients
  (Exhibit 9 HVWAP 0.00874 benchmark).
- **m5 resolved:** pct_forecasts[1] = hist_pct_today[1] explicitly set before main loop
  in Function 11. New sanity check #16 verifies this.
- **m6 resolved:** Added full percentile_rank helper function definition with strict
  less-than semantics, edge case handling, and rationale.
- **m7 resolved:** ARMA make_state initialization specified: unconditional mean for AR
  lags, zero for MA residuals, sequential Kalman filter processing.
- **m8 resolved:** Added detailed note in Function 5 explaining the look-ahead approximation,
  justifying why it is acceptable, and noting the walk-forward alternative. Added
  known limitation #10.
- **m9 resolved:** Paper References table updated: deviation limit and switch-off citations
  now read "Satish et al. 2014, p.24 (referencing Humphery-Jenner 2011)".
- Researcher inference list expanded from 17 to 28 items.
- Total: 16 sanity checks (was 13), 12 edge cases (was 11), 10 known limitations (was 8).

### [2026-04-12 11:34] Critique 2 received — assessment
- 1 major issue, 5 minor issues (down from 5 major + 9 minor in critique 1)
- Severity breakdown:
  - M1 (MAJOR): Sophisticated variant baseline uses actual volumes for observed bins during live prediction, creating near-zero surprises that don't match the training-time surprise signal. Valid and important.
  - m1 (Minor): predict_next() idempotency not stated. Valid.
  - m2 (Minor): Switch-off calibration uses switchoff_bin - 1, systematically producing below-base thresholds. Valid.
  - m3 (Minor): generate_model_a_training_forecasts uses median regime for all bins instead of reconstructing actual regime. Valid.
  - m4 (Minor): remaining_pct in Function 9 hardcoded to hist_pct even in sophisticated mode. Valid.
  - m5 (Minor): Sanity check 15 notation confusing. Valid.
- All 6 issues will be addressed in draft 3.

### [2026-04-12 11:34] Draft 3 complete — changes summary
- **M1 resolved:** Completely redesigned the sophisticated variant live-prediction
  baseline in Function 11. Pre-observation Model A forecasts are now computed ONCE
  at the start of the day (current_bin=0, no observations) and used as a FIXED
  baseline throughout the day. This ensures training/prediction consistency:
  during training, generate_model_a_training_forecasts computes Model A
  forecasts without using actual volumes, so live prediction must do the same.
  Added detailed inline comment explaining the rationale. Updated the reference
  paragraph below Function 11. Updated Researcher inference item #27.
- **m1 resolved:** Added explicit note to predict_next() interface: "This is a
  pure query that does NOT modify the model's internal state. It can be called
  multiple times and will return the same value until append_observation() is
  called." Added Researcher inference item #29.
- **m2 resolved:** Changed switch-off calibration to use the crossover bin's
  median directly (instead of switchoff_bin - 1). Added inline comment explaining
  why the previous approach systematically biased thresholds below the base value.
  Added Researcher inference item #30.
- **m3 resolved:** Replaced median regime default in generate_model_a_training_forecasts
  with per-(day, bin) regime reconstruction from historical cumulative volume
  percentiles, mirroring Function 5's approach. For bin 1, defaults to median
  percentile (no prior bins). Updated function docstring and Researcher inference
  item #28.
- **m4 resolved:** Changed remaining_pct computation in Function 9 from
  hist_pct to base (which is baseline_pct in sophisticated mode, hist_pct in
  naive mode). Updated inline comment to explain the consistency requirement.
- **m5 resolved:** Rewrote sanity check 15 to use
  "<= max_dual_arma_terms" instead of "< 11 (= max_dual_arma_terms + 1)".
- Researcher inference list expanded from 28 to 30 items.

### [2026-04-12 11:45] Critique 3 received — assessment
- 0 major issues, 2 minor issues (down from 1 major + 5 minor in critique 2)
- Critic recommends accepting draft 3 as implementation-ready.
- m1 (Minor): Pre-observation baseline creates secondary context mismatch for later
  bins in sophisticated variant. Low severity, bounded by deviation clamp. Will
  document as known trade-off and add to Known Limitations.
- m2 (Minor): N_surprise_lags CV described in calibration prose but not implemented
  in pseudocode. Will simplify to fixed hyperparameter (consistent with paper's
  global-optimal statement).

### [2026-04-12 11:46] Draft 4 complete — changes summary
- **m1 resolved:** Added detailed inline comment in Function 11 documenting the
  secondary context mismatch as a known trade-off. Explains why the mismatch
  direction (larger live surprises) is bounded by deviation clamp and is far less
  harmful than the original actual-volume approach. Describes the iterative
  baseline construction as a potential future refinement. Added Known Limitation
  #11 with full description. Updated the reference paragraph below Function 11.
  Added Researcher inference item #31.
- **m2 resolved:** Simplified Calibration step 4 from per-stock CV to a fixed
  global hyperparameter (N_surprise_lags = 5), consistent with the paper's
  statement that an optimal number was identified globally for U.S. equities.
  Added guidance for offline tuning via panel-level CV as a one-time exercise.
  Updated parameter table sensitivity from "High" to "Medium". Updated
  Researcher inference item #9 (removed surprise lag CV reference). Added
  Researcher inference item #32.
- Researcher inference list expanded from 30 to 32 items.
- Known Limitations expanded from 10 to 11 items.

# Researcher Proposer Log — Direction 4, Run 3

## Status: done
## Current Step: Draft 4 complete

### [2026-04-11 23:28] Session started
- Role: Proposer, Direction 4, Run 3
- No existing drafts found in work/researcher/direction_4/run_3/
- Producing initial implementation specification draft
- Independent run: not referencing runs 1 or 2

### [2026-04-11 23:28] Reading paper summaries
- Read summaries: satish_saxena_palmer_2014.md, chen_feng_palomar_2016.md, brownlees_cipollini_gallo_2011.md, bialkowski_darolles_lefol_2008.md
- Direction 4 centers on Satish et al. (2014): dual-model system with raw volume forecasting (3-component + regime weights) and volume percentage forecasting (dynamic VWAP extension)
- Chen et al. (2016) provides comparison benchmarks (Kalman filter approach)

### [2026-04-11 23:29] Reading full papers
- Read Satish et al. 2014 (full, pp.1-10 / pp.15-24)
  - Key insight: Exhibit 1 shows data flow with explicit labels ("Prior 21 days", "Prior 5 days", "4 Bins Prior to Current Bin")
  - "Prior 5 days" for inter-day ARMA is ambiguous: could be training window or AR memory
  - "Fewer than 11 terms" is preceded by "As a result" -- descriptive, not prescriptive
  - Volume percentage model based on Humphery-Jenner (2011), key parameters proprietary
  - No-intercept regressions: explicitly stated for validation regressions (p.19)
  - Exhibit 9: 7.55% median reduction for 15-min bins
  - Exhibit 10: 9.1% VWAP tracking error reduction, t=2.34, p<0.01
- Read Chen et al. 2016 (pp.1-14)
  - Kalman filter on log-volume, state-space with EM calibration
  - Average dynamic MAPE 0.46 across 30 securities
  - VWAP tracking 6.38 bps
  - Different universe (30 global securities vs 500 US equities) makes direct comparison imperfect

### [2026-04-11 23:30] Ambiguities identified and resolved
1. "Prior 5 days" for inter-day ARMA: Interpreted as AR prediction memory, not training window. Reasoning: fitting ARMA(5,5) on 5 observations is statistically impossible. Training window set to N_interday=126 (6 months).
2. "Fewer than 11 terms": Interpreted as descriptive outcome of AICc selection and AR decay, not a hard constraint. The "As a result" phrasing is key.
3. Number of regimes: Not disclosed. Set to 3 (tercile buckets) as simplest scheme consistent with "several weight models."
4. K_reg (surprise lag count): Not disclosed. Set to 3 as reasonable default.
5. Surprise baseline: Paper says "we could apply" raw model -- aspirational. Implemented with raw model as primary, historical as fallback option.
6. No-intercept: Paper quote is about validation regressions, but principle extends to prediction regression. Noted as Researcher inference.

### [2026-04-11 23:35] Draft 1 complete
- Output: work/researcher/direction_4/run_3/impl_spec_draft_1.md
- 9 pseudocode functions covering full pipeline
- 18 parameters documented with sources
- 11 sanity checks
- 8 edge cases
- 7 known limitations
- Key Researcher inferences: regime count/thresholds, N_interday interpretation, K_reg value, MSE objective with simplex constraint, re-estimation schedule, epsilon floor, surprise normalization details, renormalization approach

### [2026-04-11 23:41] Critique assessment
- Read researcher_critique_1.md
- **5 major issues, 7 minor issues**
- Severity assessment:
  - M1 (intraday ARMA "4 bins" annotation): Major -- valid omission. Added explicit discussion of Exhibit 1 "4 Bins Prior to Current Bin" in Function 6 with both interpretations and justification.
  - M2 (MSE vs MAPE for weight optimization): Major -- valid design question. Promoted to co-equal design decision with both options documented in Function 5.
  - M3 (train/predict mismatch in surprise denominator): Major -- valid. Added explicit discussion in Function 7 Step 3 and cross-reference in Function 8.
  - M4 (missing metric formulas): Major -- valid. Added new "Metrics" subsection with exact MAPE and percentage error formulas from p.17, plus aggregation procedure.
  - M5 (inter-day ARMA state update not in pseudocode): Major -- valid gap. Added Function 2b: UpdateInterDayState with explicit pseudocode for AR/MA buffer updates.
  - m1: Upgraded N_hist sensitivity to Medium-High, added p.16 quote about tunability.
  - m2: Strengthened deviation bounds flagging as proprietary simplification in Function 7 Step 6 and parameter table.
  - m3: Added soft constraint recommendation with warning in Function 3.
  - m4: Clarified trading vs calendar month ambiguity in Function 3 and parameter sources.
  - m5: Defined DEFAULT_REGIME = 1 (middle tercile) with justification in Function 4.
  - m6: Added explicit note about pre-market regime limitation in Function 4 and DailyOrchestration.
  - m7: Tightened per-category reduction phrasing to distinguish within-category vs aggregate.

### [2026-04-11 23:41] Draft 2 complete
- Output: work/researcher/direction_4/run_3/impl_spec_draft_2.md
- All 5 major and 7 minor critique issues addressed
- Added Function 2b (UpdateInterDayState) -- now 10 pseudocode functions
- Added Metrics subsection with exact formulas
- Added sanity check #12 (inter-day ARMA state consistency)
- Expanded Variants section to include weight optimization loss function variant
- Updated Data Flow diagram to include UpdateInterDayState
- Updated Paper References table with new entries

### [2026-04-11 23:55] Critique 2 assessment
- Read researcher_critique_2.md
- **0 major, 2 medium, 3 minor issues**
- All 12 issues from critique 1 confirmed resolved by critic
- Severity assessment:
  - Med-1 (Missing intraday ARMA state conditioning function): Medium -- valid. Function 6 calls `intraday_model.update_state(deseas_observed)` without specifying how the array is processed. Added Function 6b: ConditionIntraDayARMA with explicit reset-and-reprocess pseudocode, design choice discussion (stateless vs incremental), and note about saving training state.
  - Med-2 (Ambiguous current_bin for training forecasts): Medium -- valid. Training could use static (current_bin=0) or dynamic (current_bin=i-1) forecasts. Clarified: dynamic forecasts with caching to match prediction-time behavior. Added pre-computation step in Function 8 with cost analysis.
  - m-new-1 (Multi-step ARMA degradation): Added note in Function 6 explaining convergence behavior and new sanity check #13.
  - m-new-2 (Early-bin padding): Added design choice discussion in Function 8 with both options documented.
  - m-new-3 (Expected percentage denominator): Added discussion in Function 8 noting the second source of train/predict inconsistency and why it is a second-order effect.

### [2026-04-11 23:55] Draft 3 complete
- Output: work/researcher/direction_4/run_3/impl_spec_draft_3.md
- All 2 medium and 3 minor critique 2 issues addressed
- Added Function 6b (ConditionIntraDayARMA) -- now 11 pseudocode functions
- Added multi-step ARMA degradation note in Function 6
- Rewrote Function 8 with dynamic forecast caching, explicit denominator construction, and early-bin padding discussion
- Added sanity checks #13 (multi-step convergence) and #14 (conditioning consistency)
- Updated Data Flow diagram to include ConditionIntraDayARMA
- Updated Paper References table with 6 new entries

### [2026-04-12 00:05] Critique 3 assessment
- Read researcher_critique_3.md
- **0 major, 0 medium, 3 minor issues**
- All 5 issues from critique 2 confirmed resolved by critic
- Critique assessment: "Draft 3 is an excellent implementation specification... ready for implementation"
- Severity assessment:
  - m-new-1 (forecast purity requirement): Minor -- valid. Added explicit note in Function 6 specifying that model.forecast(steps=N) must be a pure function that does not mutate AR/MA buffers.
  - m-new-2 (redundant conditioning in DailyOrchestration): Minor -- valid. Added optimization note in Function 9 explaining the redundancy, suggesting hoisting conditioning outside inner loop, and noting the interaction with forecast purity.
  - m-new-3 (structural asymmetry in surprise denominator): Minor -- valid. Added note in Function 8 denominator discussion explaining that training uses different denominators for actual vs expected within a single surprise (daily_total vs lag_forecast_total), while prediction uses the same denominator (V_total_est). Noted the trivial fix of using daily_total for both terms during training.

### [2026-04-12 00:05] Draft 4 complete
- Output: work/researcher/direction_4/run_3/impl_spec_draft_4.md
- All 3 minor critique 3 issues addressed with annotation-level changes (no structural/algorithmic modifications)
- Added forecast purity note in Function 6 (after intraday_model.forecast call)
- Added conditioning redundancy optimization note in Function 9 (inner loop)
- Added surprise denominator structural asymmetry note in Function 8 (denominator discussion)
- Updated Paper References table with 3 new entries
- Total issues addressed across 3 critique rounds: 5 major + 2 medium + 13 minor = 20 issues

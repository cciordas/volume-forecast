## Status: done
## Current Step: Draft 3 complete (revision based on critique 2)

### [2026-04-11 22:48] Session started
- Role: Proposer, Direction 4, Run 1
- Direction: Dual-Mode Volume Forecast (Raw + Percentage)
- Assigned papers:
  - satish_saxena_palmer_2014 (Foundational)
  - chen_feng_palomar_2016 (Comparison)

### [2026-04-11 22:49] Paper summaries read
- Read satish_saxena_palmer_2014.md summary
- Read chen_feng_palomar_2016.md summary
- Key relationship: Chen et al. claims to outperform Satish et al., providing comparison context

### [2026-04-11 22:50] Full papers read
- Read satish_saxena_palmer_2014.pdf (pages 1-11, complete)
  - Confirmed correct paper (Satish, Saxena, Palmer from FlexTrade)
  - Key insights:
    - Four-component raw volume model: rolling mean + inter-day ARMA + intraday ARMA + dynamic weight overlay
    - Volume percentage model extends Humphery-Jenner (2011) dynamic VWAP
    - Exhibit 1 shows architecture diagram: Historical Window feeds from prior 21 days, ARMA Daily from prior 5 days, ARMA Intraday from current day bins
    - Regime switching via volume percentile cutoffs (proprietary values)
    - Safety constraints: 10% deviation limit, 80% volume switch-off
    - AICc for ARMA order selection, p,q in {0,...,5}
    - Deseasonalization: divide by trailing 6-month bin average
    - Intraday ARMA fitted on rolling 1-month window
    - Results: 24% median MAPE reduction, 9.1% VWAP tracking error reduction
    - MANY parameters are proprietary/undisclosed (weighting coefficients, regime thresholds, regression terms)
  
- Read chen_feng_palomar_2016.pdf (pages 1-15, complete)
  - Kalman filter state-space model for log-volume
  - Three algorithms: Kalman filter, Kalman smoother (RTS), EM
  - State dimension = 2 (eta, mu), scalar observation
  - Robust variant with Lasso sparse outlier detection
  - Complete closed-form EM updates provided (Appendix A)
  - Results: MAPE 0.46 (robust KF) vs benchmark, VWAP 6.38 bps
  - Provides useful comparison context for validation benchmarks

### [2026-04-11 22:52] Writing implementation specification
- Key challenge: many Satish et al. parameters are proprietary
- Strategy: specify the architecture fully, provide reasonable defaults where proprietary values are withheld, mark all such choices as "Researcher inference"
- Will use Chen et al. results as cross-validation benchmarks

### [2026-04-11 22:53] Draft 1 complete
- Produced impl_spec_draft_1.md
- Key decisions made:
  - N_hist = 21 days (inferred from Exhibit 1 diagram showing "Prior 21 days")
  - 3 regime buckets at [33, 67] percentile (proprietary in paper, reasonable default)
  - K_regression = 3 for surprise lags (proprietary, reasonable default)
  - Unconstrained OLS for weight calibration (not specified in paper)
- Major areas of Researcher inference (proprietary parameters):
  - Regime bucket count and boundaries
  - Weighting coefficients
  - Optimal regression terms for percentage model
  - Weight constraints (sum-to-1, non-negativity)
- Document covers all three facets: Algorithm, Parameters, Validation
- Includes pseudocode for both Model A and Model B, plus weight calibration
- Cross-referenced all algorithmic steps to paper sections

### [2026-04-11 23:01] Critique 1 received and assessed
- 4 major issues, 8 minor issues
- Severity assessment:
  - M1 (Overview conflates Model A and Model B results): Major but easy fix. The Overview incorrectly bundles the 24% MAPE reduction (Model A) with 9.1% VWAP improvement (Model B). Will separate clearly.
  - M2 (Inter-day ARMA training window unspecified): Major. Draft silently assumed "250 days or more" without citation. Must add N_interday parameter, discuss "Prior 5 days" ambiguity from Exhibit 1, provide researcher inference with justification.
  - M3 (Intraday ARMA day-boundary concatenation): Major. Concatenating across days creates artificial discontinuities at overnight gaps. Must address boundary handling explicitly and reconcile Exhibit 1 ("4 Bins Prior to Current Bin") with "rolling one-month window."
  - M4 (V_total_est undefined in Model B): Major. Creates undefined variable and circular dependency. Must define explicitly as sum of observed + raw forecasts, specify update ordering.
  - m1 (MAPE definition imprecise): Minor. Will add precise formula.
  - m2 (AICc formula incomplete): Minor. Will define AIC explicitly.
  - m3 ("fewer than 11 terms" ambiguity): Minor. Will clarify interpretation.
  - m4 (Regime classification timing): Minor. Will add minimum bins before switching.
  - m5 (Deseasonalization floor): Minor. Will integrate epsilon floor into pseudocode.
  - m6 (Rolling regression spec too vague): Minor. Will clarify pooled vs per-bin, window.
  - m7 (Renormalization conflicts with safety): Minor. Will fix interaction.
  - m8 (Missing Chen et al. context): Minor. Will add note about Chen claiming superiority.
- All issues are valid and actionable
- Re-read paper pages 1-11 to verify critique's claims -- all confirmed

### [2026-04-11 23:01] Writing draft 2
- Restructuring Overview to separate Model A and Model B results
- Adding N_interday parameter with Exhibit 1 ambiguity discussion
- Adding explicit day-boundary handling in intraday ARMA pseudocode
- Defining V_total_est explicitly with update ordering
- Addressing all 8 minor issues

### [2026-04-11 23:09] Draft 2 complete
- All 4 major issues addressed:
  - M1: Overview now clearly separates Model A results (24% MAPE) from Model B results (7.55% pct error) and VWAP simulation (9.1% tracking error)
  - M2: Added N_interday parameter (recommended 252) with explicit Exhibit 1 "Prior 5 days" ambiguity discussion, two interpretations with justification for choosing interpretation (b)
  - M3: Added comprehensive day-boundary handling discussion in Pseudocode Step 3 with three mitigation options (a/b/c), recommending naive concatenation as default; reconciled Exhibit 1 (prediction inputs) with paper text (training window)
  - M4: V_total_est now explicitly defined as sum(observed) + sum(raw forecasts); update ordering specified (Model A first, then Model B); no circular dependency
- All 8 minor issues addressed:
  - m1: Added precise MAPE formula in Calibration section with paper reference
  - m2: AICc formula now includes full AIC definition (AIC = -2*log_likelihood + 2*k) in both pseudocode and Calibration section
  - m3: Clarified "fewer than 11 terms" as descriptive (based on "As a result" phrasing), noted ambiguity in what "terms" counts
  - m4: Added min_regime_bins parameter (default 3) to avoid unreliable early-day regime classification
  - m5: epsilon floor integrated directly into pseudocode at seasonal factor computation with Researcher inference annotation
  - m6: Clarified regression as single/pooled across all bins (not per-bin), with training window = all training days
  - m7: Fixed renormalization to scale the baseline rather than post-multiply the adjusted forecast, preserving safety constraints
  - m8: Added context that Chen et al. claim to outperform Satish et al. in the Validation comparison section
- Additional improvements:
  - Added ARMA convergence failure detection criteria (optimizer flag, Hessian, standard errors)
  - Clarified forecast update is reconditioning (state update) not re-estimation
  - Expanded Paper References table with new entries for all additions

### [2026-04-11 23:16] Critique 2 received and assessed
- 0 major issues, 5 minor issues
- Severity assessment: all minor, all actionable refinements rather than corrections
  - m1 (V_total_est uses unconditional forecasts): Minor. Two separate ForecastRawVolume calls needed -- unconditional for expected_pct baseline, conditioned for V_total_est. Impact grows in second half of day.
  - m2 (Deviation constraint vs scaling): Minor. Constraint should be relative to scaled baseline, not unscaled p_hist. Otherwise effective deviation can exceed max_deviation when scale < 1.
  - m3 (Regression training window unspecified): Minor. Add N_reg_train parameter (252 days). Note lookahead bias (negligible).
  - m4 (Exhibit 6 description imprecise): Minor. Refine numbers to match figure more precisely.
  - m5 (Hysteresis not in pseudocode): Minor. Clarify as optional enhancement with example pseudocode, not required for baseline.
- Critic's recommendation: draft is ready for implementation; minor issues can be addressed in quick revision.

### [2026-04-11 23:16] Writing draft 3
- Addressing all 5 minor issues from critique 2

### [2026-04-11 23:17] Draft 3 complete
- All 5 minor issues addressed:
  - m1: ForecastVolumePercentage now makes two separate ForecastRawVolume calls: unconditional (current_bin=1) for expected_pct baseline, conditioned (current_bin=actual) for V_total_est. Data Flow diagram updated. Detailed comment explains the rationale.
  - m2: Deviation constraint now computed relative to scaled_base (= scale * p_hist[next_bin]) instead of unscaled p_hist[next_bin]. Scale and scaled_base computed in Step 4, reused in Step 6. Eliminates the interaction bug where effective deviation exceeded max_deviation when scale < 1.
  - m3: Added N_reg_train parameter (default 252, range [126, 504]) to parameter table. TrainVolumePercentageModel now explicitly defines training_days using N_reg_train. Lookahead bias documented with justification for why it's negligible.
  - m4: Exhibit 6 description refined to "10-12% at 9:30 to 30-33% by 15:30, with non-monotonicity around 10:00-10:30" and "bottom-95% from ~15% to 35-40%".
  - m5: Edge case 10 now explicitly marks hysteresis as optional (not required for baseline), includes example pseudocode with regime_hysteresis parameter, and references that the paper does not mention hysteresis.
- Additional improvements:
  - Refactored Steps 4-6 of ForecastVolumePercentage to compute scale/scaled_base once in Step 4 and reuse in Step 6 (was duplicated)
  - cum_pct in Step 5 now derived from already-computed remaining_fraction (avoids redundant computation)
  - Data Flow diagram updated to show two-call pattern and scaled constraint application
  - Paper References table expanded with 6 new entries for all additions

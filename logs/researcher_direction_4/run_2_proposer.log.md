## Status: done
## Current Step: Draft 4 complete (revision based on critique 3)

### [2026-04-11 23:02] Session started
- Role: Proposer, Direction 4, Run 2
- Direction: Dual-Mode Volume Forecast (Raw + Percentage)
- Papers assigned:
  - satish_saxena_palmer_2014 (Foundational) - Predicting Intraday Trading Volume and Volume Percentages
  - chen_feng_palomar_2016 (Comparison) - Forecasting Intraday Trading Volume: A Kalman Filter Approach

### [2026-04-11 23:02] Paper reading complete
- Read both paper summaries and full PDFs
- Satish et al. (2014): Four-component raw volume model (historical average + inter-day ARMA + intraday ARMA + dynamic weight overlay) plus volume percentage model extending Humphery-Jenner (2011) dynamic VWAP
- Chen et al. (2016): Kalman filter state-space model for log-volume; serves as comparison/context for understanding the decomposition approach
- Key insight: Satish paper is practitioner-oriented with some proprietary parameters withheld; implementation requires rediscovering regime thresholds and weighting coefficients via grid search
- The two models (raw volume + volume percentage) are tightly coupled: raw model's surprise signal feeds into percentage model

### [2026-04-11 23:03] Writing implementation specification draft 1
- Synthesizing algorithm from Satish et al. Section "Raw Volume Forecast Methodology" (pp. 17-18) and "Volume Percentage Forecast Methodology" (pp. 18-19)
- Using Exhibit 1 (p. 18) flow diagram as architectural reference
- Cross-referencing Chen et al. for comparison metrics and alternative approaches

### [2026-04-11 23:07] Draft 1 complete
- Produced `work/researcher/direction_4/run_2/impl_spec_draft_1.md`

### [2026-04-11 23:12] Critique 1 received - assessment
- 2 major issues, 7 minor issues
- All issues accepted and addressed in draft 2

### [2026-04-11 23:15] Draft 2 complete
- Produced `work/researcher/direction_4/run_2/impl_spec_draft_2.md`
- All 9 critique 1 issues addressed

### [2026-04-11 23:26] Critique 2 received - assessment
- 0 major issues, 5 minor issues -- all valid, straightforward to address
- m1 (MOST IMPORTANT): Rolling regression needs concrete specification --
  OLS type, training window, feature specification, update frequency,
  no-intercept decision. Critic suggests the p. 19 "without constant term"
  note may apply to surprise regression too.
- m2: "Today" label in Exhibit 1 needs explicit interpretation comment
- m3: Joint constraint per-bin vs per-symbol design choice needs acknowledgment
- m4: Inter-day ARMA cap at 8 terms needs Researcher inference marking
- m5: Multi-step intraday prediction error accumulation needs discussion

### [2026-04-11 23:26] Re-read paper to verify critique claims
- Exhibit 1 (p. 18): Confirmed "Today" appears as a separate labeled input
  arrow to the ARMA Intraday box, alongside "Current Bin" and "4 Bins Prior
  to Current Bin"
- p. 19: Confirmed "Note that we perform both regressions without the
  inclusion of a constant term" -- "both regressions" refers to the two
  VWAP-error regressions (volume percentage error vs VWAP tracking error),
  but could plausibly extend to the surprise regression given the paper's
  general preference for no-intercept models
- p. 18: Confirmed "dual ARMA model having fewer than 11 terms" is
  per-symbol, but inter-day ARMA is per-bin -- the per-bin vs per-symbol
  tension is real and needs acknowledgment
- p. 18: Confirmed multi-step ARMA prediction discussed only implicitly
  via the "predict only the next interval" advantage of the percentage model

### [2026-04-11 23:27] Writing implementation specification draft 3
- Addressing all 5 minor critique 2 issues:
  - m1 (rolling regression detail): Completely rewrote Part B Step 3 with
    concrete OLS specification. Added explicit training loop that pools
    (predictor, response) pairs across N_regression_fit historical days.
    Specified OLS without intercept, consistent with paper's p. 19
    preference. Added N_regression_fit=63 parameter. Added Step 3a
    (training) and Step 3b (prediction) as separate substeps. Added
    initialization item #7 for surprise regression coefficients. Updated
    calibration Phase 1 step 1e and Phase 3.
  - m2 ("Today" label): Added explicit interpretation comment in Component 3
    header: "Today" is the data source qualifier (today's observed bins),
    not a separate feature. Updated data flow diagram annotation.
  - m3 (joint constraint per-bin vs per-symbol): Added Researcher inference
    comment in the joint constraint section explaining the conservative
    max-across-bins approach and the alternative (median or per-bin).
    Updated data flow diagram to note "binding = max across bins."
  - m4 (inter-day cap at 8): Marked the 8-term cap as Researcher inference
    with explanation of the budget split rationale. Added max_interday_budget
    parameter to the parameter table. Updated paper references table.
  - m5 (multi-step prediction degradation): Added Researcher inference
    comment in Step 3d with discussion of degradation, convergence to
    unconditional mean, and optional horizon capping. Added as Known
    Limitation #10. Updated calibration Phase 1 step 2a.
  - Also added: sanity check #11 (surprise regression coefficient sign),
    edge case #10 (insufficient bins for regression), updated edge case #2
    (insufficient history for regression), updated edge case #4 (first bin
    of day for percentage model).
  - Updated data flow diagram with rolling regression details.
  - Added surprise regression beta to type specifications table.
  - Added 6 new entries to paper references table.

### [2026-04-11 23:28] Draft 3 complete
- Produced `work/researcher/direction_4/run_2/impl_spec_draft_3.md`
- All 5 critique 2 issues addressed

### [2026-04-11 23:39] Critique 3 received - assessment
- 0 major issues, 2 minor issues (cosmetic)
- Critic declares draft 3 "ready for implementation" and recommends no
  further revision rounds
- m1: Division-by-zero in renormalization when j+1 == I (last bin). The
  scale_factor denominator becomes zero. Fix: guard with IF j+1 < I.
- m2: N_interday_predict Range column says "max(p, 5)" which is confusing;
  the true minimum is p, not 5. Fix: clarify Range text.
- Both fixes are trivial single-line changes.

### [2026-04-11 23:39] Writing implementation specification draft 4
- Applied both cosmetic fixes from critique 3:
  - m1: Added IF j+1 < I guard around scale_factor computation in Part B
    Step 5 renormalization. When j+1 == I, no redistribution is needed.
  - m2: Changed N_interday_predict Range from "max(p, 5)" to
    "p minimum; 5 is a safe default covering all candidate AR orders (1-5)"

### [2026-04-11 23:40] Draft 4 complete
- Produced `work/researcher/direction_4/run_2/impl_spec_draft_4.md`
- Both cosmetic critique 3 issues addressed
- Specification is at final quality per critic's assessment

# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model

## Summary Assessment

Draft 2 is a substantial improvement over draft 1. All 3 major issues and all 8 minor
issues from critique 1 have been thoroughly addressed. The EM M-step ordering (phi
before r) is now explicit with clear rationale, Algorithm 1's output spec includes all
quantities needed by the smoother, the cross-covariance recursion is now a full
pseudocode section, and the log-likelihood convergence criterion is properly specified.
The MAPE definition, log-normal bias note, and Researcher inference labels are all in
good shape.

I identify **1 major issue** and **5 minor issues** remaining. The major issue is a
misreported benchmark number (RM VWAP tracking error). The minor issues are
completeness and clarity gaps that would not cause incorrect implementation but would
slow a developer down.

---

## Major Issues

### M1. Rolling Mean VWAP Tracking Error Reports Standard Deviation Instead of Mean

**Location:** Validation > Expected Behavior, line ~829.

The spec states:

> Rolling Mean (dynamic VWAP): average 10.68 bps.

This is **wrong**. The value 10.68 is the **standard deviation** of the RM VWAP tracking
error, not the mean. Table 4's bottom row ("Average") has two columns for RM: the first
is the mean (7.48), the second is the standard deviation (10.68). Every other model in
Table 4 follows the same (mean, std) column pair pattern.

Three independent checks confirm 7.48 is the correct mean:

1. The paper summary (`papers/chen_feng_palomar_2016.md`) correctly reports "RM: 7.48 bps."
2. The paper text (Section 4.3, page 11) states "an improvement of 15% compared with
   the RM." Checking: (7.48 - 6.38) / 7.48 = 14.7% ~ 15%. With 10.68, the improvement
   would be 40%, contradicting the paper's stated 15%.
3. The spec's own line ~843 claims "15% VWAP tracking error improvement over Rolling
   Mean," which is only consistent with RM = 7.48, not 10.68.

Additionally, the label "Rolling Mean (dynamic VWAP)" is misleading. RM does not
perform dynamic VWAP -- it produces static weights from a rolling average. Table 4
presents RM in its own column outside both the "Dynamic VWAP Tracking" and "Static VWAP
Tracking" sections. The RM weights are inherently static (computed before market open);
the tracking error measures how well these static RM-based weights replicate the true
VWAP.

**Fix:** Replace "Rolling Mean (dynamic VWAP): average 10.68 bps" with "Rolling Mean:
average 7.48 bps." Drop the "(dynamic VWAP)" qualifier since RM uses static weights.

**Source:** Paper, Table 4, bottom row "Average", RM columns (mean = 7.48, std = 10.68).

---

## Minor Issues

### m1. VWAP Tracking Error Formula Not Included

**Location:** Validation > Expected Behavior.

The spec cites VWAP tracking error benchmarks (lines 826-831) but does not include the
formula used to compute them. The paper defines this in Equation 42:

    VWAP^TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t

where D is the number of out-of-sample days, VWAP_t is the true VWAP for day t, and
replicated_VWAP_t = sum_{i=1}^{I} w_{t,i} * price_{t,i} is the model-based VWAP
replica using predicted weights and last-transaction prices.

A developer implementing the evaluation pipeline needs this formula alongside the MAPE
formula (which is already included). Without it, reproducing the tracking error benchmarks
requires going back to the paper.

**Fix:** Add the VWAP tracking error formula (Eq 42) and the VWAP definition (Eq 39)
to the Validation section, noting that errors are expressed in basis points (multiply
by 10,000) and that price_{t,i} is the last transaction price in bin i of day t.

**Source:** Paper, Section 4.3, Equations 39 and 42.

---

### m2. M-Step Has Additional Implicit Ordering Dependencies Beyond phi-before-r

**Location:** Algorithm 3 pseudocode, M-step, lines ~310-397.

The spec clearly documents the phi-before-r ordering constraint. However, there are
two additional ordering dependencies that are not called out:

1. **sigma^eta depends on a^eta:** The update for (sigma^eta)^2 (Eq A.36 / Eq 21) uses
   (a^eta)^{(j+1)} -- the newly updated AR coefficient from the current M-step. The
   (j+1) superscript is explicit in the paper's equation.

2. **sigma^mu depends on a^mu:** Similarly, (sigma^mu)^2 (Eq A.37 / Eq 22) uses
   (a^mu)^{(j+1)}.

The spec's pseudocode happens to compute a^eta before sigma^eta and a^mu before sigma^mu,
so the ordering is implicitly correct. But a developer reordering the M-step updates
(e.g., grouping all variance updates together) could break these dependencies.

**Fix:** Add a brief note after the M-step pseudocode listing all ordering constraints:
(1) phi before r (already documented), (2) a^eta before sigma^eta, (3) a^mu before
sigma^mu. Note that pi_1, Sigma_1 have no dependencies on other M-step parameters.

**Source:** Paper, Appendix A.3, Equations A.34-A.37. The (j+1) superscripts on a^eta
and a^mu within the sigma equations establish these dependencies.

---

### m3. Static-to-Dynamic Mode Transition for Evaluation Not Described

**Location:** Algorithm 1 pseudocode and Calibration procedure.

Algorithm 1 accepts a `mode` parameter that is either "dynamic" or "static" for the
entire run. But the out-of-sample evaluation requires switching modes within a single
day's evaluation:

- **Static prediction:** Run the filter dynamically through the training window to
  establish the current state, then switch to static mode (no corrections) for the
  target day's I bins.
- **Dynamic prediction:** Continue running the filter dynamically through the target
  day, using each bin's observation before predicting the next.

The current description does not explain how this mode transition works operationally.
A developer might interpret "static mode" as running the entire history without
corrections (which would give terrible results) rather than running dynamically up to
the prediction boundary and then switching to prediction-only.

**Fix:** Add a paragraph in the Calibration section (or after Algorithm 1) describing
the evaluation loop:
1. For static evaluation of day d: run the filter dynamically through day d-1 (all
   bins), then run in static mode for bins 1..I of day d.
2. For dynamic evaluation of day d: run the filter dynamically through all bins including
   day d, and collect the one-step-ahead predictions y_hat for day d's bins.
Note that for both modes, the initial state for day d comes from the filter's posterior
at the end of day d-1.

**Source:** Paper, Section 4.2, first paragraph: "The static prediction refers to
forecasting all the I bins of day t + 1 using information up to day t only. The dynamic
prediction stands for the one-bin-ahead forecasting."

---

### m4. Dynamic VWAP Multi-Step Predictions for Remaining Bins Could Use Pseudocode

**Location:** VWAP Execution Strategies, lines ~569-575.

The dynamic VWAP formula (Eq 41) requires volume_hat_{t,j}^(d) for bins j = i+1 through
I in the denominator. The spec correctly states these are "multi-step-ahead predictions
from the current filtered state" (not pre-market static forecasts, addressing critique 1
item m3). However, the procedure for computing these multi-step predictions is described
only in prose.

A developer needs to:
1. Start from x_hat[tau_i] (filtered state after correcting with bin i's observation,
   or after correcting with bin i-1 if computing the forecast for bin i).
2. For each remaining bin j = i+1, i+2, ..., I: propagate x_pred forward using the
   appropriate A matrix (all within-day, so A = [[1,0],[0,a^mu]]), and compute
   y_hat_j = C @ x_pred_j + phi_j.
3. Convert to volume: volume_hat_j = exp(y_hat_j).

This is conceptually simple but involves a nested loop (for each bin i, propagate
forward through bins i+1..I), and the distinction between the filtered state used as
the starting point for bin i vs. bin i+1 is subtle.

**Fix:** Add a brief pseudocode snippet showing the multi-step prediction loop for the
dynamic VWAP denominator, or a note clarifying that within a single day, all transitions
use A = [[1,0],[0,a^mu]] and Q plays no role in the point forecast (only in uncertainty).

**Source:** Paper, Section 4.3, Equation 41 and surrounding text.

---

### m5. VWAP Tracking Error Static Results for RM Ambiguous

**Location:** Validation > Expected Behavior, VWAP tracking error section.

The VWAP tracking error section (lines 826-831) lists results for Robust KF, Standard
KF, CMEM, and RM, all under the heading "VWAP tracking error." However, the three
Kalman/CMEM entries are labeled "(dynamic VWAP)" while RM's tracking error comes from
a static weighting scheme.

This creates a subtle apples-to-oranges comparison: the reader might think all four
numbers use the same VWAP strategy (dynamic), when in fact RM uses static weights.
The paper's Table 4 separates dynamic and static VWAP columns for the KF models but
gives RM only once (since RM is always static).

For a complete comparison, the spec should also include the static VWAP tracking errors
for the KF models:
- Robust KF (static VWAP): average 6.85 bps.
- Standard KF (static VWAP): average 7.09 bps.
- CMEM (static VWAP): average 7.71 bps.

This would let the developer validate both VWAP strategies and make the RM comparison
fair (RM static vs. KF static).

**Fix:** Add a "Static VWAP tracking error" subsection with the static VWAP results
from Table 4, and note that the RM benchmark should be compared against static VWAP
results since RM uses static weights.

**Source:** Paper, Table 4, bottom row, static VWAP columns.

---

## Positive Observations

The following improvements from draft 1 to draft 2 are well executed:

1. **Algorithm 1 output spec** now comprehensively lists all quantities needed
   downstream (x_pred, Sigma_pred, A_store, K_store, S_store) with clear notes on why
   each is stored.
2. **EM M-step ordering** is thoroughly documented with both the mathematical rationale
   (phi^{(j+1)} superscript in Eq A.38) and the implementation consequence. The compact
   form of the r update is a nice addition.
3. **Cross-covariance recursion** (Algorithm 2 Part 2) is now a proper pseudocode section
   with clear initialization, loop structure, and index mapping. The note about K_N
   being required from Algorithm 1 is explicit.
4. **Innovation-form log-likelihood** for EM convergence is correctly specified with the
   formula, and the relative change criterion is well-defined.
5. **Robust EM E-step** clearly states that Algorithm 4 is used in the forward pass, the
   standard smoother in the backward pass, and z_star values are stored for the M-step.
6. **Log-normal bias note** in the Data Flow section is well-balanced: it explains the
   issue, provides the correction formula, and recommends matching the paper's approach
   for benchmarking while noting the corrected form for other applications.
7. **Stationarity clamping** for a^eta and a^mu includes both the clamp logic and a
   warning when values fall outside (0,1), correctly labeled as Researcher inference.
8. **Paper References table** is comprehensive and includes Researcher inference items.
9. **Researcher inference labels** are consistently applied throughout (phi initialization,
   rolling window warm-start, iteration budget, zero-volume handling, etc.).

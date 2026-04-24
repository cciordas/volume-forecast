# Critique of Implementation Specification Draft 1: Direction 7, Run 7

## Summary

This is a strong draft. The specification is well-structured, comprehensive (1022
lines), and covers the full pipeline from preprocessing through VWAP execution and
cross-validation. The unified standard/robust code path via z_star=0 is a good design
decision. The non-recursive cross-covariance formula and Sigma_1 = Sigma_smooth[1] are
both improvements over the paper's literal equations. The 42-entry paper reference table
is thorough.

**Issues found:** 6 major, 9 minor.

---

## Major Issues

### M1. Convergence criterion uses absolute tolerance, not relative (Step 7, line 417-418)

The spec computes:

    relative_change = abs(log_lik - log_lik_prev) / max(abs(log_lik_prev), 1.0)

But the variable name is misleading and the guard value of 1.0 is problematic. The
innovation log-likelihood scales with N (number of bins): for N=6500 (250 days x 26
bins), typical values are O(-10000). With tol=1e-8 and a guard denominator of
max(10000, 1.0) = 10000, the effective absolute threshold is 10000 * 1e-8 = 1e-4.
This is reasonable. However, for short windows (N=520, 20 days), log-lik is O(-1000),
giving threshold 1e-5 — tighter. And during early iterations when log_lik_prev =
-infinity, the guard value of 1.0 kicks in, making the first convergence check
meaningless (any finite change satisfies the criterion).

**Recommendation:** The `j > 1` guard on line 418 prevents premature exit on the first
iteration, which is correct. But document that tol=1e-8 is *relative* tolerance, and
that convergence speed may vary with dataset size due to the log-likelihood scale. Also
add the parameter-change fallback criterion mentioned in Step 6 (line 391) as an
explicit secondary check in Step 7, not just as a comment in Step 6.

### M2. M-step phi formula uses count_i denominator but paper uses T (Step 5, lines 328-335)

The spec's phi formula (line 332) uses `count_i = |bins_i|` (number of observed bins
for position i) as the denominator. The paper's Equation A.39 uses T (total number of
training days) as the denominator:

    phi_i^{(j+1)} = (1/T) * sum_{t=1}^{T} (y_{t,i} - C @ x_hat_{t,i})

And the robust variant Equation 36 uses:

    phi_i^{(j+1)} = (1/T) * sum_{t=1}^{T} (y_{t,i} - C @ x_hat_{t,i} - z*_{t,i})

The paper assumes all bins are observed (Section 4.1: zero-volume bins are excluded
from the dataset entirely). The spec's use of count_i is a researcher inference to
handle missing data, which is a valid extension. However, this should be explicitly
marked as a departure from the paper, with discussion of whether the unobserved terms
(y[tau] undefined, but x_smooth[tau] still defined via prediction pass-through) should
contribute to the sum. Currently, unobserved bins are excluded from both numerator and
denominator, which means phi[i] is the average residual over *observed* days only.
This is one reasonable choice, but the alternative (including x_smooth[tau] terms for
unobserved bins with y[tau] imputed as C @ x_smooth[tau] + phi[i], which zeroes out
the contribution) would recover the paper's T denominator. The spec should state which
choice it makes and why.

### M3. M-step r formula uses N_obs denominator but paper uses N (Step 5, lines 339-346)

Same issue as M2 but for r. Paper Equation A.38 sums over all tau=1..N and divides by
N. The spec's line 346 uses `N_obs` (count of observed bins). This is a valid extension
for missing data, but it changes the statistical interpretation: the paper's r
represents the total observation noise variance averaged over all bins; the spec's r
represents it averaged over observed bins only.

For the standard case (no missing data), N_obs = N and the formulas are identical. For
datasets with missing bins, the choice matters. The robust Equation 35 in the paper
also sums over all tau and divides by N.

**Recommendation:** Mark this as researcher inference with explicit justification. Also
verify that the compact form on line 344-345:

    residual = y[tau] - phi_tau - z_star[tau] - C @ x_smooth[tau]
    r += residual^2 + C @ Sigma_smooth[tau] @ C^T

correctly expands to match Equation A.38. The compact form groups terms as
(y - phi - z - C@x)^2 + C@Sigma@C^T. Expanding (y - phi - z - C@x)^2 via
P = Sigma + x@x^T and collecting gives:

    y^2 + C@P@C^T - 2y*C@x_s + phi^2 - 2y*phi + 2*phi*C@x_s + z^2 - 2y*z + 2*z*C@x_s + 2*z*phi

while Equation A.38 (with phi^{j+1}) is:

    y^2 + C@P@C^T - 2y*C@x_s + (phi^{j+1})^2 - 2y*phi^{j+1} + 2*phi^{j+1}*C@x_s

The compact form is correct for the standard case (z=0). For the robust case (z != 0),
the compact form includes additional cross-terms involving z that match Equation 35
when expanded correctly. This is fine but should be verified algebraically in the spec.

### M4. Dynamic VWAP uses x_pred_i for volume forecast but calls StaticForecastRemaining with x_pred_i (Step 11, lines 566-568)

In Step 11 (line 566-568):

    x_pred_i = A @ x_filt_prev
    volume_hat = StaticForecastRemaining(x_pred_i, i, I, theta)   # Step 10

But Step 10 (line 528-529) says:

    if h == bin_start:
        # No additional transition needed; x_state is already the predicted state
        pass

This means the first bin (h = bin_start = i) uses x_pred_i directly without any
further transition. For bin i, the predicted log-volume is:

    y_hat[i] = C @ x_pred_i + phi[i]

This is correct: x_pred_i is the predicted state for time i (conditioned on info up to
bin i-1), and C @ x_pred_i + phi[i] is the one-step-ahead prediction.

However, for subsequent bins h > bin_start, Step 10 applies A = [[1,0],[0,a_mu]]
(within-day transition). This means the forecast for bin i+1 uses:

    x_curr = [[1,0],[0,a_mu]] @ x_pred_i
    y_hat[i+1] = C @ x_curr + phi[i+1]

But this applies the within-day transition to x_pred_i (which is already a predicted
state, not a filtered state). The correct multi-step prediction from a predicted state
at time i should propagate the *predicted* state forward, which is exactly what this
does. So the math is correct.

**However**, there is a subtle issue: Step 10's x_state comes from Step 11's x_pred_i,
which was computed using x_filt_prev (the filtered state from bin i-1). But Step 9
(DynamicPredict, called on line 578) also computes x_pred internally and then corrects
it. The state passed to Step 10 and the prediction used in Step 9 should be the same
x_pred for consistency. Currently they appear to be: Step 11 computes x_pred_i for
weight computation, then calls Step 9 which recomputes x_pred internally. This is
redundant but not incorrect.

**Recommendation:** Document that Step 10's volume forecasts use the *pre-correction*
state (conditioned on information through bin i-1), consistent with Equation 41's
conditioning. Add a note that x_pred_i in Step 11 and the internal x_pred in Step 9
are identical, so the redundant computation is intentional (clearer code vs. efficiency).

### M5. Missing day-boundary Q matrix in dynamic VWAP (Step 11, lines 560-564)

Step 11 constructs A and Q for the day boundary at i==1 (lines 560-564) but only uses
A to compute x_pred_i. The Q matrix is never used in Step 11 — it's not passed to
StaticForecastRemaining (Step 10), which also doesn't use Q. This means Step 10 does
not propagate Sigma (covariance) at all — it only propagates the state mean.

For the weight computation, only the predicted volumes (not their uncertainties) are
needed, so this is correct for VWAP weights. But if the log-normal bias correction
(optional, mentioned in Step 8) were to be applied in the dynamic VWAP context, the
covariance propagation would be needed.

**Recommendation:** Either (a) add a note that Step 10 intentionally omits covariance
propagation since VWAP weights use only volume ratios (the bias correction cancels in
ratios), or (b) extend Step 10 to optionally propagate Sigma_curr alongside x_curr for
use with bias-corrected forecasts.

### M6. Sigma_1 initialization in EM init differs from M-step (Step 1 vs Step 5)

In Step 1 (EM initialization, line 126):

    Sigma_1 = diag(var(daily_avg), var(resid[observed]))

In Step 5 (M-step, line 291):

    Sigma_1 = Sigma_smooth[1]

The initialization uses a diagonal matrix with empirical variances, which is reasonable.
But after the first EM iteration, Sigma_1 becomes Sigma_smooth[1], which is generally
non-diagonal (the smoother induces cross-covariance between eta and mu). This is
correct behavior — the initialization is a rough starting point, and the M-step refines
it. No code change needed, but the spec should note that Sigma_1 transitions from
diagonal (init) to non-diagonal (after first EM iteration), so the developer should
not assume or enforce diagonality.

---

## Minor Issues

### m1. D_start definition uses confusing notation (Step 5, line 279)

The spec defines D_start = {kI + 1 for k = 1, ..., T-1} with |D_start| = T-1. This
is correct but could confuse a developer because k ranges over 1 to T-1, making the
first element I+1 (first bin of day 2) and the last element (T-1)*I + 1 (first bin of
day T). The tau-1 in the summation (e.g., P[tau-1] on line 302) refers to the last bin
of the previous day (tau - 1 = kI), which is the bin where the day-boundary transition
originates.

**Recommendation:** Add a one-line comment: "D_start contains destination indices (first
bin of each day except day 1). For tau in D_start, the transition from tau-1 to tau
crosses a day boundary."

### m2. Step 2 tau=1 handling is inconsistent with initialization comment (lines 147-149)

Line 147-148 says:

    # tau = 1: use initial state as prediction (no transition applied)
    x_pred[1] = pi_1
    Sigma_pred[1] = Sigma_1

Then the loop starts at tau=1 and applies the correction step. This means the
prediction at tau=1 is just the initial state, and the first correction happens at
tau=1. This is correct and standard.

However, the `if tau >= 2` block on line 152 means A_used[1] is never set. The RTS
smoother (Step 3, line 235) accesses A_used[tau+1] for tau = N-1 downto 1, so
A_used[2] through A_used[N] are needed. A_used[1] is never accessed. This is correct
but implicit — add a note that A_used is indexed from 2 to N.

### m3. Smoother L_stored indexing convention (Step 3, line 236)

L_stored is indexed from 1 to N-1 (line 236: `L_stored[tau] = L` for tau from N-1
downto 1). In Step 4 (line 262), the cross-covariance uses L_stored[tau-1] for
tau = 2 to N. So L_stored[1] through L_stored[N-1] are accessed. This is consistent,
but the spec should note that L_stored[tau] corresponds to the smoother gain at time
tau, which connects the smoothed state at tau to the smoothed state at tau+1.

### m4. Step 8 static prediction starts from x_filt[N] but should clarify (lines 443-457)

Step 8 says the input is x_filt_last = x_filt[N], the filtered state at the last
training bin. This is correct for predicting the next day after training. But in a
rolling-window production context, the "last bin" is the last bin of the current day,
and the model needs to predict the next day. The spec should clarify that x_filt_last
comes from running the Kalman filter (forward only, no smoother) through the most
recent data, not from the EM training's last iteration's filter output (which may be
from months ago if the training window is fixed).

Actually, re-reading Step 7 (line 429): `x_filt_final = x_filt` — this IS the filter
output from the last EM iteration, run on the training data. For prediction of the
next day after training, this is the correct starting point. But Step 13 (rolling
window) re-estimates theta daily. After re-estimation, the developer needs to run the
filter forward through the most recent day's data to get x_filt for the end of that
day. The spec should note this operational detail: after EM converges, x_filt[N] from
the last E-step is the appropriate starting state for next-day prediction.

### m5. Per-ticker MAPE for AAPL (line 828)

The spec states "Best: AAPL at 0.21 (mean), std 0.17." Looking at Table 3 in the
paper, AAPL's dynamic robust KF MAPE is 0.21 with std 0.17 — this is correct.
However, the spec says "2800HK (leveraged ETF) at 1.94" as the worst. Table 3 shows
2800HK's dynamic robust KF MAPE is 1.94 with std 9.12 — correct. But 2800HK is
described in Table 2 as "Tracker Fund of Hong Kong ETF", not a leveraged ETF. The
leveraged ETF in the dataset is 1570JP ("Topix-1 Nikkei 225 Leveraged ETF").

**Recommendation:** Correct "2800HK (leveraged ETF)" to "2800HK (Tracker Fund of Hong
Kong ETF)" or simply "2800HK".

### m6. Cross-validation MAPE formula unclear (Step 13, line 636)

Line 636:

    mape_d = mean(|exp(y_actual) - exp(y_pred)| / exp(y_actual))

This converts from log-space to volume-space for MAPE computation, matching Equation
37. However, `y_pred` in dynamic mode should be the *one-step-ahead prediction* (before
correction), not the filtered estimate (after correction). The spec calls
`DynamicPredictDay` (line 633) but this function is not defined. A developer would need
to implement it as: for each bin i, compute prediction y_hat from previous filtered
state, record y_hat for MAPE, then correct using y_actual[i].

**Recommendation:** Either define DynamicPredictDay explicitly or add a one-line
description: "DynamicPredictDay runs the Kalman filter forward through day d's bins,
recording each one-step-ahead prediction y_hat[i] before correction."

### m7. EM main loop returns x_filt but not x_smooth (Step 7, line 428-429)

Step 7 returns x_filt_final and Sigma_filt_final but not the smoothed states. The
smoothed states from the last EM iteration could be useful for diagnostic purposes
(e.g., visualizing the decomposition of historical volume into eta, mu, phi). This is
not a correctness issue — prediction only needs the filtered state. But a developer
implementing visualization/debugging tools would need the smoothed states.

**Recommendation:** Optionally return x_smooth_final and Sigma_smooth_final from the
last EM iteration for diagnostics.

### m8. No explicit MAPE formula definition in Validation section

The Validation section references MAPE values from Tables 3 and 4 but doesn't provide
the formula. Equation 37 from the paper defines:

    MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M = D * I is the total out-of-sample bins. This should be included in the
Validation section since it's used in both sanity checks and cross-validation.

### m9. Tracker findings F74 (unified z_star=0) correctly implemented but citation missing

The spec implements the unified code path (z_star initialized to 0 for standard mode,
lambda=1e10 makes threshold huge so z_star stays 0). This is well-designed but the
Paper References table (line 1012) cites "Researcher inference" for parameter clamping
without noting that the unified standard/robust code path is also researcher inference.
The table entry on line 999-1002 covers the robust formulas individually but doesn't
have an explicit entry for "unified code path via z_star=0 initialization = researcher
inference."

---

## Positive Aspects

1. **Non-recursive cross-covariance** (Step 4): Using Sigma_smooth[tau] @ L[tau-1]^T
   avoids the paper's recursive form (A.20-A.21) and the need to store K[N]. Well
   justified with citation to Shumway & Stoffer.

2. **Joseph form covariance update** (Step 2): Guarantees PSD preservation. Correctly
   identified as standard practice not in the paper.

3. **Unified standard/robust code path**: Elegant design that avoids code duplication.

4. **M-step ordering constraints**: All three dependencies clearly documented with
   equation citations.

5. **Comprehensive parameter table**: 13 parameters with sensitivity ratings and ranges.

6. **Data flow diagram**: Clear ASCII visualization with types/shapes table.

7. **42-entry paper reference table**: Thorough traceability with researcher inference
   items explicitly marked.

8. **10 sanity checks**: Good coverage including synthetic recovery, monotonicity, and
   multi-initialization convergence.

---

## Verdict

This is a high-quality draft that a developer could implement from with confidence. The
6 major issues are primarily about documentation clarity (M1, M4, M5, M6) and
explicitly marking departures from the paper (M2, M3) rather than algorithmic errors.
The core algorithms are correct. One round of revision addressing the major issues
should produce a final-quality specification.

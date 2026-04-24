# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model for Intraday Volume

## Summary

The draft is well-structured, thorough, and largely faithful to the paper. It covers
the core algorithm (Kalman filter, RTS smoother, EM), the robust Lasso extension,
and VWAP execution. The pseudocode is detailed enough to implement from. However,
I identified 5 major issues (would cause incorrect implementation) and 8 minor issues
(clarity, completeness, or potential confusion).

---

## Major Issues

### M1. Kalman Filter Prediction/Correction Ordering is Inverted (Step 3, lines 172-199)

The pseudocode in Step 3 starts the loop at tau=1 and performs the **prediction step
first**, computing x_hat[tau+1|tau] before the correction step for tau+1. This is
inverted from the standard Kalman filter and from Algorithm 1 in the paper.

The paper's Algorithm 1 (page 4) processes each tau as:
1. Lines 2-3: **Predict** x_hat[tau+1|tau] and Sigma[tau+1|tau] from x_hat[tau|tau].
2. Line 4: Compute Kalman gain K_{tau+1} using Sigma[tau+1|tau].
3. Lines 5-6: **Correct** to get x_hat[tau+1|tau+1] and Sigma[tau+1|tau+1].

The spec's pseudocode does follow this order, but the loop structure is confusing
because it initializes with x_hat[1|0] and then in the loop body computes prediction
for tau+1 followed by correction for tau+1. The issue is that the **observation forecast**
at line 179 computes:

    y_hat[tau+1|tau] = C * x_hat[tau+1|tau] + phi_{tau+1}

This is placed in the "prediction step" section, which is correct. However, a developer
might misread this as: "predict tau+1, then correct tau+1, then predict tau+2." The
actual flow is: predict tau+1 state -> observe y_{tau+1} -> correct tau+1 state ->
predict tau+2 state -> ...

**Recommendation:** Restructure the loop to iterate over the observation index directly.
Make it clear that at each iteration, we (a) have x_hat[tau|tau-1] from the previous
iteration, (b) compute the forecast, (c) observe y_tau, (d) correct to x_hat[tau|tau],
(e) predict x_hat[tau+1|tau]. This matches Algorithm 1 more directly. Also explicitly
show the initialization producing x_hat[1|0] and then the first iteration starting
with tau=1.

### M2. M-Step a_eta Update Uses Wrong Index Set (Step 2, lines 108-110)

The spec defines D = {tau : tau = k*I+1, k=1,2,...} as the set of day-boundary indices.
This is the set of **first bins of each day** (i.e., the index right after the boundary).
This is correct for the summation -- we want transitions where eta actually changes,
which happen when transitioning from tau-1 (last bin of day k) to tau (first bin of
day k+1).

However, the paper's Equation A.34 (page 15) sums over tau = kI+1, consistent with
the spec. But compare with the sigma_eta^2 update in the spec (line 117): the
denominator uses (T-1), and the sum is over D. The paper's Equation A.36 (page 15)
shows:

    [(sigma^eta)^2]^{(j+1)} = 1/(T-1) * sum_{tau=kI+1} {P_tau^(1,1) + (a^eta)^{(j+1)^2} * P_{tau-1}^(1,1) - 2*a^eta^{(j+1)} * P_{tau,tau-1}^(1,1)}

The spec correctly uses (T-1) as the denominator for sigma_eta^2 (line 117). But the
denominator for a_eta (line 110) uses sum of P_{tau-1}^(1,1) over D, which is correct
per Eq A.34.

**However**, there is a subtle issue: the set D should have exactly T-1 elements (one per
day boundary, from day 1->2 through day T-1->T). The spec defines D = {tau : tau = k*I+1,
k=1,2,...} which starts at k=1 (tau=I+1) and goes to k=T-1 (tau=(T-1)*I+1). This gives
T-1 elements. This is correct but not stated explicitly. The developer needs to know
that |D| = T-1 and that the first element is tau = I+1 (not tau=1).

**Recommendation:** Explicitly state |D| = T-1 and list the boundary indices: D = {I+1,
2I+1, 3I+1, ..., (T-1)*I+1}. Also clarify that k ranges from 1 to T-1, not unbounded.

### M3. Robust EM r Update (Step 5, lines 300-307) Includes phi but Paper Equation A.38 Already Includes phi

The robust EM r update in lines 300-307 is presented as a modification of Eq A.38,
adding z_tau_star terms. Comparing with the paper's Equation 35 (page 7):

    r^{(j+1)} = 1/N * sum [y_tau^2 + C*P_tau*C^T - 2*y_tau*C*x_hat_tau + (phi_tau^{(j+1)})^2
                           - 2*y_tau*phi_tau^{(j+1)} + 2*phi_tau^{(j+1)}*C*x_hat_tau
                           + (z_tau*)^2 - 2*z_tau*y_tau + 2*z_tau*phi_tau^{(j+1)} + 2*z_tau*C*x_hat_tau]

The spec's lines 302-307 match this equation. This is correct.

**However**, there is a sign issue to verify. In the paper's Eq 35, the last term is
+2*z_tau* * C * x_hat_tau. But thinking about the derivation: the residual in the
robust model is (y_tau - phi_tau - C*x_tau - z_tau), so expanding the square:

    E[(y - phi - Cx - z)^2] = y^2 + phi^2 + C*P*C^T + z^2
                              - 2*y*phi - 2*y*Cx - 2*y*z
                              + 2*phi*Cx + 2*phi*z + 2*z*Cx

This matches Eq 35 and the spec. The signs are correct.

**Retracted as major issue.** Downgraded -- the equation is correct. But see M4 below.

### M3 (revised). Log-Likelihood Formula for EM Convergence Check is Missing

The spec says (line 141): "IF |log_likelihood^(j+1) - log_likelihood^(j)| < epsilon: BREAK"
but never defines how to compute the log-likelihood. The developer needs to know what
to compute.

The paper provides the full joint log-likelihood in Eq A.8 (page 13), but the EM
actually maximizes the expected complete-data log-likelihood Q (Eq A.10). However,
for convergence monitoring, the standard practice is to compute the **observed-data
log-likelihood**, which for a linear Gaussian state-space model can be computed from
the Kalman filter innovations:

    LL = -N/2 * log(2*pi) - 1/2 * sum_{tau=1}^{N} [log(S_tau) + e_tau^2 / S_tau]

where S_tau = C * Sigma[tau|tau-1] * C^T + r is the innovation variance and
e_tau = y_tau - C * x_hat[tau|tau-1] - phi_tau is the innovation.

This is a standard result but is not stated in the paper and is not in the spec.
Without it, a developer cannot implement the convergence check.

**Recommendation:** Add the innovation-based log-likelihood formula explicitly in the
EM pseudocode, with a note that this is a standard Kalman filter result (cite
Shumway and Stoffer 1982 or any state-space textbook).

### M4. Robust Kalman Filter: e_tau Index Mismatch Between Definition and Usage (Step 5)

In Step 5 (line 265), the innovation is defined as:

    e_tau = y_tau - phi_tau - C * x_hat[tau|tau-1]

But in Step 3 (line 188), the innovation is defined as:

    e[tau+1] = y[tau+1] - y_hat[tau+1|tau]

These use different indexing conventions. Step 3 uses a "predict tau+1, then correct
tau+1" loop, while Step 5 defines everything at index tau. A developer implementing
both variants would encounter conflicting index conventions.

More importantly, in the paper's Eq 31 (page 7), the innovation is defined at index
tau+1:

    e_{tau+1} = y_{tau+1} - phi_{tau+1} - C * x_hat[tau+1|tau]

The spec's Step 5 defines it at index tau (line 265). This is internally consistent
within Step 5 but inconsistent with Step 3 and with the paper.

**Recommendation:** Use a consistent indexing convention throughout. Either always use
the paper's convention (predict and correct at tau+1 given tau), or always use a
"current time" convention. The cleanest approach: define the correction step as
operating at time tau, where we already have x_hat[tau|tau-1] from the previous
prediction, and the innovation is e_tau = y_tau - phi_tau - C * x_hat[tau|tau-1].
Apply this consistently to both the standard and robust filters.

### M5. Dynamic VWAP Re-forecasting Logic is Incorrect (Step 6, lines 340-350)

The spec's dynamic VWAP (lines 340-350) re-forecasts bins j+1..I using
"dynamic_predict(1 step ahead, using obs through bin j)." But this is wrong for
bins j+2..I: after observing bin j, a 1-step-ahead prediction gives bin j+1 only.
For bin j+2 you need a 2-step-ahead prediction, for bin j+3 a 3-step-ahead, etc.

The paper's Equation 41 (page 10) defines the dynamic weight using volume_{t,i}^{(d)},
which are "the dynamic volume predictions." The paper states (page 8, Section 4.2):
"The dynamic prediction stands for the one-bin-ahead forecasting, where the volume at
one particular bin is predicted based on all the information up to the previous bin."

So dynamic VWAP at bin j should:
1. Use the already-observed volumes for bins 1..j (actual, not predicted).
2. For bins j+1..I, use one-step-ahead predictions that will be made **as each bin
   arrives** -- not a batch re-forecast from bin j.

The spec's formulation implies a batch re-forecast of all remaining bins, which would
require multi-step-ahead predictions for bins j+2 onward (not 1-step-ahead). The
paper's actual dynamic strategy is simpler: at each bin j, you only need the
1-step-ahead forecast for bin j+1 to set w_{t,j+1}.

**Recommendation:** Rewrite the dynamic VWAP section to clarify that it is an online
procedure: after each bin j is observed, (1) update the Kalman filter with y_{t,j},
(2) produce the 1-step-ahead forecast for bin j+1, (3) compute w_{t,j+1} using Eq 41.
The denominator of Eq 41 requires the volume predictions for all remaining bins,
which at that point must use multi-step forecasts from the current state. Clarify
this tension explicitly -- the paper's notation is slightly ambiguous on whether the
denominator uses stale predictions or is re-computed.

---

## Minor Issues

### m1. Sufficient Statistics Notation (Step 2, lines 92-99)

The spec defines P_tau (line 93) as:

    P_tau = E[x_tau * x_tau^T | y_1..y_N, theta^(j)]

and then says (line 98):

    P_tau = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T

This is correct (P_tau is the second moment, not the covariance). However, the spec
also uses Sigma[tau|N] from the smoother (which is the covariance). A developer might
confuse P_tau with Sigma[tau|N]. The notation matches the paper (Eqs A.15-A.19), but
it would help to add a one-line note: "P_tau is the second moment (not the covariance).
P_tau = Cov + mean*mean^T."

### m2. Superscript Notation for Matrix Elements (Step 2, lines 109-126)

The M-step uses notation like P_{tau,tau-1}^(1,1) and P_{tau-1}^(2,2) without
defining what the superscript means. From context and the paper, P^(i,j) means the
(i,j) element of the 2x2 matrix P. This should be stated explicitly, because a
developer might think it refers to the EM iteration index.

### m3. Static Multi-Step Prediction (Step 3, lines 206-209)

The formula:

    x_hat[tau+h|tau] = A^h * x_hat[tau|tau]

uses "A^h" and notes this means "h applications." But A_tau is time-varying (it
changes at day boundaries). For static prediction at end of day t, predicting all I
bins of day t+1, all steps are intraday, so A_tau = [[1,0],[0,a_mu]] for all h steps.
The spec correctly notes this in the paragraph following (line 211: "eta_hat remains
constant"), but the formula A^h is misleading because it suggests a single matrix
raised to a power, when actually we need to identify which A applies at each step.

**Recommendation:** Replace A^h with the explicit intraday transition matrix:
A_intraday = [[1,0],[0,a_mu]], then x_hat[tau+h|tau] = A_intraday^h * x_hat[tau|tau],
which for the 2x2 diagonal case simplifies to [eta_hat, (a_mu)^h * mu_hat]^T.
Show this simplification explicitly.

### m4. Process Noise in Multi-Step Prediction Covariance is Missing (Step 3)

The spec gives the multi-step state forecast x_hat[tau+h|tau] but does not give the
multi-step covariance Sigma[tau+h|tau]. This is needed for:
1. Computing prediction intervals / forecast uncertainty.
2. The VWAP denominator if using probabilistic weights.
3. The innovation variance S at the first correction step of the next observed bin.

The multi-step covariance is:
Sigma[tau+h|tau] = A_intraday * Sigma[tau+h-1|tau] * A_intraday^T + Q_intraday

where Q_intraday = [[0,0],[0,sigma_mu^2]]. This should be stated.

### m5. MAPE Definition is Missing

The spec cites MAPE values extensively but never defines MAPE. The paper defines it
in Eq 37 (page 7):

    MAPE = 1/M * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of out-of-sample bins. The predicted volume is in the
linear scale (exp of log-volume forecast). This should be included in the Validation
section so the developer can reproduce the benchmark comparisons.

### m6. VWAP Tracking Error Definition is Missing

Similarly, the spec cites VWAP tracking error (bps) from Table 4 but never defines
the tracking error formula. The paper's Eq 42 (page 10) defines:

    VWAP_TE = 1/D * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t

where D = 250 out-of-sample days. Include this definition.

### m7. Lambda = 0 Edge Case Description is Wrong (line 578)

The spec states: "Lambda = 0 in robust model: Degenerates to the standard Kalman filter
(z_tau_star = e_tau always, removing all innovation signal)."

When lambda = 0, the threshold lambda/(2*W_tau) = 0, so the soft-thresholding solution
(Eq 33) gives:
- If e_tau > 0: z_tau* = e_tau - 0 = e_tau
- If e_tau < 0: z_tau* = e_tau + 0 = e_tau
- If e_tau = 0: z_tau* = 0

So z_tau* = e_tau, and the cleaned innovation e_tau - z_tau* = 0 for all observations.
This means the correction step has zero innovation -- the filter never updates from
observations. This is worse than "standard Kalman filter"; it makes the filter
**completely unresponsive to data**. The spec correctly identifies the symptom ("removing
all innovation signal") but incorrectly labels the result as "degenerates to the
standard Kalman filter."

**Recommendation:** Fix to: "Lambda = 0: z_tau* = e_tau for all bins, so the corrected
innovation is always zero. The filter ignores all observations and runs in pure
prediction mode. This is pathological, not equivalent to the standard Kalman filter."

### m8. No Guidance on Warm-Start vs. Cold-Start for Rolling Window Re-estimation

The Calibration section (line 476-486) describes daily re-estimation but does not
address whether to warm-start the EM from the previous day's estimated parameters or
cold-start from the default initialization each time. Warm-starting would likely
converge faster (fewer EM iterations needed), which matters for daily re-estimation.
The paper does not discuss this, but it is a practical implementation decision the
developer will face.

**Recommendation:** Add a note recommending warm-start (initialize EM with previous
day's converged parameters) for rolling-window re-estimation, with cold-start as
a fallback if warm-start causes convergence issues.

---

## Citation Verification

I verified the following citations against the paper:

| Spec Claim | Paper Source | Verified? |
|------------|-------------|-----------|
| Model decomposition Eqs 1-5 | Section 2, Eqs 3-5 | Yes (Eq 1 is the CMEM model, Eq 2 is the multiplicative decomposition; Eqs 3-5 are the log-space model. The spec should cite Eqs 3-5 specifically.) |
| Algorithm 1 (Kalman filter) | Section 2.2, page 4 | Yes |
| Algorithm 2 (RTS smoother) | Section 2.3.1, page 5 | Yes |
| Algorithm 3 (EM) | Section 2.3.2, page 6 | Yes |
| M-step Eqs A.32-A.39 | Appendix A.3, pages 15 | Yes, all equations match |
| Cross-covariance Eq A.20-A.21 | Appendix A.2, page 14 | Yes |
| Soft-thresholding Eqs 33-34 | Section 3.1, page 7 | Yes |
| Robust EM Eqs 35-36 | Section 3.2, page 7 | Yes |
| MAPE results Table 3 | Section 4.2, page 9 | Yes, values match |
| VWAP results Table 4 | Section 4.3, page 12 | Yes, values match |
| EM convergence Figure 4 | Section 2.3.3, page 6 | Yes |
| VWAP Eqs 39-42 | Section 4.3, pages 8-10 | Yes |
| 64% / 29% improvement claims | Section 4.2, page 8 | Yes |

**Minor citation issue:** The overview (line 7) cites "Paper: chen_feng_palomar_2016,
Section 4, Tables 3-4" but the improvement percentages (64%, 29%, 15%, 9%) are
actually stated in the abstract and Section 5 (Conclusion), page 11. Table 3 contains
the raw MAPE numbers but not the percentage improvements directly.

---

## Overall Assessment

The draft is strong and substantially implementable. The 5 major issues are:
1. **M1** (filter loop structure): clarity issue that could cause off-by-one errors.
2. **M3** (missing log-likelihood formula): blocks convergence check implementation.
3. **M4** (index mismatch between standard and robust filters): would cause confusion.
4. **M5** (dynamic VWAP logic): incorrectly implies batch multi-step re-forecasting.
5. **M2** (D set boundary): minor but explicit enumeration prevents errors.

Of these, M3 and M5 are the most critical. M3 is a true gap -- without the
log-likelihood formula, the EM convergence check cannot be implemented. M5 describes
the wrong algorithm for dynamic VWAP.

The minor issues are mostly about missing definitions (MAPE, VWAP TE), notation
clarity, and practical implementation guidance. None blocks implementation but
addressing them would reduce the chance of bugs.

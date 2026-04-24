# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model

## Summary Assessment

The draft is a high-quality, well-structured specification that covers the core
algorithms (Kalman filter, smoother, EM, robust Lasso extension) with clear pseudocode
and accurate citations. The data flow diagram is excellent, and the parameter table is
thorough. However, I identify **3 major issues** and **8 minor issues** that need
correction before a developer can implement this reliably.

---

## Major Issues

### M1. Incorrect VWAP Tracking Error for Standard Kalman Filter

**Location:** Validation > Expected Behavior, line ~666.

The spec states:

> Standard Kalman Filter (dynamic VWAP): average 4.87 bps (mean of column).

This is **wrong**. The value 4.87 is the dynamic VWAP tracking error for **AAPL
specifically** (Table 4, first data row, "Kalman Filter" column). The correct
cross-security average from the bottom row of Table 4 is **6.39 bps**.

For reference, the paper summary (`papers/chen_feng_palomar_2016.md`) correctly
reports "Standard Kalman Filter: 6.39 bps." The same error does not appear in
the robust KF (6.38 bps) or CMEM (7.01 bps) entries, which are correct.

**Fix:** Replace 4.87 with 6.39.

**Source:** Paper, Table 4, bottom row "Average", Kalman Filter dynamic columns.

---

### M2. EM M-Step Ordering: r Must Be Computed After phi

**Location:** Algorithm 3 pseudocode, lines ~293-310.

The M-step pseudocode computes the observation noise variance `r` (lines 293-305)
**before** the seasonality vector `phi` (lines 307-309). However, the paper's
closed-form update for r (Equation A.38 / Eq 23) uses `phi^{(j+1)}` -- the
**newly updated** phi from the current M-step iteration, not the old phi from
the previous iteration:

    r^{(j+1)} = (1/N) sum [ y_tau^2 + C P_tau C^T - 2 y_tau C x_hat_tau
                + (phi_tau^{(j+1)})^2 - 2 y_tau phi_tau^{(j+1)}
                + 2 phi_tau^{(j+1)} C x_hat_tau ]

The superscript `(j+1)` on phi in A.38 is explicit: the M-step derives r and phi
by jointly setting partial derivatives of Q(theta | theta_old) to zero
(Equations A.30-A.31), and the resulting closed-form for r depends on the new phi.

Since `phi^{(j+1)}` depends only on E-step quantities (Eq A.39:
`phi_i = (1/T) sum_t (y_{t,i} - C x_hat_{t,i})`), it can be computed
independently before r. The current ordering would cause the EM to use stale phi
values in the r update, producing incorrect r estimates and potentially slower or
incorrect convergence.

**Fix:** In the M-step pseudocode, move the phi update before the r update.
Explicitly note the ordering dependency.

**Source:** Paper, Appendix A.3, Equations A.38-A.39. The `(j+1)` superscript on
phi in A.38 establishes the dependency.

---

### M3. Algorithm 1 Output Does Not Include Quantities Needed by Algorithm 2

**Location:** Algorithm 1 (Kalman Filter) output specification, lines ~112-114;
Algorithm 2 (Kalman Smoother) input specification, lines ~177-181.

Algorithm 2 requires as input:

- `x_pred[1..N]` -- predicted state means **before** correction
- `Sigma_pred[1..N]` -- predicted state covariances **before** correction
- `A[1..N]` -- transition matrices at each step

But Algorithm 1's output specification only lists:

- `y_hat[1..N]` -- forecasts
- `x_hat[1..N]` -- filtered (post-correction) state estimates
- `Sigma[1..N]` -- filtered (post-correction) covariances

The predicted quantities `x_pred` and `Sigma_pred` are computed inside the loop
(lines 130-131) but are overwritten each iteration and never stored or returned.
Similarly, the transition matrices `A[1..N]` are constructed inside the loop but
not returned. A developer following the output spec would not store these, and
then Algorithm 2 would fail for lack of inputs.

**Fix:** Add `x_pred[1..N]`, `Sigma_pred[1..N]`, `A[1..N]`, and `K[1..N]` (the
Kalman gain, needed for the smoother cross-covariance initialization at A.21) to
Algorithm 1's output specification. Note that these must be stored during the
forward pass for use by the backward smoother pass.

**Source:** Paper, Algorithm 2 input requirements; Appendix A, Equation A.21
(requires K_N).

---

## Minor Issues

### m1. Missing Log-Likelihood Formula for EM Convergence

**Location:** Algorithm 3 pseudocode, convergence check at lines ~313-315.

The pseudocode says "Compute log-likelihood or monitor parameter changes" and
uses `|theta_new - theta_old| < tol` as the convergence criterion. However:

1. The paper's EM derivation is based on maximizing the expected log-likelihood
   Q(theta | theta_old). The natural convergence criterion is the change in the
   observed-data log-likelihood (or Q itself) between iterations.
2. The "parameter change" criterion is not well-defined for a parameter vector
   containing quantities of different scales (a^eta near 1, r potentially much
   smaller, phi a vector of I elements).
3. The paper provides the full log-likelihood in Equation A.8, which can be
   evaluated from the Kalman filter's innovation sequence: log L = -0.5 * sum_tau
   [log(2 pi S_tau) + e_tau^2 / S_tau], where S_tau is the innovation variance
   and e_tau is the innovation.

**Fix:** Add the innovation-form log-likelihood formula and use relative change
in log-likelihood as the convergence criterion:
`|log L^{(j+1)} - log L^{(j)}| / |log L^{(j)}| < tol`.

**Source:** Paper, Appendix A.1, Equation A.8; standard Kalman filter
log-likelihood (innovation form).

---

### m2. Robust EM: Smoother/Lasso Interaction Not Fully Specified

**Location:** Robust EM Modifications, lines ~413-431.

The spec describes the modified M-step updates for r and phi (Eqs 35-36) that
use `z_star` values. However, the E-step procedure for the robust model is not
specified. Key ambiguity:

- The standard E-step runs the Kalman filter (forward) then the RTS smoother
  (backward).
- The robust Kalman filter (Algorithm 4) modifies the correction step with Lasso.
- Does the E-step for the robust model use Algorithm 4 (robust filter) in the
  forward pass, or the standard Algorithm 1?
- The RTS smoother (Algorithm 2) has no Lasso modification. Can it be applied
  directly to the outputs of the robust filter, or does using "cleaned"
  innovations in the forward pass already account for outliers?

The paper's Section 3.2 states that z_1*...z_N* from Problem (30) are used in the
M-step. This implies the forward pass uses the robust filter to obtain z* values,
and the smoother operates on the "cleaned" filtered states. But this should be
stated explicitly.

**Fix:** Add a paragraph specifying: (1) the E-step forward pass uses Algorithm 4
(robust filter) to produce both filtered states and z* values; (2) the RTS
smoother (Algorithm 2) is applied to the filtered outputs of Algorithm 4 without
modification; (3) z* values from the forward pass are stored and used in the
M-step for r and phi updates.

**Source:** Paper, Section 3.2, paragraph before Eq 35.

---

### m3. Dynamic VWAP: "Static Predictions" for Remaining Bins Is Imprecise

**Location:** VWAP Execution Strategies, lines ~448-459.

The spec states: "The remaining volume forecasts for bins i+1..I in the
denominator use static predictions (conditioned on information up to bin i-1)."

This is potentially misleading. The term "static predictions" elsewhere in the
spec refers to predictions made before market open (using only prior-day
information). Here, the remaining-bin forecasts for the dynamic VWAP denominator
are multi-step-ahead predictions from the **current** filtered state (after
observing bins 1 through i-1), not the pre-market static predictions.

A developer could reasonably interpret "static predictions" as the pre-computed
static forecasts from before market open, which would give different (worse)
results.

**Fix:** Replace "use static predictions (conditioned on information up to bin
i-1)" with "use multi-step-ahead predictions from the current filtered state
(propagating the predict step without corrections for bins i+1 through I)." This
makes clear that these are forward projections from the latest posterior, not the
pre-market forecasts.

**Source:** Paper, Eq 41 and surrounding text on page 10.

---

### m4. Bias Correction for Volume-Space Forecasts Not Addressed

**Location:** Data Flow, line ~507; Validation > Known Limitations, item 4.

The spec converts log-volume forecasts to volume via `volume_hat = exp(y_hat)`.
However, for a Gaussian random variable Y ~ N(mu, sigma^2), E[exp(Y)] =
exp(mu + sigma^2/2), not exp(mu). This means exp(y_hat) is a **biased**
estimator of expected volume -- it systematically underestimates the conditional
mean volume.

The MAPE metric (Eq 37 in the paper) compares predicted volume to actual volume.
If the paper's implementation uses exp(y_hat) without bias correction and still
achieves 0.46 MAPE, then the developer should do the same for reproducibility.
But this should be explicitly noted as a design choice, and the bias correction
formula should be provided as an option:

    volume_hat_corrected = exp(y_hat + S_tau / 2)

where S_tau is the innovation variance (or the predictive variance of y_hat).

The paper does not mention this correction. For VWAP weights (which are ratios),
the bias cancels if all bins have similar predictive variance, so it matters less
there. But for absolute volume prediction and MAPE evaluation, it could matter.

**Fix:** Add a note in the Data Flow section about the log-normal bias, provide
the correction formula, and recommend using exp(y_hat) without correction for
MAPE benchmarking (to match the paper) but noting the corrected form for
applications requiring unbiased volume estimates.

**Source:** Standard log-normal distribution property (Researcher inference;
not discussed in the paper).

---

### m5. Smoother Cross-Covariance Computation Is Unclear

**Location:** Algorithm 2, lines ~206-213.

The cross-covariance `P_cross[tau]` computation is described in comments rather
than in the main pseudocode body. The initialization at tau=N and the backward
recursion for `Sigma_{tau,tau-1|N}` are presented as side notes, making them easy
to miss or misimplement. Specifically:

- The initialization `Sigma_{N,N-1|N} = (I - K_N @ C) @ A_N @ Sigma[N-1]`
  requires K_N (the Kalman gain at the last step), which is not listed in
  Algorithm 2's inputs.
- The backward recursion for `Sigma_{tau,tau-1|N}` is nested inside a comment
  block rather than being a clearly delineated loop.
- The relationship between `Sigma_cross`, `Sigma_{tau,tau-1|N}`, and `P_cross`
  should be spelled out: `P_cross[tau] = Sigma_{tau,tau-1|N} + x_smooth[tau] @
  x_smooth[tau-1]^T`.

**Fix:** Promote the cross-covariance recursion to a full pseudocode section
within Algorithm 2, with explicit initialization and loop structure. Add K_N to
Algorithm 2's inputs (or note it must be stored from Algorithm 1).

**Source:** Paper, Appendix A, Equations A.20-A.22.

---

### m6. EM Convergence Speed Claim Is Vague

**Location:** Calibration, line ~638-641.

The spec says: "The paper shows convergence within 'a few iterations' (Paper,
Section 2.3.3). In practice, 20-50 iterations with tolerance 1e-6 on relative
log-likelihood change should suffice."

The paper's Figure 4 shows convergence from synthetic data (known true
parameters). On real data, convergence speed may differ. The recommendation of
"20-50 iterations" has no paper backing and is labeled neither as paper evidence
nor as Researcher inference.

**Fix:** Mark "20-50 iterations" explicitly as Researcher inference and add a
note that the developer should monitor actual convergence on real data rather
than relying on a fixed iteration budget.

---

### m7. Parameter Stationarity Constraint Not Enforced in EM

**Location:** Parameters table, a^eta and a^mu rows.

The spec notes that a^eta and a^mu should be in (0, 1) for stationarity, but the
EM M-step updates (Eqs 19-20 / A.34-A.35) are unconstrained ratios of sufficient
statistics. There is no guarantee that the EM-estimated a^eta or a^mu will fall
in (0, 1). The paper does not discuss this, but in practice:

- a^eta is often very close to 1 for persistent daily volume levels. The EM could
  return a^eta > 1, making the daily component nonstationary.
- a^mu should be moderate, but could be negative if intraday dynamics are
  anti-persistent.

**Fix:** Add a note recommending post-hoc clamping of a^eta and a^mu to (0, 1)
after each M-step, with a warning if the unclamped values fall outside this
range (as it may indicate model misspecification or insufficient data).

**Source:** Researcher inference; the paper does not address constraint enforcement
in the EM.

---

### m8. MAPE Definition Missing from Spec

**Location:** Validation > Expected Behavior.

The spec cites MAPE values extensively but never defines MAPE. The paper defines
it in Equation 37:

    MAPE = (1/M) sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of out-of-sample bins. A developer needs this formula
to reproduce the benchmark numbers. Notably, the MAPE is computed in **volume
space** (not log-volume space), so the exp() conversion is part of the evaluation
pipeline.

**Fix:** Add the MAPE formula (Eq 37) to the Validation section, explicitly
noting that it operates on volume (not log-volume) and that predicted volume is
obtained via exp(y_hat).

**Source:** Paper, Section 3.3, Equation 37.

---

## Positive Observations

The following aspects of the spec are well done and should be preserved:

1. **State-space formulation** is clear and accurate, correctly handling the
   time-varying A_tau and Q_tau at day boundaries.
2. **Day boundary detection** using `tau mod I == 1` is correct and well-explained.
3. **Soft-thresholding derivation** for the robust filter matches the paper's
   Equations 33-34 precisely.
4. **Paper references table** is comprehensive and accurate for the sections I
   verified.
5. **Edge cases section** is thoughtful, particularly the zero-volume handling
   recommendation and the numerical precision notes.
6. **Researcher inference labels** are consistently applied where the spec goes
   beyond the paper.
7. **VWAP strategies** are clearly described with correct formulas.

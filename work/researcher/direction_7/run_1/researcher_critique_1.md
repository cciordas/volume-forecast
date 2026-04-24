# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model

## Summary

The draft is a well-structured, detailed specification that closely follows Chen et
al. (2016). The pseudocode is generally clear and the paper references are precise.
However, I found 3 major issues (algorithmic errors or omissions that would produce
incorrect results), 4 medium issues (ambiguities or gaps that could lead to
misimplementation), and 5 minor issues (clarity improvements). All are detailed below
with paper evidence.

---

## Major Issues

### M1. Dynamic VWAP formula is incorrectly simplified

**Spec section:** Algorithm, VWAP Execution Strategies, DYNAMIC_VWAP (lines 424-438)

**Problem:** The spec's dynamic VWAP formula uses a fixed denominator
(`sum_{j=i_current+1}^{I} volume_forecast_dynamic[j]`) and a fixed multiplier
(`1 - cum_allocated`) for all remaining bins. The paper's Equation 41 is recursive:
each bin's weight depends on the cumulative sum of ALL previous weights (including
dynamically computed ones), and the denominator starts at the CURRENT bin, not a
fixed starting point.

**Paper evidence (Section 4.3, Equation 41):**

```
w_{t,i}^{(d)} = {
  (vol_hat_{t,i}^{(d)} / sum_{j=i}^{I} vol_hat_{t,j}^{(d)}) * (1 - sum_{j=1}^{i-1} w_{t,j}^{(d)}),   i = 1,...,I-1
  1 - sum_{j=1}^{I-1} w_{t,j}^{(d)},                                                                    i = I
}
```

Key differences from the spec:
1. The denominator is `sum_{j=i}^{I}` (from CURRENT bin i to I), not from a fixed
   i_current+1 to I.
2. The multiplier `1 - sum_{j=1}^{i-1} w_{t,j}^{(d)}` updates at each bin as previous
   weights are computed. The spec uses a fixed `cum_allocated` that does not update
   within the loop.
3. The paper's formula computes weights for ALL bins i = 1,...,I in a single forward
   pass. The spec introduces an `i_current` concept that confuses the actual weight
   computation with the dynamic revision strategy.

**Correct pseudocode:**

```
DYNAMIC_VWAP(volume_forecast_dynamic[1..I]):
    for i = 1 to I-1:
        remaining_vol = sum_{j=i}^{I} volume_forecast_dynamic[j]
        cumulative_w = sum_{j=1}^{i-1} w[j]
        w[i] = (volume_forecast_dynamic[i] / remaining_vol) * (1 - cumulative_w)
    w[I] = 1 - sum_{j=1}^{I-1} w[j]
    return w[1..I]
```

At each intraday revision point (after observing bin k), the volume forecasts
`volume_forecast_dynamic[k+1..I]` are updated using the latest Kalman filter
predictions, and the formula above is reapplied to compute revised weights for the
remaining bins.

**Impact:** A developer implementing the spec's formula would produce incorrect VWAP
weights, leading to wrong tracking error results and inability to reproduce Table 4.


### M2. EM M-step parameter ordering: r depends on the UPDATED phi

**Spec section:** Algorithm, Algorithm 3 (EM), M-step (lines 256-264)

**Problem:** The spec's pseudocode lists the r update (line 256) BEFORE the phi
update (line 262). However, the r update formula explicitly uses `phi^{(j+1)}` (the
NEW phi from the current M-step iteration), not the old phi. A developer reading the
pseudocode sequentially would compute r using the OLD phi, producing incorrect r
estimates.

**Paper evidence (Appendix A, Equations A.38-A.39, and Algorithm 3 lines 16-17):**

Equation A.38 explicitly writes `phi_tau^{(j+1)}` (the j+1 superscript denoting the
current iteration's update). Equation A.39 computes phi^{(j+1)} independently of r.
These form a system where phi must be computed first:

1. Compute phi^{(j+1)} from A.39 (depends only on smoothed states and observations).
2. Compute r^{(j+1)} from A.38 (depends on phi^{(j+1)} from step 1).

All other M-step updates (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, pi_1, Sigma_1)
depend only on smoothed sufficient statistics and can be computed in any order.

**Fix:** Reorder the pseudocode so phi is computed before r, and add an explicit note
that r depends on the updated phi.

**Impact:** Using the old phi in the r update would slow EM convergence and could
lead to slightly different parameter estimates. Not catastrophic (it's effectively an
ECM variant), but unfaithful to the paper's derivation.


### M3. Missing log-likelihood formula for convergence check

**Spec section:** Algorithm, Algorithm 3 (EM), convergence check (line 266)

**Problem:** The spec says `delta = |log_likelihood(theta_new) - log_likelihood(theta_old)|`
but never defines the log-likelihood function. A developer cannot implement the
convergence check without this formula.

**Paper evidence (Appendix A, Equation A.8):**

The joint log-likelihood is:

```
log P({x_tau}, {y_tau})
  = - sum_{tau=1}^{N} (y_tau - phi_tau - C x_tau)^2 / (2r) - (N/2) log(r)
    - sum_{tau=2}^{N} (mu_tau - a_mu * mu_{tau-1})^2 / (2 sigma_mu^2) - ((N-1)/2) log(sigma_mu^2)
    - sum_{tau=kI+1} (eta_tau - a_eta * eta_{tau-1})^2 / (2 sigma_eta^2) - ((T-1)/2) log(sigma_eta^2)
    - (1/2)(x_1 - pi_1)^T Sigma_1^{-1} (x_1 - pi_1) - (1/2) log|Sigma_1|
    - ((2N+T)/2) log(2 pi)
```

However, the EM actually monitors the expected complete-data log-likelihood Q
(Equation A.10), not the observed-data log-likelihood. The spec should either:
(a) Provide the Q function formula (A.10) and monitor its value, or
(b) Provide the observed-data log-likelihood via the innovation form:
    `log L = -(N/2) log(2 pi) - (1/2) sum_{tau=1}^{N} [log(S_tau) + e_tau^2 / S_tau]`
    where S_tau is the innovation variance and e_tau is the innovation from the
    Kalman filter. This is more practical.

**Impact:** Without the formula, a developer cannot implement the convergence check,
which is listed as a sanity check (edge case 5, line 716).


---

## Medium Issues

### N1. Variable naming confusion: W_inv is actually S

**Spec section:** Algorithm, Algorithm 1 (lines 99-102)

The variable `W_inv[tau]` is computed as the innovation variance S (= C Sigma C^T + r)
but named `W_inv`, which suggests it's the INVERSE of something. The comment on lines
101-102 tries to clarify: "W_inv is actually S (innovation variance), not W." This is
confusing. In Algorithm 4 (robust filter), the variable W is correctly used as S^{-1}
(line 311).

**Fix:** Rename `W_inv[tau]` to `S[tau]` in Algorithm 1, consistent with Algorithm 4
and standard Kalman filter notation. Alternatively, use a single naming convention
throughout (S for innovation variance everywhere).


### N2. Robust EM E-step: smoother interaction with sparse outliers z* is underspecified

**Spec section:** Algorithm, Robust EM Modifications (lines 341-361)

The spec says the robust EM only modifies r and phi in the M-step. But during the
E-step, the Kalman filter produces different filtered estimates because the correction
step uses `e_corrected = e - z*` instead of `e`. The smoother then runs on these
modified filtered estimates. The spec does not explicitly state:

1. That the robust Kalman filter (not the standard one) should be used in the E-step.
2. That the smoother algorithm itself is unchanged (it uses the filtered estimates
   from the robust filter, but the backward recursion formulas are identical).
3. That z* values from the robust filter E-step are stored and reused in the M-step.

**Paper evidence (Section 3.2, paragraph 1):** "Let z*_1...z*_N denote the solutions
of problem (30) calculated in the E-step, the estimations of parameter r and phi_i in
the M-step in Algorithm 3 are replaced with..."

**Fix:** Add an explicit note in the Robust EM section stating that the E-step uses
the robust Kalman filter, the smoother is unchanged but operates on robust-filtered
estimates, and z* values are saved from the E-step for use in the M-step.


### N3. Multi-step prediction has an unresolved placeholder

**Spec section:** Algorithm, Prediction Modes, MULTI_STEP_PREDICTION (line 405)

The pseudocode contains `A_step = <appropriate A_tau for this step>`, which is not
concrete enough for implementation. The developer needs to know exactly how to
determine the transition matrix at each step of the h-step prediction.

**Fix:** Replace the placeholder with explicit logic:

```
# Determine bin index and whether this step crosses a day boundary
current_bin = ((tau_start + step - 1) mod I)
if current_bin == 0:  # day boundary
    A_step = [[a_eta, 0], [0, a_mu]]
else:
    A_step = [[1, 0], [0, a_mu]]
```


### N4. MAPE formula not defined

**Spec section:** Validation (lines 615-637)

The spec cites MAPE values from Table 3 but never defines the MAPE formula. A developer
needs this to verify their implementation produces matching results.

**Paper evidence (Section 3.3, Equation 37):**

```
MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau
```

where M is the total number of out-of-sample bins, volume_tau is the actual volume
(in shares, not log-space), and predicted_volume_tau = exp(y_hat_tau) * shares_outstanding.

**Fix:** Add the MAPE definition to the Validation section.


---

## Minor Issues

### P1. Covariance update form is inconsistent with the recommendation

**Spec section:** Algorithm, Algorithm 1 (line 109) vs. Edge Cases (line 711-714)

Line 109 uses `Sigma[tau|tau] = Sigma[tau|tau-1] - K[tau] * W_inv[tau] * K[tau]^T`,
which is the K*S*K^T form. Line 711-714 then recommends the Joseph form for numerical
stability. The pseudocode should use the Joseph form directly (or at minimum use the
paper's form K*C*Sigma from Algorithm 1 line 6, which is slightly more stable than
K*S*K^T), rather than presenting a less stable form and then saying "consider using
something else."


### P2. Log-normal bias correction not mentioned

**Spec section:** Algorithm, Prediction Modes (line 374) and Data Flow, Post-processing
(line 472)

The spec converts log-volume to volume via `exp(y_hat) * shares_outstanding`. For a
log-normal model, E[V] = exp(mu + sigma^2/2), not exp(mu). The point prediction
exp(y_hat) is the conditional median, not the conditional mean. The innovation variance
S_tau from the Kalman filter provides the sigma^2 needed for bias correction:
`E[V_tau | F_{tau-1}] = exp(y_hat_tau + S_tau/2) * shares_outstanding`.

The paper does not discuss this, so for exact reproducibility of Table 3 MAPE values,
exp(y_hat) should be used as-is. But a note should be added warning that this is the
median prediction and that bias correction may improve real-world performance.


### P3. Missing VWAP tracking error formula

**Spec section:** Validation, Expected Behavior (lines 626-636)

The spec cites VWAP tracking error results from Table 4 but does not define the
tracking error metric. The paper defines it in Equation 42:

```
VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t
```

where D is the number of out-of-sample days. This should be included alongside the
MAPE definition for validation.


### P4. Robust filter threshold: lambda vs. lambda/2

**Spec section:** Algorithm, Algorithm 4 (line 315)

The spec writes `threshold = lambda / (2.0 * W[tau])`. Looking at the paper's Equation
33, the soft-thresholding solution uses `lambda / (2 * W_{tau+1})` where
`W_{tau+1} = (C Sigma_{tau+1|tau} C^T + r)^{-1}`. The spec correctly computes
W = 1/S, so `lambda / (2 * W)` = `lambda * S / 2`. This is correct, but the spec
should clarify the threshold is in terms of S: `threshold = lambda * S[tau] / 2`,
which is more transparent and avoids confusion about W vs S.


### P5. Rolling calibration warm-start not discussed

**Spec section:** Parameters, Calibration (lines 567-599)

The calibration procedure runs EM from `theta_init` for each out-of-sample day (line
594). In a rolling-window scheme, a natural optimization is to initialize the EM with
the parameter estimates from the previous day's calibration (warm-starting). This
would dramatically reduce the number of EM iterations needed per day. The spec should
note this as a practical recommendation.


---

## Verification of Specific Citations

I verified the following citations against the paper:

| Spec Claim | Paper Source | Verdict |
|---|---|---|
| C = [1, 1] | Section 2, page 3 | Correct |
| A_tau time-varying structure (a_eta at boundaries, 1 within day) | Section 2, page 3, bullet points | Correct |
| EM M-step a_eta (A.34) | Appendix A, Eq A.34 | Correct: sums over tau = kI+1 |
| EM M-step a_mu (A.35) | Appendix A, Eq A.35 | Correct: sums over tau = 2..N |
| EM M-step sigma_eta_sq divisor T-1 (A.36) | Appendix A, Eq A.36 | Correct |
| EM M-step sigma_mu_sq divisor N-1 (A.37) | Appendix A, Eq A.37 | Correct |
| EM M-step r (A.38) | Appendix A, Eq A.38 | Correct formula, but uses phi^{(j+1)} |
| EM M-step phi (A.39) | Appendix A, Eq A.39 / Eq 24 | Correct |
| pi_1, Sigma_1 (A.32-A.33) | Appendix A, Eqs A.32-A.33 | Correct |
| Cross-covariance initialization (A.21) | Appendix A, Eq A.21 | Correct |
| Cross-covariance recursion (A.20) | Appendix A, Eq A.20 | Correct |
| Soft-thresholding (Eq 33-34) | Section 3.1, Eqs 33-34 | Correct |
| Robust r update (Eq 35) | Section 3.2, Eq 35 | Correct |
| Robust phi update (Eq 36) | Section 3.2, Eq 36 | Correct |
| Dynamic MAPE: Robust KF 0.46, KF 0.47 | Table 3, Average row | Correct |
| Static MAPE: Robust KF 0.61, KF 0.62 | Table 3, Average row | Correct |
| Dynamic VWAP TE: Robust KF 6.38 bps | Table 4, Average row | Correct |
| CMEM failure under outliers | Table 1 (N/A entries) | Correct |
| Static VWAP weights (Eq 40) | Section 4.3, Eq 40 | Correct |
| Dynamic VWAP weights (Eq 41) | Section 4.3, Eq 41 | INCORRECT (see M1) |
| EM convergence robustness | Section 2.3.3, Figure 4 | Correct claim |

## Overall Assessment

The spec is solid overall. The core algorithms (Kalman filter, smoother, EM with
closed-form M-step) are correctly transcribed. The major issues are:

1. The dynamic VWAP formula (M1) is algorithmically wrong and must be fixed.
2. The r/phi ordering (M2) is a subtle but important dependency.
3. The missing log-likelihood formula (M3) leaves a gap in implementability.

Fixing these three plus the medium issues would produce a specification ready for
implementation.

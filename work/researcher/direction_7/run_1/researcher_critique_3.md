# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model

## Summary

Draft 3 resolves all 5 issues from critique 2. The static VWAP tracking error
reference values now correctly read Table 4 (6.85/6.89/7.71/7.48 for
RKF/KF/CMEM/RM), the filter return signatures are consistent with their
callers, relative convergence tolerance is used, the compact r formula is
primary, and the analytical 2x2 inverse note is included.

I performed a full re-verification of all core algorithms against the paper PDF
and found only 2 minor issues. The specification is implementation-ready.

---

## Minor Issues

### P1. Robust filter omits y_hat from return, inconsistent with standard filter

**Spec section:** Algorithm, Algorithm 4 (line 426) vs Algorithm 1 (line 137);
also Calibration procedure (line 789)

**Problem:** The standard Kalman filter returns y_hat as its last value (line 137):

```
return x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, K, S, e, y_hat
```

The robust Kalman filter returns z_star in that position (line 426):

```
return x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, K, S, e, z_star
```

The robust filter needs z_star for the EM M-step, which is correct. But the
calibration procedure (line 789) calls the robust filter for out-of-sample
prediction:

```
forecasts[d] = ROBUST_KALMAN_FILTER(y_day_d, theta, best_lambda, I)
```

The caller needs y_hat for prediction and MAPE evaluation. Since the robust
filter doesn't return y_hat, the developer must compute it separately from
x_hat_pred and phi, which is undocumented. Critique 2 (N2) suggested returning
both z_star and y_hat; the proposer added z_star but dropped y_hat.

**Fix:** Add y_hat to the robust filter return:

```
return x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, K, S, e, z_star, y_hat
```

And update the robust EM call (line 457) to ignore the extra return:

```
x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, K, S, e, z_star, _ = ROBUST_KALMAN_FILTER(...)
```

This keeps the two filter interfaces parallel (both return y_hat) while
preserving z_star for the EM.


### P2. Robust EM log-likelihood is a heuristic, not the exact observed-data log-likelihood

**Spec section:** Algorithm, Robust EM (lines 459-461)

**Problem:** The robust EM computes the log-likelihood using corrected
innovations:

```
e_corrected[tau] = e[tau] - z_star[tau]
log_lik = -(N/2)*log(2*pi) - (1/2)*sum [log(S[tau]) + e_corrected[tau]^2/S[tau]]
```

For the standard filter, the innovation e[tau] is exactly the prediction error
and the innovation-form log-likelihood is the exact observed-data
log-likelihood. For the robust filter, z_star is a deterministic function of
e[tau] (via soft-thresholding), so e_corrected has a truncated distribution
(it is zero whenever |e| < threshold, and clipped otherwise). The formula above
is therefore not the exact observed-data log-likelihood for the robust model --
it is a practical heuristic for convergence monitoring.

This does not affect correctness: the EM is guaranteed to converge because the
E-step and M-step are well-defined (the paper's derivation holds). The
log-likelihood is only used for the convergence check, and a non-decreasing
heuristic is sufficient for that purpose.

**Fix:** Add a brief note after the robust EM log-likelihood computation:

```
# Note: this is a practical convergence monitor, not the exact observed-data
# log-likelihood for the robust model (z* is a function of e, so e_corrected
# does not follow the standard Gaussian innovation distribution). The EM
# convergence is guaranteed by the E-step/M-step structure regardless.
```

The paper does not discuss log-likelihood computation for the robust EM
(Researcher inference applies here as well).

---

## Verification of Critique 2 Resolutions

All 5 issues from critique 2 were addressed:

| Issue | Status | Notes |
|-------|--------|-------|
| N1: Static VWAP TE values | Fixed | Lines 832-841: 6.85/6.89/7.71/7.48 with cross-check against AAPL/FB individual values |
| N2: Filter return signature | Fixed | Lines 137, 221, 426, 457 now consistent (filter returns filt+pred+A+K+S+e+extra) |
| P1: Convergence tolerance | Fixed | Lines 302-316: relative tolerance with max(..., 1.0) guard and explanatory note |
| P2: Compact r formula | Fixed | Lines 294-301: compact form primary, expanded as verification comment |
| P3: Analytical 2x2 inverse | Fixed | Lines 194-207: note with formula and attribution as Researcher inference |

## Additional Citation Verifications (New in This Round)

| Spec Claim | Paper Source | Verdict |
|---|---|---|
| a_eta sum over D_plus = {I+1, 2I+1, ..., (T-1)I+1} | Appendix A, Eq A.34, sum over tau=kI+1 | Correct |
| sigma_eta_sq uses NEW a_eta (sequential M-step) | Appendix A, Eq A.36, uses a_eta^{(j+1)} explicitly | Correct |
| sigma_mu_sq uses NEW a_mu | Appendix A, Eq A.37, uses a_mu^{(j+1)} explicitly | Correct |
| sigma_eta_sq divisor T-1 | Appendix A, Eq A.36, context |D|=T-1 from Eq A.28 | Correct |
| sigma_mu_sq divisor N-1 | Appendix A, Eq A.37 | Correct |
| Smoother cross-covariance initialization | Appendix A, Eq A.21 | Correct |
| Smoother cross-covariance recursion | Appendix A, Eq A.20 | Correct |
| P_{tau,tau-1} = Sigma_cross + x_s x_s^T (outer product) | Appendix A, Eq A.22 | Correct |
| Static VWAP TE, KF = 6.89 | Table 4 Average, cross-checked: AAPL gap RKF-KF = 0.02 bps | Correct |
| Dynamic MAPE, CMEM = 0.65 | Table 3 Average row, dynamic section | Correct |
| Dynamic MAPE, RM = 1.28 | Table 3 Average row, static section (RM is static only) | Correct (paper makes same cross-comparison on p.8) |
| 15% improvement: (7.48-6.38)/7.48 = 14.7% | Paper p.11: "improvement of 15%" | Correct |
| 9% improvement: (7.01-6.38)/7.01 = 9.0% | Paper p.11: "9% when compared with the dynamic CMEM" | Correct |
| Robust filter covariance update uses Joseph form | Paper Eq 32 uses simple form; Joseph form is Researcher inference | Correctly marked |
| Relative tolerance with max guard | Researcher inference | Correctly marked |

## Overall Assessment

Draft 3 is implementation-ready. All algorithmic steps are correctly specified
with precise paper citations. The core algorithms (Kalman filter, smoother, EM,
robust extension, VWAP strategies) have been verified against the paper across
three rounds of review. The two remaining issues are cosmetic: a missing y_hat
in the robust filter return (P1) and an unlabeled heuristic in the robust EM
convergence check (P2). Neither would block a competent developer.

Recommendation: fix P1 and P2 if another round is planned, or proceed to
implementation as-is -- both issues are resolvable during development without
ambiguity.

# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model

## Summary

Draft 2 is a significant improvement over draft 1. All 3 major issues, 4 medium issues,
and 5 minor issues from critique 1 have been addressed. The dynamic VWAP formula is now
correct, the EM M-step ordering is explicit with clear commentary, the log-likelihood
formula is provided, variable naming is consistent, and the robust EM interaction is
clearly specified.

However, I found 1 medium issue (incorrect Table 4 reference values for static VWAP
tracking error), 1 medium issue (Kalman filter return signature is inconsistent with
how the EM calls it), and 3 minor issues. These are detailed below.

---

## Medium Issues

### N1. Static VWAP tracking error reference values are incorrect (column misread from Table 4)

**Spec section:** Validation, Expected Behavior, lines 793-797

**Problem:** The spec claims:

```
- Standard Kalman Filter: 8.98 bps.
- CMEM: 8.97 bps. RM: 7.48 bps.
```

These values appear to be column-reading errors from Table 4:

- **8.98** is the *standard deviation* of the Robust Kalman Filter's static VWAP TE
  (column 2 of the static section), not the *mean* of the Standard Kalman Filter
  (column 3).
- **8.97** is the *standard deviation* of the Standard Kalman Filter's *dynamic* VWAP TE
  (column 4 of the dynamic section), not the *mean* of the CMEM static VWAP TE.

**Paper evidence (Table 4, Average row):**

Cross-checking with individual securities confirms the correct column alignment:

For AAPL (static VWAP TE): Robust KF = 4.99, KF = 4.97, CMEM = 5.88, RM = 5.84.
The gap between Robust KF and Standard KF is 0.02 bps.

For FB (static VWAP TE): Robust KF = 6.16, KF = 6.17, CMEM = 7.06, RM = 6.96.
The gap between Robust KF and Standard KF is 0.01 bps.

The spec's claim of Robust KF = 6.85 vs Standard KF = 8.98 implies a 2.13 bps average
gap, which is completely inconsistent with the individual securities showing gaps of
0.01-0.02 bps. The correct Average row values for static VWAP TE are:

| Model | Mean (bps) |
|-------|-----------|
| Robust KF | 6.85 |
| Standard KF | ~6.89 |
| CMEM | ~7.71 |
| RM | 7.48 |

Additional cross-check: the paper text (page 11) states "an improvement of 15% when
compared with the RM" for dynamic VWAP. Using dynamic RKF = 6.38 and static RM = 7.48:
(7.48 - 6.38) / 7.48 = 14.7% ≈ 15%. This confirms RM = 7.48 for static is correct,
and this number appears in the correct column in the spec's static section.

**Fix:** Replace the static VWAP TE values with the correctly-read Table 4 averages.
Also note that the RM value cited in the dynamic VWAP section (line 791) comes from
Table 4's static columns, since RM is inherently a static strategy. Add a note like
"(RM is a static method; this cross-comparison is taken from the static column of
Table 4)."

**Impact:** A developer comparing their implementation's VWAP tracking error against
these reference values would incorrectly conclude their Standard KF static implementation
is much better than the paper's (6.89 vs 8.98), or their CMEM baseline is too good.


### N2. Kalman filter return signature is inconsistent with EM caller

**Spec section:** Algorithm, Algorithm 1 (line 131) vs Algorithm 3 (line 200); also
Algorithm 4 (line 392) vs Robust EM (line 431)

**Problem:** The EM algorithm (line 200) calls:

```
x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, K, S, e = KALMAN_FILTER(y, theta, I)
```

This expects 7 return values, including separate arrays for filtered (x_hat[tau|tau],
Sigma[tau|tau]) and predicted (x_hat[tau|tau-1], Sigma[tau|tau-1]) estimates.

But the Kalman filter's return statement (line 131) is:

```
return x_hat, Sigma, K, S, e, y_hat
```

This returns 6 values with ambiguous `x_hat` and `Sigma` (are these the filtered or
predicted arrays?). The developer would need to guess that both filtered AND predicted
arrays must be stored and returned separately.

The same issue exists for Algorithm 4 (Robust Kalman Filter):
- Returns: `x_hat, Sigma, K, S, e, z_star` (line 392)
- But Robust EM (line 431) passes `x_hat_pred, Sigma_pred` to the smoother, which
  are not in the return.

Additionally, the smoother (Algorithm 2, line 163) requires `A[tau]` for each time step,
and the EM (line 208) passes `A` to the smoother. But neither Kalman filter algorithm
returns the A array. A developer would need to either reconstruct A from parameters
and I, or store it during the filter pass.

**Fix:** Update both Kalman filter return statements to explicitly return all arrays
needed by their callers:

```
return x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, K, S, e, y_hat
```

And for the robust variant:

```
return x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, K, S, e, z_star, y_hat
```

Also add a comment in the filter loop body clarifying that both filtered and predicted
arrays must be stored at each step:

```
# Store for smoother and EM:
# x_hat_pred[tau] = x_hat[tau|tau-1]  (before correction)
# Sigma_pred[tau] = Sigma[tau|tau-1]  (before correction)
# x_hat_filt[tau] = x_hat[tau|tau]    (after correction)
# Sigma_filt[tau] = Sigma[tau|tau]    (after correction)
```

**Impact:** A developer reading the filter pseudocode would implement it with a single
state array (overwriting predicted with filtered at each step), then find that the EM
code expects both. The inconsistency would require backtracking to redesign the data
structures.


---

## Minor Issues

### P1. Convergence tolerance: absolute vs relative not discussed

**Spec section:** Algorithm, Algorithm 3 (line 281)

The convergence check uses `delta = |log_lik - log_lik_prev|` with a tolerance of 1e-6
(from the calibration procedure, line 732). This is an absolute tolerance on the
log-likelihood difference.

The log-likelihood scale depends on N (more data points = larger absolute log-likelihood
values). For N = 500 days * 26 bins = 13,000 bins, the absolute log-likelihood could
be O(10,000), making a change of 1e-6 negligibly small (converges very quickly, possibly
prematurely). For short windows (N = 50 * 26 = 1,300), the same tolerance could be
relatively larger.

**Fix:** Either (a) use a relative tolerance `|log_lik - log_lik_prev| / |log_lik_prev|`
instead, or (b) add a note explaining that the tolerance may need scaling with N, and
suggest a practical value like `tol = 1e-6 * N` for absolute tolerance, or `tol = 1e-8`
for relative tolerance.


### P2. Expanded r formula is error-prone; recommend the compact form

**Spec section:** Algorithm, Algorithm 3 (lines 272-278) and Robust EM (lines 447-454)

The r update formula is presented in fully expanded form with 6+ additive terms (line
272-274), then a compact equivalent is given in comments (lines 276-277):

```
# Equivalently: r = (1/N) * sum_{tau=1}^{N}
#   [(y[tau] - phi[i_tau])^2 - 2*(y[tau] - phi[i_tau])*C @ x_hat_s[tau] + C @ P[tau] @ C^T]
```

The expanded form is directly from Eq A.38 and is useful for paper verification, but
the compact form is what a developer should actually implement (fewer terms = fewer
transcription errors). The same issue applies to the robust r in Eq 35.

**Fix:** Present the compact form as the primary pseudocode and relegate the expanded
form to a verification comment. For the robust variant, the compact form is:

```
r = (1/N) * sum [(y[tau] - phi[i_tau] - z_star[tau])^2
                 - 2*(y[tau] - phi[i_tau] - z_star[tau]) * C @ x_hat_s[tau]
                 + C @ P[tau] @ C^T]
```


### P3. Smoother gain inversion: analytical 2x2 formula

**Spec section:** Algorithm, Algorithm 2 (line 163)

The smoother gain requires `inv(Sigma_pred[tau])`, which the pseudocode expresses via
a generic `inv()` call. Since the state dimension is always 2, a developer should use
the analytical 2x2 inverse formula:

```
For a 2x2 matrix [[a, b], [c, d]]:
inv = (1 / (a*d - b*c)) * [[d, -b], [-c, a]]
```

This avoids pulling in a general-purpose linear algebra solver for a trivial inversion,
and is both faster and more numerically transparent. The same applies to
`inv(Sigma_1)` in the joint log-likelihood if used.

**Fix:** Add a note in the pseudocode or data flow section stating that all matrix
inversions in this model are 2x2 and should be implemented analytically.


---

## Verification of Critique 1 Resolutions

All 12 issues from critique 1 were addressed:

| Issue | Status | Notes |
|-------|--------|-------|
| M1: Dynamic VWAP formula | Fixed | Lines 556-569 correctly implement recursive Eq 41 |
| M2: EM M-step phi before r | Fixed | Lines 227-230, 266-274 with explicit ordering comments |
| M3: Missing log-likelihood | Fixed | Lines 201-204 (innovation form), lines 296-306 (explanation) |
| N1: W_inv naming → S | Fixed | S[tau] used consistently throughout |
| N2: Robust EM E-step underspecified | Fixed | Lines 402-410 clearly state all three points |
| N3: Multi-step prediction placeholder | Fixed | Lines 521-529 have explicit A_step logic |
| N4: MAPE formula not defined | Fixed | Lines 799-807 define MAPE with all terms |
| P1: Covariance update form | Fixed | Lines 106-108 use Joseph form directly in pseudocode |
| P2: Log-normal bias correction | Fixed | Lines 486-489 include note on median vs mean |
| P3: VWAP tracking error formula | Fixed | Lines 809-818 define VWAP_TE |
| P4: Robust threshold clarity | Fixed | Line 359 uses `lambda * S[tau] / 2.0` with derivation |
| P5: Rolling warm-start | Fixed | Lines 746-749 implement warm-start in calibration |

## Additional Citation Verifications (New in This Round)

| Spec Claim | Paper Source | Verdict |
|---|---|---|
| Joseph form: (I-KC) Sigma (I-KC)^T + K*r*K^T | Researcher inference (standard practice) | Correctly marked as such |
| Innovation-form log-likelihood | Standard state-space result; Shumway & Stoffer (1982) | Correct formula and attribution |
| Log-normal bias correction note | Researcher inference | Correctly marked as such |
| Warm-start recommendation | Researcher inference | Correctly marked as such |
| MAPE definition (Eq 37) | Section 3.3, Eq 37 | Correct |
| VWAP_TE definition (Eq 42) | Section 4.3, Eq 42 | Correct |
| Robust phi (Eq 36) includes z* subtraction | Section 3.2, Eq 36 | Correct |
| Robust r (Eq 35) includes z* terms | Section 3.2, Eq 35 | Correct |
| Static VWAP TE, Robust KF = 6.85 | Table 4, Average row, static RKF mean | Correct |
| Static VWAP TE, Standard KF = 8.98 | Table 4, Average row | INCORRECT (8.98 is RKF std, not KF mean) |
| Static VWAP TE, CMEM = 8.97 | Table 4, Average row | INCORRECT (8.97 is dynamic KF std, not static CMEM mean) |
| Dynamic VWAP TE, Robust KF = 6.38 | Table 4, Average row | Correct |
| Dynamic VWAP TE, Standard KF = 6.39 | Table 4, Average row | Correct |
| Dynamic VWAP TE, CMEM = 7.01 | Table 4, Average row | Correct |

## Overall Assessment

Draft 2 is close to implementation-ready. The core algorithms (Kalman filter, smoother,
EM, robust extension, VWAP strategies) are correctly specified with precise paper
citations. The major algorithmic issues from critique 1 are fully resolved.

The remaining issues are:

1. **N1 (Static VWAP TE values):** Factual errors in validation reference values.
   Straightforward fix: re-read Table 4 columns carefully.

2. **N2 (Filter return signature):** An implementability gap that would cause a developer
   to backtrack when connecting the filter to the EM. Straightforward fix: update
   return statements to match what callers expect.

3. **P1-P3:** Minor clarity improvements that would make implementation smoother but
   are not blocking.

Fixing N1 and N2 would produce a specification ready for implementation.

# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model for Intraday Volume

**Direction:** 7, **Run:** 3
**Role:** Critic
**Date:** 2026-04-10

## Summary

Draft 3 resolves all 5 issues from critique 2 correctly and thoroughly. The spec
is now a high-quality, implementation-ready document. This review found 0 major
issues, 1 medium issue, and 3 minor issues. The medium issue affects correctness
only in the presence of missing observations and only for model comparison
purposes (not EM convergence). All issues are localized fixes requiring a few
lines each. The spec is ready for implementation.

---

## Resolution of Previous Issues

All 5 issues from critique 2 are resolved:

| ID | Issue | Resolution | Verified |
|----|-------|------------|----------|
| MD-NEW-1 | Log-likelihood formula used smoothed states (Q function) | Replaced with innovation-based prediction error decomposition (lines 349-382). Includes clear explanation of relationship to Eq A.8, correctly labeled as Researcher inference. | Yes -- standard Kalman filter PED form; S[tau] and e[tau] correctly defined |
| MD-NEW-2 | Dynamic VWAP interleaving not explicit | Full step-by-step procedure added (lines 564-614) with before/after observation structure, filter correction, forecast update, and explicit last-bin remainder | Yes -- matches Equation 41 intent and page 10 text |
| MN-NEW-1 | D_start missing upper bound | Now reads "k=1,2,...,T-1" with "(exactly T-1 elements)" parenthetical (line 272) | Yes |
| MN-NEW-2 | Cross-time P notation undefined | Explicit element definitions added at lines 267-270 with clarifying note | Yes |
| MN-NEW-3 | Static prediction A^h misleading | Replaced with explicit day-boundary + intraday decomposition (lines 507-529) with both iterative and closed-form expressions | Yes -- correctly shows a_eta applied once at day boundary, then a_eta_tau=1 intraday |

---

## Medium Issues

### MD-NEW-3. Log-likelihood constant term uses N instead of N_obs when missing observations are excluded

**Spec section:** Step 3, EM Algorithm, log-likelihood computation (lines 349-358)

The spec writes:

    log_lik^(j) = -0.5 * sum_{tau=1}^{N} [ ln(S[tau]) + e[tau]^2 / S[tau] ]
                  - (N/2) * ln(2*pi)

And then at line 358: "For missing observations (observed[tau] = false), exclude
that tau from the sum."

If the sum excludes missing observations, then the constant term should use
N_obs (the count of observed bins) rather than N:

    log_lik^(j) = -0.5 * sum_{tau: observed[tau]} [ ln(S[tau]) + e[tau]^2 / S[tau] ]
                  - (N_obs/2) * ln(2*pi)

where N_obs = |{tau : observed[tau] = true}|.

Using N in the constant term while summing over fewer than N terms produces an
incorrect log-likelihood value. This does not affect EM convergence (the
constant term cancels in the difference log_lik^(j) - log_lik^(j-1) since N_obs
is the same across iterations). However, it produces an incorrect absolute
log-likelihood, which matters if the value is used for:
- Model comparison via AIC = -2*log_lik + 2*k or BIC = -2*log_lik + k*ln(N_obs)
- Cross-validation across datasets with different numbers of missing observations
- Debugging by comparing against an independent log-likelihood computation

For liquid stocks (the paper's focus), N_obs approx N and the issue is negligible.
For any security with frequent zero-volume bins, the error grows with the
number of missing observations.

**Fix:** Replace the formula with:

```
N_obs = count of tau where observed[tau] = true

log_lik^(j) = -0.5 * sum_{tau: observed[tau]} [ ln(S[tau]) + e[tau]^2 / S[tau] ]
              - (N_obs/2) * ln(2*pi)
```

Also update the convergence check description (line 384) to note that the
relative change formula works correctly because N_obs is constant across
iterations (the same observations are missing in every E-step).

---

## Minor Issues

### MN-NEW-4. First EM iteration convergence check references undefined log_lik^(0)

**Spec section:** Step 3, EM Algorithm, convergence criterion (line 384)

The convergence check is:

    Until |log_lik^(j) - log_lik^(j-1)| / |log_lik^(j-1)| < tol  OR  j >= max_iter

On the first iteration (j=1), log_lik^(0) has never been computed (no E-step
has run before iteration 1). A developer implementing this literally would
access an uninitialized value.

**Fix:** Add a note: "On the first iteration (j=1), skip the convergence check
(or equivalently, initialize log_lik^(0) = -infinity)."

### MN-NEW-5. Innovation naming inconsistency between initial correction and log-likelihood

**Spec section:** Step 1 initial correction (lines 128-138) vs log-likelihood (lines 349-357)

The initial correction step at tau=1 computes `e_init` and `W_init`, while the
main loop computes `e[tau+1]` and `W[tau+1]`. The log-likelihood formula uses
`e[tau]` and `S[tau]` (where S = innovation variance = 1/W) and sums from
tau=1 to N. For tau=1, the developer must realize that:
- e[1] = e_init
- S[1] = 1/W_init = C * Sigma[1|0] * C^T + r

The main loop uses W (precision) while the log-likelihood uses S (variance).
The relationship S = 1/W is mathematically trivial but not stated.

**Fix:** Either:
(a) Add a note after the initial correction: "Store e[1] = e_init and
    S[1] = C * Sigma[1|0] * C^T + r for the log-likelihood computation." Or
(b) Use consistent naming: rename W to S^{-1} throughout, or define S[tau]
    alongside W[tau] in the main loop. Option (a) is simpler.

### MN-NEW-6. Dynamic VWAP initial forecast loop has ambiguous propagation ordering

**Spec section:** Step 6, Dynamic VWAP, initial forecast loop (lines 576-580)

The spec writes:

    For h = 1, ..., I:
      y_hat[tI+h|tI] = C * x_hat[tI+h|tI] + phi[h]
      volume_hat[h] = exp(y_hat[tI+h|tI]) * shares_out[t+1]
      (For h > 1, propagate: x_hat[tI+h|tI] = A_intraday * x_hat[tI+h-1|tI]
       where A_intraday = [[1,0],[0,a_mu]])

The parenthetical propagation note for h > 1 appears AFTER the forecast
computation, but logically the propagation must occur BEFORE the forecast
(x_hat[tI+h|tI] must exist before it can be used in the forecast). For h=1,
the state was already computed by the day-boundary transition (lines 573-574),
so no propagation is needed.

A developer reading top-to-bottom could implement the forecast before the
propagation, producing garbage values for h > 1.

**Fix:** Restructure as:

```
For h = 1, ..., I:
  if h > 1:
    x_hat[tI+h|tI] = A_intraday * x_hat[tI+h-1|tI]   -- propagate
  y_hat[tI+h|tI] = C * x_hat[tI+h|tI] + phi[h]
  volume_hat[h] = exp(y_hat[tI+h|tI]) * shares_out[t+1]
```

This matches the explicit ordering already used in the static prediction
section (Step 5, lines 514-516).

---

## Correctness Verification of New Material in Draft 3

| Spec Claim | Paper Source | Verified? |
|-----------|-------------|-----------|
| Innovation-based log-likelihood formula | Researcher inference (standard KF PED) | Correct -- sum of -0.5*(ln(S) + e^2/S) - (N/2)*ln(2pi) is the standard form (but see MD-NEW-3 re N vs N_obs) |
| Relationship to Equation A.8 explained | Appendix A.1, Equation A.8 | Yes -- correctly identifies A.8 as joint log-likelihood / Q function context |
| Innovation-based form labeled Researcher inference | N/A | Yes -- correctly labeled |
| Dynamic VWAP step-by-step procedure | Section 4.3, Equation 41, page 10 text | Yes -- interleaving of observe/correct/reforecast matches paper's description |
| Dynamic VWAP volume_hat[i] is forecast, not observation | Section 4.3, Equation 41 | Yes -- w uses volume^(d) which is the dynamic prediction |
| Static prediction explicit day-boundary | Section 2, A_tau definition | Yes -- a_eta applied once at boundary, then 1 intraday; mu decays as (a_mu)^h throughout |
| D_start has T-1 elements | Equations A.34, A.36 sum ranges | Yes -- k=1,...,T-1 gives T-1 day transitions |
| Cross-time P element notation | Consistent with P[tau] notation at lines 265-266 | Yes -- mirrors (1,1) and (2,2) element definitions |

---

## Overall Assessment

Draft 3 is an excellent implementation specification. All 10 issues from
critique 1 and all 5 issues from critique 2 have been correctly resolved across
the three drafts. The spec now provides:

- Complete, step-by-step pseudocode for all algorithms (filter, smoother, EM,
  robust extension, prediction modes, VWAP execution)
- Correct M-step formulas with proper dependency ordering and (j+1) superscripts
- Explicit missing observation handling integrated into the main pseudocode
- Innovation-based log-likelihood (the correct and standard form) with clear
  explanation of its relationship to the paper's Equation A.8
- Detailed dynamic VWAP procedure with explicit observation/correction/reforecast
  interleaving
- Explicit static prediction with day-boundary decomposition
- Comprehensive parameter table, initialization guidance, calibration procedure,
  validation benchmarks, edge cases, and known limitations
- Accurate and complete paper reference mapping

The single medium issue (N vs N_obs in the log-likelihood constant) is a
correctness issue only for implementations with missing observations and only
when the absolute log-likelihood value is used (not for convergence detection).
The three minor issues are presentational.

Issue count: 0 major, 1 medium, 3 minor.

**Recommendation:** The spec is ready for implementation. The medium issue
takes one line to fix. A developer can implement from this spec without reading
the paper, which was the target quality level.

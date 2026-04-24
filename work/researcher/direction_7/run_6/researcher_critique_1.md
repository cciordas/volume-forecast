# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model for Intraday Volume

**Direction:** 7, **Run:** 6
**Reviewer role:** Critic
**Draft reviewed:** `impl_spec_draft_1.md` (973 lines)

## Summary Assessment

The draft is comprehensive and well-structured, covering all six algorithms with
detailed pseudocode, 13 parameters, and thorough validation criteria. The paper
citations are generally precise and verifiable. However, I identified 5 major
issues and 8 minor issues that a developer would encounter when translating this
spec to code. The most critical is the incomplete cross-covariance recursion
pseudocode (Algorithm 2), which is the hardest part of the implementation and
currently contains placeholder code that contradicts the correct formula given
later in the same section.

---

## Major Issues

### M1. Cross-covariance recursion pseudocode is incomplete and self-contradictory (Algorithm 2, lines 285-317)

**Problem:** The pseudocode for the backward cross-covariance recursion breaks
down at lines 304-306. Line 304 has a placeholder comment (`L_tau_plus_1 = ...
# (stored from smoothing loop above)`), and line 306 gives a formula
(`Sigma_filt[tau+1] * L_tau^T`) that does not match the correct recursive formula
stated later in lines 309-314. The pseudocode block effectively gives up and
defers to a prose explanation outside the code block.

A developer copying the pseudocode would get the wrong answer from lines 304-306,
then find the correct formula in the prose and need to reconcile the two.

**Paper evidence:** Eq A.20 gives the recursive formula:

    Sigma_{tau,tau-1|N} = Sigma_{tau|tau} * L_{tau-1}^T
        + L_tau * (Sigma_{tau+1,tau|N} - A_tau * Sigma_{tau|tau}) * L_{tau-1}^T

Eq A.21 gives the initialization:

    Sigma_{N,N-1|N} = (I - K_N * C) * A_{N-1} * Sigma_{N-1|N-1}

**Required fix:** Replace the broken pseudocode (lines 285-317) with a clean,
self-contained backward loop that:

1. Stores all smoother gains L_tau from the smoothing loop (the spec mentions
   this but the smoother loop doesn't explicitly save them).
2. Initializes Sigma_cross at tau = N-1 using Eq A.21.
3. Iterates backward from tau = N-2 downto 1 using Eq A.20, where at each step:
   - `Sigma_cross[tau]` represents `Sigma_{tau+1,tau|N}` (make the indexing
     convention explicit)
   - The formula uses `L_{tau+1}`, `L_tau`, `A_{tau+1}`, `Sigma_filt[tau+1]`,
     and the previously computed `Sigma_cross[tau+1]`

Remove the prose "detailed formula" section and put everything inside the
pseudocode block. There should be one authoritative version, not two
contradictory ones.

**Severity:** Major -- this is the most implementation-critical algorithm detail
and the spec currently provides broken code for it.

---

### M2. Robust EM r update formula has inconsistent versions (Algorithm 5, lines 547-561)

**Problem:** The spec provides two versions of the robust r update. The first
(lines 547-553) contains the term `-2 * (z_star[tau])^2` which is incorrect.
The second "simplified" version (lines 556-561) contains `+ (z_star)^2` which
is correct. A developer reading top-to-bottom would implement the first
(wrong) version.

Additionally, line 552 has a stray superscript: `phi_new[bin(tau)]^{(j+1)}`
mixing notation styles (array indexing and iteration superscript), making the
formula harder to parse.

**Paper evidence:** Eq 35 in the paper reads (reconstructing from the
observation equation y = Cx + phi + v + z):

    r^(j+1) = (1/N) sum [ y^2 + CP_tau C^T - 2y C x_hat
               + (phi^(j+1))^2 + (z*)^2
               - 2y phi^(j+1) - 2z* y
               + 2 phi^(j+1) C x_hat + 2z* C x_hat + 2z* phi^(j+1) ]

The sign on `(z*)^2` is positive.

**Required fix:** Remove the first (wrong) version. Keep only one formula that
matches the paper. Express it cleanly in pseudocode notation without mixing
superscript styles.

**Severity:** Major -- incorrect formula would produce wrong parameter estimates
that silently degrade forecast quality.

---

### M3. Lambda limiting behavior described incorrectly (Variants section, lines 693-697)

**Problem:** The spec states the robust filter "subsumes the standard Kalman
filter as the special case lambda = infinity (no outliers ever detected) or
equivalently lambda = 0 with z_star always zero when the data is clean."

The second clause is wrong. When lambda = 0, the threshold
`lambda / (2 * W_tau) = 0`, so the soft-thresholding operator yields
`z_star = e_tau` for ANY nonzero innovation -- every observation is treated
as an outlier, not none. This is the opposite of what the spec claims.

**Paper evidence:** Eq 33 defines the soft-thresholding solution. When
threshold = 0, the cases collapse to z_star = e_tau - 0 = e_tau (for
e_tau > 0) or z_star = e_tau + 0 = e_tau (for e_tau < 0). Only e_tau = 0
gives z_star = 0.

**Required fix:** Remove the "lambda = 0" clause. Correct statement: "lambda
-> infinity yields threshold -> infinity, so z_star = 0 always (standard
Kalman filter). Conversely, lambda -> 0 yields threshold -> 0, so z_star
-> e_tau (all innovations are absorbed as outlier noise, and the Kalman
correction is effectively disabled)."

**Severity:** Major -- a developer using lambda = 0 as a default or testing
edge case would get catastrophically wrong behavior and might not realize it.

---

### M4. VWAP tracking error benchmarks likely cite wrong CMEM numbers (Validation section, lines 812-816)

**Problem:** The spec states CMEM achieves "8.97 dynamic / 10.91 static" basis
points for VWAP tracking error. However, the paper's text (Section 4.3,
paragraph after Table 4) states "an improvement of 9% when compared with the
dynamic CMEM." Working backward: if robust KF = 6.38 bps and the improvement
is 9%, then CMEM = 6.38 / 0.91 = 7.01 bps, not 8.97.

The value 8.97 appears in Table 4 but is likely the standard deviation of the
robust KF's static VWAP tracking (or another column), not the CMEM dynamic
mean. The table has 14 columns (7 methods x 2 stats each) and it is easy to
misread.

**Paper evidence:** Section 4.3, final paragraph: "the robust Kalman filter
with a dynamic VWAP strategy... gives an average VWAP tracking error of 6.38
bps... an improvement of 15% compared with the RM, and of 9% when compared
with the dynamic CMEM." 15% over RM: 6.38 / 0.85 = 7.51 (close to the
reported RM of 7.48). 9% over CMEM: 6.38 / 0.91 = 7.01.

The paper summary (chen_feng_palomar_2016.md) independently reports CMEM
dynamic VWAP = 7.01 bps and RM = 7.48 bps, consistent with the paper text.

**Required fix:** Verify CMEM and RM numbers against Table 4 column headers
carefully. Likely corrections: CMEM dynamic = 7.01, RM dynamic = 7.48. For
static VWAP, re-read the table and report correct values.

**Severity:** Major -- wrong benchmark numbers make validation unreliable. A
developer comparing their implementation against these targets would chase
phantom bugs or incorrectly declare success.

---

### M5. Missing log-likelihood formula for EM convergence monitoring (Algorithm 3, line 440)

**Problem:** The spec says "Compute log-likelihood (Paper, Eq A.8) or monitor
parameter changes. Terminate if relative change in log-likelihood < tol." But
Eq A.8 is the joint log-likelihood over hidden states and observations, which
requires the smoothed states and is not available during the E-step in a form
suitable for convergence checking.

The standard approach for monitoring EM convergence in state-space models is to
compute the innovations form of the log-likelihood during the forward filter:

    LL = -0.5 * sum_{tau=1}^{N} [ log(S_tau) + innovation_tau^2 / S_tau ]
         - (N/2) * log(2*pi)

where S_tau is the innovation variance and innovation_tau = y_tau - C *
x_pred[tau] - phi[bin(tau)]. This is a byproduct of the Kalman filter forward
pass and costs nothing extra to compute.

Without this formula, a developer must either guess the correct log-likelihood
computation (error-prone) or fall back to monitoring parameter changes (which
is less reliable as a convergence criterion).

**Required fix:** Add the innovations log-likelihood formula to Algorithm 3,
computed inside the forward filter loop and returned as an output. Cite the
standard Kalman filter log-likelihood identity (Shumway and Stoffer, or the
derivation from Eq A.3 in the paper).

**Severity:** Major -- convergence monitoring is the termination criterion
for the entire calibration procedure, and the spec gives no implementable
formula for it.

---

## Minor Issues

### m1. Smoother gain L_tau not explicitly stored in Algorithm 2 smoothing loop

**Problem:** The smoother loop (lines 264-283) computes L_tau at each step but
does not explicitly store it in an output array. The cross-covariance recursion
(which runs in a separate backward pass) requires all L_tau values. Line 304
acknowledges this with a placeholder. The `RETURN` on line 317 includes
`L_gains` but the loop never populates this array.

**Required fix:** Add `L_gains = array of N-1 2x2 matrices` to the storage
declarations and `L_gains[tau] = L_tau` inside the smoother loop.

---

### m2. MAPE formula not provided (Validation section)

**Problem:** The spec cites MAPE values and references Eq 37 but never states
the formula. A developer needs to know: (a) MAPE is computed on linear-scale
volumes, not log-scale, and (b) the normalization is by the actual volume, not
the predicted volume.

**Paper evidence:** Eq 37: MAPE = (1/M) * sum |volume_tau - predicted_volume_tau|
/ volume_tau, where the sum is over M out-of-sample bins.

**Required fix:** Add the MAPE formula in the Validation section or Data Flow
section, emphasizing that it operates on exponentiated (linear-scale) volumes.

---

### m3. VWAP tracking error formula not provided (Validation section)

**Problem:** The spec cites Eq 42 for VWAP tracking error but does not include
the formula. This is needed both for understanding the benchmarks and for
implementing the evaluation metric.

**Paper evidence:** Eq 42: VWAP^TE = (1/D) * sum_{t=1}^{D}
|VWAP_t - replicated_VWAP_t| / VWAP_t, where D is the number of out-of-sample
days.

**Required fix:** Add the formula, defining VWAP_t (Eq 39: sum of
price * volume / total volume) and replicated_VWAP_t (sum of
w_{t,i} * price_{t,i}).

---

### m4. Jensen's inequality bias in log-to-linear conversion not mentioned (Data Flow, Step 5)

**Problem:** Step 5 converts log-volume forecasts to linear scale via simple
exponentiation: `exp(y_hat)`. For a log-normal random variable,
E[X] = exp(E[log X] + Var[log X] / 2), so exp(E[log X]) systematically
underestimates E[X]. The paper uses exp(y_hat) without correction (as
evidenced by the good MAPE results), but this bias exists and could matter
for securities with high forecast variance.

**Required fix:** Add a note in Step 5 acknowledging the Jensen's inequality
bias and stating that the paper does not apply a correction factor. Optionally
mention that the correction would be exp(y_hat + S_tau / 2) where S_tau is the
innovation variance, but this was not used in the paper's evaluation.

---

### m5. Dynamic VWAP function signature unclear on forecast availability (Algorithm 6, lines 606-638)

**Problem:** The `dynamic_vwap_weights` function takes
`volume_forecasts_dynamic` as input, but at bin i, forecasts for bins i+1
through I have not yet been computed by the dynamic Kalman filter (which
produces one-step-ahead forecasts only). The function implicitly requires
multi-step-ahead forecasts for the remaining bins (i through I), but the spec
doesn't explain how these are obtained.

**Paper evidence:** Eq 41 uses `volume_{t,j}^(d)` for j = i to I. For j = i,
this is the one-step-ahead forecast. For j > i, these must be multi-step-ahead
forecasts produced by running the prediction step forward without correction
(similar to static_forecast but starting from the current filtered state at
bin i-1).

**Required fix:** Add a note explaining that at each bin i, the dynamic VWAP
strategy requires multi-step-ahead forecasts from bin i to I (obtained by
running the prediction-only forward pass from the current filtered state).
This is conceptually between "pure dynamic" (one-step) and "pure static"
(full-day) forecasting.

---

### m6. Cross-covariance array indexing convention undocumented

**Problem:** The spec declares `Sigma_cross = array of N-1 2x2 matrices` with
the comment `# Sigma_{tau,tau-1|N}` but does not state the index-to-meaning
mapping. From context, `Sigma_cross[tau]` appears to mean
`Sigma_{tau+1,tau|N}` (i.e., the cross-covariance between steps tau+1 and
tau). This is confusing because the array comment suggests
`Sigma_cross[tau] = Sigma_{tau,tau-1|N}`.

In the M-step, `P_cross[tau]` is defined as
`Sigma_cross[tau-1] + x_smooth[tau] * x_smooth[tau-1]^T`, confirming that
`Sigma_cross[tau-1] = Sigma_{tau,tau-1|N}`. So the array is 0-indexed
conceptually: `Sigma_cross[k]` = `Sigma_{k+1,k|N}` for k = 1, ..., N-1.

**Required fix:** Add an explicit index mapping comment:
`Sigma_cross[k] = Sigma_{k+1,k|N}` for k = 1, ..., N-1. This is critical for
avoiding off-by-one errors in the M-step.

---

### m7. Shares outstanding may not change daily (Data Flow, Step 1)

**Problem:** Step 1 normalizes by `shares_outstanding[t]` indexed by day. The
spec implies this changes daily. In practice, shares outstanding changes only
with corporate actions (stock splits, buybacks, new issuance) and is
otherwise constant. A developer might expect daily-frequency data.

**Required fix:** Clarify that shares_outstanding is the most recently reported
value as of day t, typically obtained from a corporate actions database or the
previous day's closing record. Most days it is unchanged.

---

### m8. Paper's Algorithm 3 integrates smoother and sufficient statistics in one pass

**Problem:** The spec separates the smoother (Algorithm 2) from the EM
(Algorithm 3) and has Algorithm 3 call Algorithm 2. This is architecturally
cleaner but differs from the paper's Algorithm 3, which computes smoother
estimates and sufficient statistics P_tau, P_{tau,tau-1} in a single backward
loop (lines 4-8). A developer cross-referencing with the paper might be
confused by the structural difference.

**Required fix:** Add a brief note in Algorithm 3 stating that the paper
combines the smoother and sufficient statistics computation into one backward
pass (Algorithm 3, lines 4-8), but the spec separates them for clarity. Both
produce identical results.

---

## Algorithmic Clarity Assessment

The pseudocode is generally translatable to code with the following exceptions:

1. **Algorithm 2 cross-covariance section** (M1): Not directly translatable --
   requires rewriting.
2. **Algorithm 5 r update** (M2): Two contradictory versions -- developer must
   choose and might choose wrong.
3. **EM convergence check** (M5): No formula given -- developer must look it up
   externally.

All other algorithms (Kalman filter, smoother main loop, EM M-step for
non-robust parameters, robust correction step, static/dynamic forecasting,
VWAP weights) are clear and directly translatable.

## Completeness Assessment

- **Parameters:** Complete. All 13 parameters documented with sensitivity and
  ranges.
- **Initialization:** Complete and well-reasoned.
- **Calibration:** Complete, including cross-validation procedure.
- **Edge cases:** 7 cases identified, all relevant.
- **Missing metric formulas:** MAPE (m2) and VWAP tracking error (m3) should be
  included since they are needed for validation.

## Citation Verification

I verified the following citations against the paper:

| Spec claim | Paper source | Verified? |
|---|---|---|
| Observation equation (Eq 3) | Section 2, Eq 3 | Yes |
| State transition (Eqs 4-5) | Section 2, Eqs 4-5 | Yes |
| Algorithm 1 lines 1-7 | Section 2.2, Algorithm 1 | Yes (reordered but equivalent) |
| Multi-step prediction (Eq 9) | Section 2.2, Eq 9 | Yes |
| Smoother (Algorithm 2) | Section 2.3.1, Algorithm 2 | Yes |
| Cross-covariance (Eqs A.20-A.22) | Appendix A.2 | Yes (but spec pseudocode is wrong) |
| EM M-step (Eqs A.32-A.39) | Appendix A.3 | Yes |
| Robust filter (Eqs 29-34) | Section 3.1 | Yes |
| Robust EM (Eqs 35-36) | Section 3.2 | Eq 36 yes; Eq 35 has sign error in spec |
| MAPE averages (Table 3) | Section 4.2, Table 3 | Yes (0.46/0.47/0.65/1.28) |
| VWAP averages (Table 4) | Section 4.3, Table 4 | CMEM number likely wrong (M4) |
| EM convergence (Figure 4) | Section 2.3.3 | Yes |
| Lambda behavior | Section 3.1, Eqs 33-34 | Lambda=0 claim wrong (M3) |

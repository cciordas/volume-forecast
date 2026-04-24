# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model for Intraday Volume

**Direction:** 7, **Run:** 6
**Reviewer role:** Critic
**Draft reviewed:** `impl_spec_draft_2.md` (1154 lines)

## Summary Assessment

Draft 2 is a substantial improvement over draft 1. All 5 major issues and 7 of 8
minor issues from critique 1 have been addressed. The cross-covariance recursion
(M1) is now correct and self-contained, the robust r formula sign error (M2) is
fixed, the lambda limiting behavior (M3) is corrected, the innovations
log-likelihood (M5) is properly integrated into Algorithm 1, and the evaluation
metric formulas (m2, m3) are now included.

No major issues remain. I found 4 minor issues, of which the most important is a
residual error in the VWAP benchmark numbers (partial fix of M4 from critique 1).
The spec is close to implementation-ready.

---

## Verification of Critique 1 Fixes

### M1 (Cross-covariance pseudocode): FIXED

The Algorithm 2 cross-covariance section (lines 311-355) is now a single,
self-contained backward loop. Verified against the paper:

- L_gains stored explicitly in the smoothing loop (line 299). ✓
- Initialization at k=N-1 using Eq A.21 (line 327): matches paper exactly. ✓
- Backward recursion from k=N-2 downto 1 using Eq A.20 (lines 344-353):
  correctly substitutes tau=k+1 in the paper formula, yielding
  `Sigma_cross[k] = Sigma_filt[k+1] * L_gains[k]^T + L_gains[k+1] *
  (Sigma_cross[k+1] - A_{k+1} * Sigma_filt[k+1]) * L_gains[k]^T`. ✓
- Index convention `Sigma_cross[k] = Sigma_{k+1,k|N}` documented at
  declaration (line 278-279) and in the recursion comments. ✓
- Prose "detailed formula" section removed; single authoritative version. ✓

### M2 (Robust r update sign error): FIXED

Lines 600-608 now contain a single correct formula with `+(z_star[tau])^2`.
Verified against Paper Eq 35: expanding E[(y - Cx - phi - z)^2] produces a
positive (z*)^2 term. The explicit derivation in the comments (lines 593-598)
makes the sign transparent. ✓

### M3 (Lambda limiting behavior): FIXED

Lines 797-802 now correctly state: lambda -> infinity gives threshold -> infinity,
so z_star = 0 (standard KF); lambda -> 0 gives threshold -> 0, so z_star -> e_tau
(all innovations absorbed as outlier noise). Verified against Eqs 33-34. ✓

### M4 (VWAP benchmarks): PARTIALLY FIXED

The robust KF numbers are correct (6.38 dynamic, 6.85 static) and the CMEM
dynamic number is corrected to 7.01. The cross-check against Section 4.3 text
is included. However, residual errors remain -- see m1 below.

### M5 (Log-likelihood formula): FIXED

Lines 152-156 add the innovations log-likelihood computation inside the forward
filter loop. Formula matches the standard Kalman filter identity. Convergence
check in Algorithm 3 (lines 474-477) uses relative change in this quantity. The
citation to Shumway and Stoffer (1982) and derivation from Eq A.3 is included. ✓

### m1 (L_gains stored): FIXED (line 299) ✓
### m2 (MAPE formula): FIXED (lines 762-767) ✓
### m3 (VWAP tracking error formula): FIXED (lines 769-777) ✓
### m4 (Jensen's inequality): FIXED (lines 748-754) ✓
### m5 (Dynamic VWAP multi-step): FIXED (lines 648-707) ✓
### m6 (Cross-covariance index convention): FIXED (lines 278-279, 313-315) ✓
### m7 (Shares outstanding): FIXED (lines 717-720) ✓
### m8 (Smoother/EM structural note): FIXED (lines 264-267) ✓

---

## Remaining Minor Issues

### m1. VWAP benchmark numbers still partially misattributed (residual from M4)

**Problem:** Lines 921-922 state:

> "For comparison: CMEM achieves 7.01 dynamic / not separately reported static;
>  rolling mean (RM) achieves 7.48 dynamic"

Two errors remain:

(a) CMEM static IS reported in Table 4. The Average row shows CMEM static VWAP
tracking error = 7.71 bps (mean) in the "Static VWAP Tracking" section. The spec
incorrectly says this is "not separately reported."

(b) RM 7.48 is from the STATIC column of Table 4, not dynamic. The Rolling Mean
is inherently a static strategy (it produces a fixed volume profile from historical
averages; there is no dynamic variant). Table 4 has no "Dynamic VWAP Tracking"
column for RM -- it appears only in the "Static VWAP Tracking" section.

The paper text (Section 4.3, final paragraph) compares dynamic RKF (6.38) to
static RM (7.48), which is a cross-strategy comparison: the best dynamic approach
vs a static baseline. The spec should make this distinction clear rather than
implying RM has a dynamic result.

**Paper evidence:** Table 4, Average row. Dynamic section has 3 methods (Robust KF,
KF, CMEM) x 2 stats = 6 columns. Static section has 4 methods (Robust KF, KF,
CMEM, RM) x 2 stats = 8 columns. Reading across: Dynamic RKF=6.38/8.86,
KF=6.39/8.97, CMEM=7.01/11.09; Static RKF=6.85/8.98, KF=6.89/9.09,
CMEM=7.71/11.16, RM=7.48/10.68.

**Required fix:** Replace lines 921-922 with:

> "For comparison: CMEM achieves 7.01 dynamic / 7.71 static; rolling mean (RM)
>  achieves 7.48 static (RM is inherently static -- no dynamic variant exists).
>  The paper's reported 15% improvement is dynamic RKF (6.38) vs static RM (7.48),
>  a cross-strategy comparison."

---

### m2. Robust EM convergence monitoring uses undefined log-likelihood

**Problem:** Algorithm 5 (robust EM, lines 573-626) says the E-step uses the robust
Kalman filter (Algorithm 4) instead of the standard filter. Algorithm 3 (standard
EM, lines 367-482) monitors convergence via the innovations log-likelihood computed
in Algorithm 1 (lines 152-156). But Algorithm 4 (robust correction) does not
compute or return a log-likelihood, and Algorithm 5 does not specify how to compute
it for the robust case.

When using the robust filter, the observation model is
y_tau = C*x_tau + phi_tau + v_tau + z_tau, and the z_star values are estimated
jointly in the forward pass. The raw innovation (y - C*x_pred - phi) includes the
outlier component. Two reasonable options exist:

(a) Compute the innovations log-likelihood using the CLEANED innovation
    (e_clean = e_tau - z_star) and the same S_tau. This measures how well the
    state-space model fits the data after outlier removal.

(b) Compute it using the raw innovation. This measures overall fit including
    outlier effects and would be noisier.

Neither is technically the marginal log-likelihood of the robust model (which
would require integrating over z_tau), but option (a) is more natural and
consistent with the filtering being done on the cleaned data.

**Required fix:** Add a note in Algorithm 5 specifying that the robust EM convergence
should use the cleaned innovations log-likelihood:
`LL = -0.5 * sum [ log(S_tau) + e_clean^2 / S_tau + log(2*pi) ]`,
or alternatively, state that monitoring parameter changes
(`max(|theta_new - theta_old| / |theta_old|) < tol`) is an acceptable fallback.
This is a practical choice, not something the paper specifies, so mark it as
Researcher inference.

---

### m3. Dynamic VWAP function input semantics ambiguous

**Problem:** The `dynamic_vwap_weights` function (line 648) takes `x_filt_current`
as input and `current_bin = i`. The comment (lines 653-654) says "At bin i
(1-indexed), we have observed bins 1..i-1." This means `x_filt_current` must be
the filtered state at bin i-1 (i.e., x_{i-1|i-1}), NOT at bin i.

But the parameter name `x_filt_current` and `current_bin = i` together suggest
"the filtered state at the current bin i," which a developer could reasonably
interpret as x_{i|i} (the state after observing bin i). Using x_{i|i} instead of
x_{i-1|i-1} would shift all forecasts by one bin, producing wrong VWAP weights.

**Required fix:** Either:
- Rename the parameter to `x_filt_prev_bin` and add a comment: "filtered state
  after observing bin i-1, i.e., x_{(i-1)|(i-1)}", or
- Change the input to accept `x_pred_current_bin` (the predicted state at bin i,
  x_{i|i-1}) and skip the first prediction step in the loop.

The key is making the input semantics unambiguous so a developer integrating this
function into the main Kalman filter loop knows exactly which state to pass.

---

### m4. Redundant conditional in dynamic_vwap_weights

**Problem:** Lines 679-693 contain an IF/ELSE with identical branches:

```
IF h == 0:
    A = [[1, 0], [0, a_mu]]
    Q = [[0, 0], [0, sigma_mu_sq]]
    x_next = A * x_curr
    remaining_forecasts[h] = C * x_next + phi[target_bin]
    x_curr = x_next
ELSE:
    A = [[1, 0], [0, a_mu]]
    ...  (identical code)
```

The h == 0 case was presumably intended to differ (perhaps using the predicted
state directly rather than re-predicting), but as written both branches are
identical. This is not a correctness issue but clutters the pseudocode and may
confuse a developer into thinking there should be a difference.

**Required fix:** Remove the IF/ELSE and use a single loop body. If the h == 0 case
was intended to handle a special transition (e.g., using the already-computed
predicted state from the main filter loop), add a comment explaining why it is
the same.

---

## Algorithmic Clarity Assessment

All algorithms are now directly translatable to code. The cross-covariance
recursion (previously the main blocker) is clean, complete, and well-documented
with explicit index mappings. The robust r formula has a single correct version
with a transparent derivation.

The only translational ambiguity is the dynamic VWAP function input (m3), which
could cause an off-by-one error if the developer misinterprets which filtered
state to pass.

## Completeness Assessment

- **Parameters:** Complete. All 13 parameters documented. ✓
- **Initialization:** Complete. ✓
- **Calibration:** Complete, including cross-validation. ✓
- **Edge cases:** 7 cases, all relevant. ✓
- **Metric formulas:** MAPE and VWAP^TE now included. ✓
- **Benchmark numbers:** Nearly complete; minor misattributions in VWAP table (m1).
- **Robust EM convergence:** Underspecified (m2).

## Citation Verification

All citations verified in critique 1 remain correct. New citations added in
draft 2 verified:

| Spec claim | Paper source | Verified? |
|---|---|---|
| Innovations log-likelihood (lines 152-156) | Derivable from Eq A.3; Shumway & Stoffer (1982) | Yes (standard identity) |
| Cross-covariance init (line 327) | Appendix A.2, Eq A.21 | Yes |
| Cross-covariance recursion (lines 344-353) | Appendix A.2, Eq A.20 | Yes (verified index substitution) |
| CMEM dynamic = 7.01 (line 921) | Table 4, Average row, CMEM dynamic mean | Yes |
| RM = 7.48 (line 922) | Table 4, Average row, RM static mean | Number correct; labeled "dynamic" but is actually static |
| 15%/9% improvement (lines 924-925) | Section 4.3, final paragraph | Yes |
| Smoother/EM structural note (lines 264-267) | Algorithm 3, lines 4-8 vs Algorithm 2 | Yes |

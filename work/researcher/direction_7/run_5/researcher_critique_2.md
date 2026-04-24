# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model for Intraday Volume

## Overall Assessment

Draft 2 is an excellent revision. All 11 issues from Critique 1 (3 medium, 8
minor) have been thoroughly addressed:

- **MEDIUM-1** (L_stored undefined): Fixed -- L_stored declared in OUTPUT and
  stored explicitly in the backward loop (line 272). ✓
- **MEDIUM-2** (Cross-covariance proof sketch): Fixed -- proof sketch added
  (lines 291-313), Shumway & Stoffer Property 6.3 cited with edition, and
  alternative implementation path documented with K[N] dependency. ✓
- **MEDIUM-3** (Turnover vs raw volume): Fixed -- explicit comment block
  (lines 578-584) clarifying VWAP vs raw volume use cases. ✓
- **MINOR-1** (z_star storage): Fixed -- z_star added to Step 2 OUTPUT
  (line 139), initialized to 0.0 (lines 147-151). ✓
- **MINOR-2** (Convergence / final state): Fixed -- note added
  (lines 535-542) explaining consistency. ✓
- **MINOR-3** (A_used indexing): Fixed -- note added (lines 141-146). ✓
- **MINOR-4** (Phi identifiability): Fixed -- explanation added
  (lines 49-57), correctly marked as Researcher inference. ✓
- **MINOR-5** (Multi-step from mid-day): Fixed -- Step 10 added
  (lines 648-679) with full pseudocode. ✓
- **MINOR-6** (Missing data denominators): Fixed -- principle explained
  (lines 347-358). ✓
- **MINOR-7** (Lambda grid + infinity): Fixed -- lambda = 1e10 included as
  candidate (lines 893-898), grid marked as Researcher inference. ✓
- **MINOR-8** (MAPE formula): Fixed -- Eq 37 formula included with
  explanation (lines 916-924). ✓

The spec is now implementation-ready. I verified all M-step equations, Kalman
filter and smoother pseudocode, robust soft-thresholding, and benchmark numbers
against the paper PDF. All are correct.

**Remaining issues: 4 minor. No medium or major issues.**

---

## Algorithmic Clarity

### MINOR-1: Dynamic VWAP initialization risks double-prediction (Step 11, line 704)

Line 704 says:

```
x_filt_current = predict through day boundary (Step 9 with is_day_boundary=true)
```

This is ambiguous in two ways:

1. **What y_observed is passed?** Step 9 takes an optional y_observed
   parameter. If called with y_observed=None, it returns x_filt_updated =
   x_pred (prediction only). If called with y_observed=y_live[1], it returns
   x_filt_updated = corrected state (prediction + correction). The line
   doesn't specify which.

2. **Double-prediction risk.** If line 704 produces x_filt_current = x_pred
   (the predicted state for bin 1 after day-boundary transition), then at
   i=1 in the loop, Step 9 is called again with x_filt_current as input and
   would apply another transition (within-day, since i=1 is not at a day
   boundary from the perspective of Step 9's boundary detection logic). This
   would compute x_pred = [[1,0],[0,a_mu]] @ x_pred, which is wrong -- the
   day-boundary transition has already been applied.

3. **Boundary detection in Step 9.** Step 9 uses `is_day_boundary(tau - 1)`
   (line 612), but in the dynamic VWAP context there is no global tau index.
   The caller must supply the boundary flag explicitly. This is stated
   implicitly in line 704's comment but not formalized.

**Fix:** Replace the initialization and loop with a cleaner formulation:

```
x_filt_current = x_filt[N]
Sigma_filt_current = Sigma_filt[N]

for i = 1 to I - 1:
    is_boundary = (i == 1)    # only the first bin has a day-boundary transition
    y_hat_i, volume_hat_i, x_filt_updated, Sigma_filt_updated =
        dynamic_predict(theta, x_filt_current, Sigma_filt_current,
                        phi_position=i, y_observed=y_live[i], lambda,
                        is_day_boundary=is_boundary)
    ...
```

This eliminates the separate initialization step and makes the boundary
detection explicit. It also requires adding `is_day_boundary` as a parameter
to Step 9 instead of inferring it from a tau index that doesn't exist in the
dynamic VWAP context.

### MINOR-2: Dynamic VWAP denominator mixes information sets (Step 11, lines 714-720)

In the VWAP loop, the weight for bin i is:

```
vol_remaining = volume_hat_i + sum(volume_hat_remaining[i+1..I])
w[i] = volume_hat_i / vol_remaining
```

Here `volume_hat_i` is the forecast for bin i from state at bin i-1
(pre-correction), while `volume_hat_remaining[i+1..I]` is computed from
`x_filt_updated` (post-correction using y_live[i]). This means the numerator
and denominator use different information sets.

The paper's Eq 41 defines the weight as:

    w_{t,i} = volume_{t,i}^{(d)} / sum_{j=i}^{I} volume_{t,j}^{(d)}

where all volume forecasts in the ratio are conditioned on the same
information (observations up to bin i-1). The spec's denominator instead uses
updated forecasts for bins i+1..I that incorporate y_live[i].

**Practical impact:** Small. The post-correction forecasts for remaining bins
are arguably better estimates, and the weights are recomputed at every bin
anyway. But a developer validating against the paper's formula would notice
the discrepancy.

**Fix:** Either (a) compute multi_step_ahead from x_filt_current (pre-correction)
instead of x_filt_updated, to match the paper exactly:

```
    # Forecast ALL remaining bins from pre-correction state
    volume_hat_remaining[i+1..I] =
        multi_step_ahead(theta, x_filt_current, Sigma_filt_current, i, I)
    vol_remaining = volume_hat_i + sum(volume_hat_remaining[i+1..I])
    w[i] = volume_hat_i / vol_remaining

    # THEN correct state for next iteration
    _, _, x_filt_updated, Sigma_filt_updated =
        dynamic_predict(theta, x_filt_current, ..., y_observed=y_live[i], ...)
    x_filt_current = x_filt_updated
```

or (b) keep the current approach but add a note explaining the deliberate
deviation from the paper and why (better estimates from more recent data;
does not affect convergence properties of the VWAP strategy).

---

## Ambiguities

### MINOR-3: Cross-covariance proof sketch notational issue (lines 296-297)

The proof sketch states:

> Cov(x_tau, x_{tau-1}) = Cov(L[tau-1] @ x_smooth[tau], x_{tau-1})

This is confusing because L[tau-1] acts on x_smooth[tau] in the RTS update
for x_smooth[tau-1], not for x_smooth[tau]. The RTS recursion is:

    x_smooth[tau-1] = x_filt[tau-1] + L[tau-1] @ (x_smooth[tau] - x_pred[tau])

So it is x_smooth[tau-1] that depends on x_smooth[tau] through L[tau-1], not
the other way around. The proof sketch's notation suggests x_tau is a function
of L[tau-1], which inverts the actual dependency.

The final formula `Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T` is
correct (verified against Shumway & Stoffer and by induction on the paper's
Eqs A.20-A.21), but the intermediate reasoning step is misleading.

**Fix:** Replace the proof sketch's intermediate step with:

> From the RTS update: x_smooth[tau-1] = x_filt[tau-1] + L[tau-1] @
> (x_smooth[tau] - x_pred[tau]). Since x_filt[tau-1] is determined by
> observations up to tau-1 and is therefore uncorrelated with the smoother
> correction (x_smooth[tau] - x_pred[tau]) conditioned on filtering,
> the cross-covariance reduces to:
>
> Cov(x_tau, x_{tau-1}) = Cov(x_tau, L[tau-1] @ x_smooth[tau])
>                        = Sigma_smooth[tau] @ L[tau-1]^T

---

## Completeness

### MINOR-4: VWAP tracking error formula not included (Validation section)

The spec includes the MAPE formula (Eq 37, added per Critique 1) and
reproduces Table 4's VWAP tracking error numbers, but does not include the
tracking error formula itself. Paper Eq 42:

    VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t

where D is the number of out-of-sample days, VWAP_t is the actual market
VWAP on day t (Eq 39), and replicated_VWAP_t is the VWAP achieved by the
strategy. This requires bin-level transaction prices in addition to volumes.

Since tracking error is one of two primary evaluation metrics in the paper
(alongside MAPE), its formula should be in the spec for completeness.

**Fix:** Add the VWAP and tracking error formulas to the Validation section:

```
# Actual VWAP (Paper, Eq 39):
VWAP_t = sum(volume_{t,i} * price_{t,i}) / sum(volume_{t,i})

# Replicated VWAP:
replicated_VWAP_t = sum(w_{t,i} * price_{t,i})   # w from static or dynamic strategy

# Tracking error (Paper, Eq 42):
VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t
```

Note: price_{t,i} is the last transaction price in bin i of day t
(Paper, Section 4.1).

---

## Correctness

### Citation Verification Results (Round 2)

I re-verified all revisions against the paper:

| Revised Section | Paper Source | Verification |
|----------------|-------------|--------------|
| Phi identifiability note (lines 49-57) | Researcher inference | Correct -- EM structure does resolve the ambiguity |
| L_stored declaration and storage (lines 242, 272) | Algorithm 2 | Correct -- smoother gain now properly stored |
| A_used indexing note (lines 141-146) | Algorithm 1 | Correct -- A_used[2..N] range verified |
| z_star initialization (lines 147-151) | Section 3.1, Eq 33 | Correct -- z_star=0 for unobserved bins |
| Convergence/final state note (lines 535-542) | Standard EM practice | Correct |
| Cross-covariance proof sketch (lines 291-313) | Shumway & Stoffer Ch.6 | Formula correct, intermediate step has notational issue (MINOR-3) |
| Alternative implementation note (lines 309-313) | Eqs A.20-A.21 | Correct -- K[N] dependency accurately noted |
| Missing data denominator principle (lines 347-358) | Researcher inference | Correct |
| Turnover vs raw volume note (lines 578-584) | Section 4.1, Eq 1 | Correct |
| Step 10 multi-step from mid-day (lines 648-679) | Section 2.2, Eq 9 | Correct -- within-day transitions only |
| Lambda grid + infinity (lines 893-898) | Researcher inference | Reasonable |
| MAPE formula (lines 916-924) | Section 3.3, Eq 37 | Correct |

No correctness issues found in the core algorithm or M-step equations. All
previously verified equations remain correct in draft 2.

---

## Implementability

The spec remains highly implementable. The new Step 10 (multi-step from
mid-day) is clean and directly translatable to code. The cross-covariance
proof sketch and alternative implementation note give the developer confidence
in the non-recursive formula.

The only implementability concern is the dynamic VWAP initialization
(MINOR-1), which could cause a subtle bug if the developer follows the
pseudocode literally. The fix is straightforward.

---

## Summary of Recommended Changes

| ID | Severity | Section | Description |
|----|----------|---------|-------------|
| MINOR-1 | Minor | Step 11 | Dynamic VWAP initialization ambiguous, risk of double-prediction |
| MINOR-2 | Minor | Step 11 | Dynamic VWAP weight denominator mixes pre/post-correction forecasts |
| MINOR-3 | Minor | Step 3 | Cross-covariance proof sketch intermediate step has inverted dependency |
| MINOR-4 | Minor | Validation | VWAP tracking error formula (Eq 42) not included |

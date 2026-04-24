# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model for Intraday Volume

**Direction:** 7, **Run:** 6
**Reviewer role:** Critic
**Draft reviewed:** `impl_spec_draft_3.md` (1196 lines)

## Summary Assessment

Draft 3 successfully addresses all 4 minor issues from critique 2. The VWAP
benchmark table is now correct and well-documented with cross-strategy
comparisons clearly labeled. The robust EM convergence monitoring is specified
with appropriate Researcher inference markings. The dynamic VWAP function input
semantics are unambiguous. The redundant branching is removed.

No major issues remain. I found 3 minor issues. The most important is a missing
day-boundary transition in the dynamic VWAP function when invoked at the first
bin of a trading day. The practical impact is small (a_eta is typically close to
1.0, and at bin 1 the dynamic strategy offers no advantage over static), but it
is a correctness defect that would produce slightly wrong forecasts.

The spec is implementation-ready. A competent developer can translate every
algorithm to working code from this document.

---

## Verification of Critique 2 Fixes

### m1 (VWAP benchmark misattributions): FIXED

Lines 931-939 now correctly state:
- CMEM: 7.01 dynamic / 7.71 static
- RM: 7.48 static (with explicit note that RM is inherently static)
- 15% improvement = dynamic RKF (6.38) vs static RM (7.48), cross-strategy
- 9% improvement = dynamic RKF (6.38) vs dynamic CMEM (7.01)
- Arithmetic verification included (6.38/0.85 = 7.51 ~ 7.48; 6.38/0.91 = 7.01)

Verified against Table 4 Average row and Section 4.3 final paragraph. All
numbers and attributions now match the paper. ✓

### m2 (Robust EM convergence undefined): FIXED

Lines 617-632 now specify:
- Primary: cleaned-innovation log-likelihood using e_clean and S_tau
- Fallback: max relative parameter change < tol
- Correctly marked as Researcher inference
- Reasoning is sound: cleaned-innovation LL is the natural analog of the
  standard EM's innovations LL ✓

### m3 (Dynamic VWAP input semantics): FIXED

Lines 665-675: parameter renamed to `x_filt_prev` with explicit documentation
that it is x_{(i-1)|(i-1)}, the filtered state after observing bin i-1. The
special case for i=1 (state from last bin of previous day) is noted. ✓

### m4 (Redundant IF/ELSE): FIXED

Lines 697-703: single loop body, no branching. All iterations use the same
within-day transition. ✓

---

## Remaining Minor Issues

### m1. Day-boundary transition missing in dynamic_vwap_weights for i=1

**Problem:** The `dynamic_vwap_weights` function (lines 697-703) always uses the
within-day transition matrix `A = [[1, 0], [0, a_mu]]` for every prediction
step. Lines 674-675 correctly note that when `current_bin = 1`, `x_filt_prev`
is the filtered state from the last bin of the previous day. But the first
prediction step (h=0) then transitions FROM the last bin of the previous day TO
the first bin of the new day -- this is a day-boundary crossing that should use
`A = [[a_eta, 0], [0, a_mu]]`.

Using the within-day matrix at this step means eta is carried forward unchanged
(multiplied by 1 instead of a_eta) and receives no process noise contribution
in the point forecast. Since a_eta is typically close to 1.0 (Paper, Figure 4a),
the numerical impact is small. Additionally, at i=1 the dynamic VWAP has no
information advantage over the static VWAP (no intraday observations yet), so
this primarily affects the relative weighting of bin 1 vs later bins, which is
a second-order effect on tracking error.

**Paper evidence:** The state transition at day boundaries uses
A = [[a_eta, 0], [0, a_mu]] with Q = [[sigma_eta_sq, 0], [0, sigma_mu_sq]]
(Paper, Section 2, Equations 4-5). Both the main Kalman filter loop (Algorithm
1, lines 184-188) and the static_forecast function (lines 234-237) correctly
handle this case. Only dynamic_vwap_weights omits it.

**Required fix:** Add a conditional for the first prediction step:

```
FOR h = 0 TO (I - i):
    target_bin = i + h
    IF h == 0 AND i == 1:
        # Day-boundary transition (previous day's last bin -> today's bin 1)
        A = [[a_eta, 0], [0, a_mu]]
    ELSE:
        # Within-day transition
        A = [[1, 0], [0, a_mu]]
    x_next = A * x_curr
    remaining_forecasts[h] = C * x_next + phi[target_bin]
    x_curr = x_next
```

**Severity:** Minor -- correctness defect with small practical impact due to
a_eta ~ 1.0 and the lack of information advantage at bin 1.

---

### m2. RM MAPE benchmark phrased imprecisely

**Problem:** Lines 921-922 state "rolling mean achieves 1.28 for both
(Paper, Table 3)." However, Table 3 reports the rolling mean only in the
Static Volume Prediction section (Average row: mean = 1.28, std = 5.54).
There is no Dynamic column for RM in Table 3, matching the same pattern as
Table 4 (where RM also appears only in the static section).

It is technically defensible that RM gives the same result regardless of
strategy label (since RM produces a fixed profile from historical averages and
does not update intraday), but the phrasing "for both" implies the paper reports
separate dynamic and static RM numbers, which it does not.

**Paper evidence:** Table 3, Average row. Dynamic section has 3 methods
(Robust KF, KF, CMEM) x 2 stats = 6 columns. Static section has 4 methods
(Robust KF, KF, CMEM, RM) x 2 stats = 8 columns. RM appears only in static.

**Required fix:** Change "rolling mean achieves 1.28 for both" to "rolling mean
achieves 1.28 static (RM is inherently static; Paper, Table 3 reports it only
in the static section)." This is consistent with the VWAP section (lines 932-
933) which already uses the correct phrasing for RM.

**Severity:** Minor -- documentation consistency with the VWAP section's
already-correct treatment of RM.

---

### m3. Algorithm 1 RETURN omits innovations and S_values arrays

**Problem:** Algorithm 1 (lines 127-128) declares `innovations` and `S_values`
arrays and populates them inside the loop (lines 159-160), but the RETURN
statement (line 208) does not include them:

    RETURN x_filt, Sigma_filt, x_pred, Sigma_pred, K_gains, log_lik

The `innovations` array is needed for sanity check #4 (lines 967-970: "Compute
the sample mean and ACF of innovations"). The `S_values` array is needed if
the developer wants to implement the Jensen's inequality bias correction
(line 763: "exp(y_hat + S_tau / 2)"). While these arrays could be recomputed
outside Algorithm 1, they are already computed and stored -- not returning them
is an oversight.

For the robust EM case, the analogous arrays (e_clean, S_tau) would need to be
stored and returned from the robust forward pass; this is mentioned in prose
(lines 619-624) but not reflected in a return signature.

**Required fix:** Add `innovations` and `S_values` to Algorithm 1's RETURN:

    RETURN x_filt, Sigma_filt, x_pred, Sigma_pred, K_gains, log_lik,
           innovations, S_values

**Severity:** Minor -- the information is already computed; this is a return
signature completeness issue.

---

## Algorithmic Clarity Assessment

All algorithms are directly translatable to code. The pseudocode is clean,
self-contained, and every step maps unambiguously to a computation. The only
translational issue is the dynamic_vwap_weights day-boundary case (m1), which
would produce a subtle bug detectable only by comparing against static_forecast
at bin 1.

## Completeness Assessment

- **Parameters:** Complete. All 13 parameters documented with sensitivity and
  ranges. ✓
- **Initialization:** Complete and well-reasoned. ✓
- **Calibration:** Complete, including cross-validation procedure. ✓
- **Edge cases:** 7 cases, all relevant. ✓
- **Metric formulas:** MAPE and VWAP^TE included with full definitions. ✓
- **Benchmark numbers:** Correct for all methods and strategies. ✓
- **Robust EM convergence:** Specified with appropriate Researcher inference. ✓
- **Revision history:** Comprehensive, tracing each fix to specific critique
  items. ✓

## Citation Verification

All citations verified in critiques 1 and 2 remain correct. Fixes introduced
in draft 3 verified:

| Spec claim | Paper source | Verified? |
|---|---|---|
| CMEM static = 7.71 (line 932) | Table 4, Average row, CMEM static mean | Yes |
| RM = 7.48 static only (lines 932-933) | Table 4, Average row; RM in static section only | Yes |
| 15% = dynamic RKF vs static RM (lines 935-936) | Section 4.3, final paragraph | Yes |
| 9% = dynamic RKF vs dynamic CMEM (line 936) | Section 4.3, final paragraph | Yes |
| Robust EM convergence (lines 617-632) | Researcher inference (correctly marked) | N/A -- not from paper |

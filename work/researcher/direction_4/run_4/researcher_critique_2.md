# Critique 2: Implementation Specification Draft 2 — Dual-Mode Volume Forecast (Direction 4, Run 4)

## Summary

Draft 2 comprehensively addresses all 3 major, 2 medium, and 6 minor issues raised in Critique 1. The exp-transformation for Nelder-Mead weight optimization is correctly formulated. The `make_state()` purity semantics are thoroughly documented with clear rationale. The `run_daily()` orchestration function fills the critical gap in the interaction protocol between Models A and B. The 11-term soft guardrail and blocked time-series cross-validation are both well-implemented.

Only 4 minor issues remain, none of which would block a competent developer. The spec is implementation-ready.

---

## Resolution of Critique 1 Issues

| Critique 1 Issue | Severity | Resolved? | Notes |
|-----------------|----------|-----------|-------|
| M1: Nelder-Mead bounds | Major | Yes | Exp-transformation in log-space (lines 286-306). Correctly uses log(1/3) initial values. |
| M2: ARMA state purity | Major | Yes | `make_state()` documented as pure (lines 329-339, 374-393). Clear explanation of why purity matters for both live and training usage. |
| M3: Missing orchestration | Major | Yes | `run_daily()` added (lines 646-739). Covers pre-market, market open, intraday loop, and end-of-day. |
| Med1: 11-term constraint | Medium | Yes | Soft guardrail with warning (lines 128-134, 190-194). Correctly cites paper language. |
| Med2: Temporal cross-validation | Medium | Yes | Blocked time-series CV with contiguous day blocks (lines 490-531). |
| m1: N_hist sourcing | Minor | Yes | Caveat added (lines 74-77), sensitivity upgraded to Medium-High (line 939). |
| m2: n_eff approximation | Minor | Yes | Explicit note added (lines 168-175). |
| m3: Missing MAPE formula | Minor | Yes | `compute_evaluation_mape` function added (lines 765-789). |
| m4: V_total_est note | Minor | Yes | Explanatory note added (lines 557-562). |
| m5: Undefined helpers | Minor | Yes | H_for_day / D_for_day / A_for_day logic inlined (lines 238-274). |
| m6: Renormalization propagation | Minor | Yes | Implicit renormalization explained (lines 620-636). |

---

## Remaining Minor Issues

### m1. `append_observation()` method undefined

**Location:** `run_daily`, lines 688-689.

**Problem:** The daily light update calls `model_a.interday_model[i].append_observation(volume[stock, bin=i, day=date-1])` to update the inter-day ARMA state with yesterday's volume. This method is not defined elsewhere in the spec. A developer needs to know:
- Does it extend the internal observation buffer, shifting out the oldest observation (sliding window)?
- Does it only update the state vector (most recent p observations and q residuals) for prediction purposes, without altering the fitted coefficients?
- If the model was fitted on 63 days and we append 20 observations over a month, does the prediction buffer grow to 83 observations, or does the oldest get dropped?

**Impact:** Low — the semantics are inferable (append for prediction, not re-fitting, since re-fitting happens monthly). A developer experienced with ARMA libraries would default to the correct behavior.

**Recommendation:** Add a one-line clarification: "append_observation() adds the new data point to the model's observation history for prediction purposes (updates the lag buffer used by predict()). Coefficients remain fixed until the next full re-estimation."

### m2. `compute_validation_mape` function signature incomplete

**Location:** Lines 792-832.

**Problem:** The function body references `N_hist` (line 817, `d - N_hist`) and `min_volume_floor` (line 809) but neither is in the function signature. The function takes `stock` and `historical_data` but the local variable names suggest direct parameter access rather than object attribute access.

**Impact:** Low — a developer would either pass these as additional parameters or access them from a config object.

**Recommendation:** Either add `N_hist` and `min_volume_floor` to the function signature, or note that they are accessed from a global configuration.

### m3. Daily light update mutates model_a despite purity emphasis

**Location:** `run_daily`, lines 680-689.

**Problem:** The spec strongly emphasizes that `forecast_raw_volume` is pure and does not modify `model_a` (lines 329-339). However, `run_daily` mutates `model_a.H[i]` and calls `append_observation()` during the pre-market phase. While this is not a logical error (mutation happens before intraday forecasting begins, not during), the juxtaposition may confuse a developer who reads the purity guarantees as applying to the model object at all times.

**Impact:** Low — the temporal separation (mutate pre-market, then read-only intraday) makes the behavior correct.

**Recommendation:** Add a brief clarification at the top of `run_daily`: "Pre-market updates mutate model_a in place. Once intraday forecasting begins, model_a is treated as read-only (forecast_raw_volume and forecast_volume_percentage are pure)."

### m4. `ols_no_intercept` helper not defined

**Location:** Lines 523, 535.

**Problem:** `ols_no_intercept(X, y)` is used in Model B training but not defined. While OLS without intercept is standard (beta = (X'X)^{-1} X'y), the spec defines other helper functions (`assign_regime`, `percentile_rank`, `compute_evaluation_mape`) at a similar level of triviality.

**Impact:** Very low — any developer knows OLS.

**Recommendation:** Add a brief one-line definition or note: "ols_no_intercept(X, y): ordinary least squares regression forced through the origin. Returns beta = (X'X)^{-1} X'y."

---

## Citation Verification (Draft 2)

Re-verified all citations against the paper PDF. No new misrepresentations introduced in draft 2.

| New/Changed Claim | Source | Verified? |
|-------------------|--------|-----------|
| Exp-transformation for weight non-negativity | Researcher inference (correctly marked) | N/A — design choice |
| make_state() purity semantics | Researcher inference (correctly marked) | N/A — design choice |
| run_daily orchestration | Researcher inference, synthesized from Exhibit 1 and pp.17-19 (correctly marked) | Yes — consistent with paper's data flow |
| Blocked time-series CV | Researcher inference (correctly marked) | N/A — standard practice |
| Soft guardrail for 11-term | Satish et al. p.18, "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms" | Yes — correctly identifies this as observation, not constraint |
| N_hist = 21 sensitivity "Medium-High" | Upgrade from Critique 1 recommendation | N/A — judgment call, reasonable |

---

## Overall Assessment

Draft 2 is a thorough, well-structured, and implementation-ready specification. All major and medium issues from Critique 1 have been properly resolved. The 4 remaining minor issues are clarification-level: a developer could implement correctly from the current draft without further revision.

**Severity summary:** 0 major, 0 medium, 4 minor.

**Recommendation:** This spec is ready for implementation. The 4 minor issues can be addressed as inline comments during development rather than requiring another revision round.

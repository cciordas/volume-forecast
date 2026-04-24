# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model for Intraday Volume

## Summary

Draft 2 is a substantial improvement over draft 1. The proposer addressed all 5 major and
all 8 minor issues from critique 1. The document is now detailed, well-structured, and
largely ready for implementation. I found 1 major issue (a correctness problem in the
Kalman filter Algorithm 1 citation mapping) and 4 minor issues. The spec is close to
final quality.

---

## Resolution of Critique 1 Issues

| Issue | Status | Notes |
|-------|--------|-------|
| M1 (filter loop structure) | Resolved | Loop now iterates over tau directly: forecast, correct, predict. Clear and matches Algorithm 1. |
| M2 (D set enumeration) | Resolved | D explicitly enumerated as {I+1, 2I+1, ..., (T-1)*I+1}, |D|=T-1 stated (lines 123-127). |
| M3 (missing log-likelihood) | Resolved | Innovation-based LL formula added (lines 174-178) with Shumway & Stoffer citation. |
| M4 (index mismatch standard vs robust) | Resolved | Step 5 now explicitly states it uses the same index convention as Step 3, iterating over tau with forecast/correct/predict at each tau (lines 345-346, 348-392). |
| M5 (dynamic VWAP logic) | Resolved | Rewritten as an online procedure with multi-step re-forecasting after each bin observation (lines 460-504). Explicit note on denominator interpretation added. |
| m1 (P_tau notation) | Resolved | Note added: "P_tau is the SECOND MOMENT (not the covariance)" (lines 101-104). |
| m2 (superscript notation) | Resolved | Explicit paragraph defining P^(i,j) as matrix element notation, not EM iteration (lines 43). |
| m3 (A^h multi-step) | Resolved | Multi-step prediction now separates day-boundary step from intraday steps, uses explicit A_boundary and A_intraday matrices with simplification shown (lines 250-284). |
| m4 (multi-step covariance) | Resolved | Covariance computed recursively alongside the mean in the multi-step loop (lines 263-270, 286-295). |
| m5 (MAPE definition) | Resolved | MAPE formula added with paper citation Eq 37 (lines 420-426). |
| m6 (VWAP TE definition) | Resolved | VWAP tracking error formula added with Eqs 39, 42 citations (lines 428-438). |
| m7 (lambda=0 edge case) | Resolved | Corrected to state filter runs in pure prediction mode, not standard KF (line 746). |
| m8 (warm-start guidance) | Resolved | Warm-start recommendation added in calibration section (lines 641-649). |

All 13 issues from critique 1 are adequately addressed.

---

## Major Issues

### M1. Kalman Filter Algorithm 1 Line References Are Shifted (Step 3, lines 228-240)

The spec maps its pseudocode steps to Algorithm 1 in the paper using line references
(e.g., "Alg 1, line 4", "Alg 1, line 5"). However, the mapping is incorrect because
the spec's loop structure differs from Algorithm 1's.

In the paper's Algorithm 1 (page 4):
- Line 2: predict mean: x_hat[tau+1|tau] = A_tau * x_hat[tau|tau]
- Line 3: predict covariance: Sigma[tau+1|tau] = A * Sigma[tau|tau] * A^T + Q
- Line 4: compute Kalman gain: K_{tau+1}
- Line 5: correct conditional mean: x_hat[tau+1|tau+1]
- Line 6: correct conditional covariance: Sigma[tau+1|tau+1]

The spec's Step 3 restructures this as (at each tau):
1. Forecast and innovate (using x_hat[tau|tau-1])
2. Correct to get x_hat[tau|tau] -- cited as "Alg 1, line 5"
3. Predict x_hat[tau+1|tau] -- cited as "Alg 1, line 2"

The issue: the spec cites "Alg 1, line 4" for the Kalman gain (line 229), "Alg 1, line 5"
for the corrected mean (line 232), and "Alg 1, line 6" for the corrected covariance
(line 236). But in Algorithm 1, these lines operate on index tau+1 (K_{tau+1},
x_hat[tau+1|tau+1], Sigma[tau+1|tau+1]), while the spec operates on index tau. The
equations are mathematically equivalent (just a relabeling), but citing specific Algorithm 1
line numbers while using a different index convention is misleading.

Similarly, the prediction step cites "Alg 1, line 2" and "Alg 1, line 3" (lines 239-240),
but in Algorithm 1 these come BEFORE the correction, while in the spec they come AFTER.

**Recommendation:** Either:
(a) Remove the Algorithm 1 line references and instead cite the equations by number
    (Eqs 7-8 for prediction, the Kalman gain formula from the text on page 4, Eq 9 for
    the observation forecast). Or
(b) Add a note: "The spec reorders Algorithm 1's steps to iterate over the observation
    index tau directly. The equations are identical; only the loop structure differs."

This is a major issue because a developer cross-checking against the paper will be confused
by the line number references not matching the spec's ordering or indexing convention.

---

## Minor Issues

### m1. Day Boundary Detection Condition Is Ambiguous (Step 3, lines 203-209)

The spec defines A_tau using the condition "tau mod I == 0" for day boundaries (line 204).
However, A_tau is used in the PREDICTION step (line 239):

    x_hat[tau+1|tau] = A_tau * x_hat[tau|tau]

This means A_tau governs the transition FROM tau TO tau+1. A day boundary occurs when
transitioning from the last bin of day t (tau = t*I) to the first bin of day t+1
(tau+1 = t*I+1). So the correct condition is: A_tau uses a_eta and sigma_eta^2 when
tau is the last bin of a day, i.e., tau mod I == 0.

The spec states this correctly (line 204: "tau mod I == 0"). However, the comment says
"tau is a day boundary" which is ambiguous -- is the boundary the last bin of day t or
the first bin of day t+1? The set D (line 127) defines day boundaries as the FIRST bins
(tau = kI+1), which is the opposite convention.

This creates a consistency issue: D uses "first bin of new day" as the boundary marker,
while A_tau/Q_tau use "last bin of old day" as the boundary marker. Both are correct
(they refer to the same transition), but a developer might be confused.

**Recommendation:** Add a clarifying note near lines 203-209: "The transition from tau to
tau+1 crosses a day boundary when tau is the last bin of a day (tau mod I == 0), equivalently
when tau+1 is the first bin of a new day (tau+1 in D). Both conditions identify the same
transition."

### m2. Robust EM r Update: Sign of z_star*C*x_hat Term (Step 5, lines 400-406)

The robust r update (line 405) includes:

    + 2*z_star[tau]*C*x_hat[tau|N]

Comparing with the paper's Eq 35 (page 7): the last term is +2*z_tau* * C * x_hat_tau.
This matches the spec.

However, the derivation from the squared residual (y - phi - Cx - z)^2 gives cross-terms:
- -2*y*z (present in spec as -2*z_star*y, line 404)
- +2*phi*z (present in spec as +2*z_star*phi, line 405)
- +2*z*Cx (present in spec as +2*z_star*C*x_hat, line 405)

Wait -- this is actually the expected value, so the cross-term 2*z*Cx becomes
2*z_star * C * E[x_tau | y_{1:N}] = 2*z_star * C * x_hat[tau|N]. The sign is correct.

**Retracted.** The equation is correct. No action needed.

### m2 (revised). Static Multi-Step Prediction: First Step Uses A_boundary But Simplified Formula Does Not Reflect This (Lines 272-276)

The multi-step prediction correctly separates the day-boundary step (line 261, using
A_boundary with a_eta) from subsequent intraday steps (line 268, using A_intraday with
a_tau_eta=1). However, the "simplification" comment at lines 273-276 states:

    x_hat[t*I+h | t*I] = [a_eta * eta_hat, (a_mu)^h * mu_hat]^T
    for h = 1, ..., I

This is correct for the eta component (a_eta is applied once at h=1 and then multiplied
by 1 for h=2..I, so eta stays at a_eta * eta_hat). For the mu component: at h=1 it is
a_mu * mu_hat, at h=2 it is a_mu * (a_mu * mu_hat) = (a_mu)^2 * mu_hat, etc. So at
step h it is (a_mu)^h * mu_hat. This is correct.

**Retracted.** The simplification is correct upon careful verification. No action needed.

### m2 (final). Robust EM: z_star Values Come from Robust E-Step But Spec Does Not Clarify Which EM Iteration's z_star (Step 5, lines 394-414)

The robust EM modification (lines 396-414) uses z_star[tau] in the M-step updates for r
and phi. These z_star values are computed during the robust Kalman filter forward pass
in the E-step (Step 5's filter). However, the spec doesn't make it explicit that z_star
values must be stored from the E-step's forward pass and then used in the M-step.

In the standard EM (Step 2), the E-step runs the filter and smoother, producing smoothed
quantities x_hat[tau|N] used in the M-step. For the robust EM, the z_star values are
computed during the FILTER (forward pass), not the smoother (backward pass). The spec
should clarify:
1. z_star[tau] for each tau is computed and stored during the robust Kalman filter
   forward pass.
2. These are then used (alongside the smoother outputs) in the M-step.
3. The smoother itself is the STANDARD RTS smoother (Step 4), not a "robust smoother."

Point 3 is important: the paper's robust extension only modifies the filter's correction
step (Eq 32) and the M-step updates for r and phi (Eqs 35-36). The smoother (Algorithm 2)
is unchanged. This is implied by the spec saying "All other M-step updates remain the
same" but is never explicitly stated for the smoother.

**Recommendation:** Add a brief note in Step 5 or in the robust EM section: "During robust
EM, the E-step runs (a) the robust Kalman filter (storing z_star[tau] for each tau),
followed by (b) the standard RTS smoother (Algorithm 2, unchanged). The M-step then uses
both the smoother outputs and the stored z_star values."

### m3. VWAP Tracking Error Units Conversion Not Specified (Lines 430-437)

The spec defines VWAP tracking error (line 430) as:

    VWAP_TE = (1/D) * sum |VWAP_t - replicated_VWAP_t| / VWAP_t

and states the result is "expressed in basis points (1 bps = 0.01%)" (line 437). However,
the formula produces a fraction (e.g., 0.000638 for 6.38 bps). The developer needs to
multiply by 10,000 to convert to basis points. This conversion step is not explicit.

The paper's Eq 42 (page 10) defines VWAP^TE without specifying the unit conversion
directly, but Table 4 reports values in basis points. A developer computing
|VWAP - repl_VWAP| / VWAP and getting ~0.0006 might not realize this needs to be
multiplied by 10,000 to match the paper's Table 4.

**Recommendation:** Add: "Multiply by 10,000 to express in basis points." Or write the
formula explicitly as:

    VWAP_TE (bps) = (10000/D) * sum |VWAP_t - replicated_VWAP_t| / VWAP_t

### m4. Dynamic VWAP: Eq 41 Weight Formula Structure (Lines 476-481)

The spec's dynamic VWAP weight formula (line 480) is:

    w[t+1,i] = (volume_forecast[i] / remaining_volume) * (1 - cumulative_weight)

Comparing with the paper's Eq 41 (page 10):

    w_{t,i}^(d) = volume_{t,i}^(d) / (sum_{j=i}^{I} volume_{t,j}^(d)) * (1 - sum_{j=1}^{i-1} w_{t,j}^(d))

These match: the first factor is the predicted volume share among remaining bins, and
the second factor is (1 - cumulative weight spent so far). The spec correctly implements
this with a cumulative_weight accumulator.

However, there is a subtle redundancy: this formula is equivalent to simply computing

    w[t+1,i] = volume_forecast[i] / sum_{j=1}^{I} volume_forecast[j]

when volume_forecast values do NOT change between bins (static case). In the dynamic case
where volume_forecast[j] is updated after each observation, the two formulas diverge
because the denominator changes. The spec's formulation with (1 - cumulative_weight) is
correct for the dynamic case.

**However**, the spec initializes volume_forecast for all I bins from static predictions
(lines 470-471), then updates them after each observation (lines 491-492). On the FIRST
bin (i=1), cumulative_weight=0 and remaining_volume = sum of all forecasts, so
w[t+1,1] = volume_forecast[1] / sum_all. This is correct and equivalent to the static
weight for the first bin.

The concern is what happens at i=1 BEFORE any observation has been made on the new day.
The filter state is from end of day t. The 1-step-ahead forecast for bin 1 of day t+1
crosses the day boundary. The spec's "static_predict" at line 471 produces this correctly
(it calls the multi-step prediction from Step 3 which handles the boundary). This is fine.

**Retracted as issue.** The implementation is correct.

---

## New Issues Not in Critique 1

### m4 (actual). Observation Noise Variance r Must Be Positive -- No Constraint in EM

The M-step update for r (Eq A.38 / line 151) computes r as a sample average of squared
terms. In theory, this is always positive because it is a sum of expected squared residuals.
However, in practice with finite precision arithmetic, a near-zero or slightly negative r
could occur if the model overfits. The spec does not mention any positivity constraint or
floor for r.

This is unlikely to be a practical issue given the problem structure (observation noise
is real in volume data), but worth a brief note.

**Recommendation:** Add to edge cases: "If r approaches zero or becomes negative due to
numerical issues, clip to a small positive value (e.g., 1e-10). This should not occur
with reasonable data."

---

## Citation Verification (Draft 2)

I verified the following new or changed citations against the paper:

| Spec Claim | Paper Source | Verified? |
|------------|-------------|-----------|
| Improvement percentages cite Section 5 (line 7) | Section 5 / Conclusion, page 11 | Yes, corrected from draft 1 |
| LL formula cites Shumway & Stoffer 1982 (line 173) | Not in paper; standard result | Appropriate external citation |
| MAPE Eq 37 (line 426) | Section 3.3, page 7 | Yes |
| VWAP TE Eq 42 (line 439) | Section 4.3, page 10 | Yes |
| Robust phi update Eq 36 (line 411) | Section 3.2, page 7 | Yes, matches Eq 36 |
| Multi-step prediction separates boundary (line 261) | Section 2.2, Eqs 7-8 + state-space definition | Correct; paper defines A_tau as time-varying at boundaries |
| Dynamic VWAP Eq 41 (line 480) | Section 4.3, page 10 | Yes, matches Eq 41 |
| Algorithm 1 line references (lines 229-240) | Algorithm 1, page 4 | Lines cited but index convention differs (see M1 above) |

All other citations from draft 1 remain accurate in draft 2.

---

## Overall Assessment

Draft 2 is a strong, nearly complete implementation specification. All 13 issues from
critique 1 were resolved. The remaining issues are:

1. **M1** (Algorithm 1 line references): misleading cross-references that will confuse
   a developer checking against the paper. Easy to fix by removing line numbers or
   adding a clarifying note.
2. **m1** (day boundary convention): ambiguity between "last bin of old day" and "first
   bin of new day" -- both correct but should be connected explicitly.
3. **m2** (robust EM z_star flow): clarify that z_star comes from the forward pass and
   the smoother is unchanged.
4. **m3** (VWAP TE units): add multiplication by 10,000 for basis points.

None of these issues would cause an incorrect implementation on their own, but M1 could
cause significant confusion during development and debugging. The minor issues are
polish items.

**Recommendation:** One more revision round should be sufficient to bring the spec to
final quality. The changes are small and localized.

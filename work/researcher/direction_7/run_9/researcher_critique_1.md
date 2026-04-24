# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model

## Summary

The draft is a strong first attempt — well-structured, comprehensive in scope, and
generally accurate in its treatment of the Chen et al. (2016) paper. However, it contains
two algorithmic bugs that would produce incorrect results if coded as written, one
significant clarity problem (two conflicting versions of the robust filter), and several
completeness gaps that would force a developer to consult the paper directly. I count
2 major issues, 1 high-medium issue, 5 medium issues, and 4 minor issues.

---

## Major Issues

### M1. Kalman Filter Loop Ordering Bug (Lines 100-124)

**Problem:** The pseudocode for Algorithm 1 places the prediction step BEFORE the
correction step within each loop iteration. At tau=1, the prediction step computes:

    x_hat_{2|1} = A_1 * x_hat_{1|1}

But x_hat_{1|1} does not exist yet — only x_hat_{1|0} is available from initialization.
The correction step (which computes x_hat_{1|1} from x_hat_{1|0}) appears BELOW the
prediction step. A developer executing this pseudocode top-to-bottom would reference an
undefined quantity.

**Trace through iteration tau=1:**
- Prediction: needs x_hat_{1|1} → UNDEFINED (only x_hat_{1|0} exists)
- Correction: computes x_hat_{1|1} = x_hat_{1|0} + K_1 * (...) → too late

**Paper evidence:** The paper's Algorithm 1 (page 4) avoids this by using a shifted
indexing convention: it predicts x_hat_{tau+1|tau} and then corrects x_hat_{tau+1|tau+1}
within the same iteration, so the prediction for step tau+1 uses x_hat_{tau|tau} from the
PREVIOUS iteration's correction output. However, this still requires an implicit initial
correction to obtain x_hat_{1|1} before the loop begins (since initialization gives
x_hat_{1|0}).

**Fix:** Reorder the loop body to correction-then-prediction:

```
Initialize: x_hat_{1|0} = pi_1, Sigma_{1|0} = Sigma_1

For tau = 1, 2, ..., N:
    # --- Correction step (update with observation y_tau) ---
    innovation = y_tau - C * x_hat_{tau|tau-1} - phi_tau
    S_tau = C * Sigma_{tau|tau-1} * C^T + r
    K_tau = Sigma_{tau|tau-1} * C^T / S_tau
    x_hat_{tau|tau} = x_hat_{tau|tau-1} + K_tau * innovation
    Sigma_{tau|tau} = Sigma_{tau|tau-1} - K_tau * S_tau * K_tau^T

    # --- Prediction step (propagate to tau+1) ---
    x_hat_{tau+1|tau} = A_tau * x_hat_{tau|tau}
    Sigma_{tau+1|tau} = A_tau * Sigma_{tau|tau} * A_tau^T + Q_tau

    # --- Forecast output ---
    y_hat_{tau+1|tau} = C * x_hat_{tau+1|tau} + phi_{tau+1}
```

This way, at tau=1, the correction uses x_hat_{1|0} (from init) to produce x_hat_{1|1},
and then the prediction uses x_hat_{1|1} to produce x_hat_{2|1}. All quantities are
defined before use.

### M2. Robust Filter Double-Subtraction Bug (Lines 275-276)

**Problem:** The spec defines:

    e_tau_modified = e_tau - z_tau*                          (line 275)
    x_hat_{tau+1|tau+1} = x_hat_{tau+1|tau} + K_{tau+1} * (e_tau_modified - z_tau*)   (line 276)

Substituting the definition of e_tau_modified:

    x_hat = x_hat_pred + K * ((e - z*) - z*) = x_hat_pred + K * (e - 2*z*)

This subtracts z* twice. The paper's Equation (32) (page 7) states:

    x_hat_{tau+1|tau+1} = x_hat_{tau+1|tau} + K_{tau+1} * (e_{tau+1} - z*_{tau+1})

which subtracts z* only once.

**Impact:** This would systematically over-correct for outliers, shrinking state updates
toward zero more aggressively than intended. On clean data (z* = 0) there is no effect,
so this bug would only manifest when outliers are present — exactly when the robust
filter is most needed.

**Fix:** Line 276 should read:

    x_hat_{tau+1|tau+1} = x_hat_{tau+1|tau} + K_{tau+1} * e_tau_modified

Or equivalently, remove the intermediate variable and write directly:

    x_hat_{tau+1|tau+1} = x_hat_{tau+1|tau} + K_{tau+1} * (e_{tau+1} - z*_{tau+1})

---

## High-Medium Issues

### HM1. Two Conflicting Versions of the Robust Filter (Lines 248-311)

**Problem:** The spec presents a first version of the robust correction step (lines
258-277), then says "Wait -- let me be more precise" (line 279) and presents a second
version (lines 281-311). This creates several problems:

1. A developer cannot tell which version is authoritative.
2. The first version contains the double-subtraction bug (M2) and uses inconsistent
   indexing (W_tau references Sigma_{tau+1|tau} but the innovation uses e_tau).
3. The second version (lines 281-291) is closer to the paper but still contains
   mixed notation (e_{tau+1} vs z_{tau+1}).
4. Lines 293-308 then provide a third representation (the soft-thresholding operator
   and the "clamped residual" interpretation).

**Fix:** Consolidate into a single, clean presentation of the robust correction step.
Delete the first attempt and the "Wait" transition. Present one authoritative version
with consistent indexing (either all tau or all tau+1), matching the paper's Equations
(30)-(34).

---

## Medium Issues

### ME1. Missing MAPE Formula (Validation Section)

**Problem:** The spec cites MAPE repeatedly as the primary evaluation metric (lines
475-488, sanity check 7-8) and references "Chen et al. (2016), Section 3.3, Equation
(37)" in the paper references table, but never actually defines MAPE. A developer
implementing validation needs this formula.

**Paper evidence:** Equation (37), page 7:

    MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of bins in the out-of-sample set and volumes are in the
original (linear, not log) scale.

**Fix:** Add the MAPE definition in the Validation section, ideally near the first
reference to MAPE values.

### ME2. Missing VWAP Tracking Error Formula

**Problem:** The spec reports VWAP tracking error in basis points (lines 490-494) and
references Equation (42) in the paper references table, but does not define the tracking
error metric.

**Paper evidence:** Equation (42), page 10:

    VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t

where D is the number of out-of-sample days, VWAP_t is the actual VWAP (Equation 39),
and replicated_VWAP_t uses model-predicted weights. The result is expressed in basis
points (multiply by 10000).

**Fix:** Add this formula alongside the VWAP weight definitions in the Data Flow section
(near lines 378-383) or in the Validation section.

### ME3. Missing Log-Likelihood Formula for EM Convergence

**Problem:** The spec states "Relative change in log-likelihood < epsilon" as the EM
convergence criterion (line 231) but does not provide the log-likelihood formula. A
developer cannot implement the convergence check without it.

**Two options from the paper:**

1. The expected complete-data log-likelihood Q(theta | theta^(j)) from Equation (A.10),
   pages 14-15 (computed from the sufficient statistics in the E-step).
2. The observed-data log-likelihood from the Kalman filter innovations:
   L = -(N/2)*log(2*pi) - (1/2)*sum_tau [log(S_tau) + innovation_tau^2 / S_tau]
   (standard result for linear Gaussian state-space models, not explicitly in the paper).

**Fix:** Provide one of these formulas. Option 2 (innovation-based) is simpler to
implement since S_tau and innovation_tau are already computed in the filter. Alternatively,
specify convergence based on relative parameter change (which the spec already mentions
as an alternative on line 231 with |theta^(j) - theta^(j-1)| < epsilon), but then define
what norm to use on the parameter vector.

### ME4. Missing Handling of Missing Observations in Algorithm 1

**Problem:** Edge case 1 (lines 541-548) correctly describes how to handle zero-volume
bins: skip the correction step and run only the prediction step. However, this logic is
not integrated into the Algorithm 1 pseudocode. A developer reading the pseudocode will
implement the standard loop; the edge case section is easy to miss and doesn't show HOW
to modify the loop.

**Fix:** Add a conditional to Algorithm 1's correction step:

```
if y_tau is observed (bin has nonzero volume):
    [run correction step]
else:
    x_hat_{tau|tau} = x_hat_{tau|tau-1}
    Sigma_{tau|tau} = Sigma_{tau|tau-1}
```

### ME5. VWAP Equation (39) Definition Missing

**Problem:** The spec references Equation (39) for VWAP definition (line 386) and uses
it in the tracking error, but the actual VWAP formula is not stated. The developer needs
to know that:

    VWAP_t = sum_{i=1}^{I} volume_{t,i} * price_{t,i} / sum_{i=1}^{I} volume_{t,i}

where price_{t,i} is the last transaction price in bin i. This definition, and the fact
that price data is required, should appear in the Data Flow section.

**Paper evidence:** Equation (39), page 8.

**Fix:** Add the VWAP definition and note that price data (last trade price per bin)
is a required input for VWAP evaluation.

---

## Minor Issues

### MI1. Covariance Update — Joseph Form Should Be Primary (Line 119)

**Problem:** The spec uses the standard covariance update
`Sigma_{tau|tau} = Sigma_{tau|tau-1} - K_tau * S_tau * K_tau^T` as the primary formula,
with the Joseph form mentioned only as a fallback in sanity check 4 (line 523). In
practice, the standard formula is numerically unstable (can produce non-positive-definite
covariances due to floating-point errors). Since the state dimension is only 2x2, the
Joseph form has negligible computational overhead.

**Fix:** Use the Joseph form as the primary covariance update in Algorithm 1:

    Sigma_{tau|tau} = (I - K_tau * C) * Sigma_{tau|tau-1} * (I - K_tau * C)^T + K_tau * r * K_tau^T

### MI2. Normalization Clarification (Lines 339-340)

**Problem:** The spec says "shares_outstanding_t (daily shares outstanding for
normalization)" but doesn't clarify that this is a single daily value applied uniformly
to all bins within the same day, nor that it typically comes from a separate reference
data source (not from the volume feed itself).

**Fix:** Add a note: "shares_outstanding_t is a single value per day (same for all bins
i within day t), typically sourced from a reference data provider."

### MI3. Table 3 Averages Need Clarification (Lines 476-488)

**Problem:** The spec quotes "average MAPE = 0.46 across 30 securities" from Table 3.
From the paper's Table 3, the "Average" row shows mean = 0.46 for dynamic RKF. However,
this is the average of per-ticker MAPEs, not a single global MAPE. The distinction
matters if a developer computes a single MAPE across all bins of all tickers vs. averaging
per-ticker MAPEs.

**Fix:** Clarify: "average of per-ticker dynamic MAPEs = 0.46 (Table 3, Average row)."

### MI4. Phi Update Ordering Note Could Be Stronger (Lines 239-241)

**Problem:** The spec correctly notes that phi must be updated before r in the M-step
(because the r update in Equation A.38 uses the newly computed phi^(j+1)). However, this
is buried in a note below Algorithm 3 and could easily be missed by a developer
implementing the M-step.

**Fix:** Integrate the ordering constraint directly into the M-step pseudocode by placing
the phi update BEFORE the r update (which the spec already does in the listing), and add
an inline comment: `# IMPORTANT: phi must be computed before r (r uses phi^(j+1))`.

---

## Verification of Citations

I verified the following key citations against the paper:

| Spec Claim | Paper Source | Verified? |
|---|---|---|
| Decomposition y = eta + phi + mu + v | Section 2, Eq (3) | Yes |
| State-space formulation x_{tau+1} = A_tau x_tau + w_tau | Section 2, Eqs (4)-(5) | Yes |
| A_tau time-varying (a^eta at day boundaries, 1 otherwise) | Section 2, page 3 | Yes |
| Q_tau time-varying ((sigma^eta)^2 at day boundaries, 0 otherwise) | Section 2, page 3 | Yes |
| Kalman filter Algorithm 1 structure | Section 2.2, page 4 | Yes, but ordering issue (M1) |
| Smoother Algorithm 2 (RTS) | Section 2.3.1, Algorithm 2, page 5 | Yes |
| Cross-covariance initialization Eq A.21 | Appendix A, Eq (A.21) | Yes |
| Cross-covariance recursion Eq A.20 | Appendix A, Eq (A.20) | Yes |
| EM M-step: a^eta update | Eq (A.34) vs draft line 207 | Yes |
| EM M-step: a^mu update | Eq (A.35) vs draft line 210 | Yes |
| EM M-step: (sigma^eta)^2 update | Eq (A.36) vs draft lines 213-214 | Yes |
| EM M-step: (sigma^mu)^2 update | Eq (A.37) vs draft lines 217-218 | Yes |
| EM M-step: r update | Eq (A.38) vs draft lines 221-223 | Yes |
| EM M-step: phi update | Eq (A.39) vs draft lines 226-227 | Yes |
| EM M-step: pi_1 update | Eq (A.32) vs draft line 200 | Yes |
| EM M-step: Sigma_1 update | Eq (A.33) vs draft line 203 | Yes |
| Robust filter soft-thresholding | Eqs (33)-(34), page 7 | Yes |
| Robust filter correction (Eq 32) | Eq (32), page 7 | Bug in draft (M2) |
| Robust EM: modified r | Eq (35), page 7 | Yes (matches draft lines 322-328) |
| Robust EM: modified phi | Eq (36), page 7 | Yes (matches draft line 330) |
| VWAP static weights | Eq (40), page 10 | Yes |
| VWAP dynamic weights | Eq (41), page 10 | Yes |
| Dynamic MAPE averages (0.46, 0.47, 0.65, 1.28) | Table 3, Average row | Yes |
| Static MAPE averages (0.61, 0.62, 0.90, 1.28) | Table 3, Average row | Yes |
| VWAP tracking error (6.38, 6.39, 7.01, 7.48) | Text on page 11 | Yes (6.38 confirmed; others from Table 4) |
| SPY dynamic MAPE ~0.24, static ~0.36 | Table 3, SPY row | Yes (RKF: 0.24 dynamic, 0.36 static) |
| EM convergence from multiple inits | Section 2.3.3, Figure 4 | Yes |
| Cross-validation period Jan-May 2015 | Section 4.1, page 8 | Yes |
| Out-of-sample period Jun 2015-Jun 2016, D=250 | Section 4.1, page 8 | Yes |

---

## Positive Notes

The following aspects of the draft are well done and require no changes:

1. The three-component decomposition and its rationale are clearly explained.
2. The EM M-step equations are all correct and properly cited.
3. The Kalman smoother (Algorithm 2) is correctly presented.
4. The sufficient statistics (x_hat, P_tau, P_{tau,tau-1}) are clearly defined.
5. The parameter table is comprehensive with useful sensitivity annotations.
6. The log-normal bias correction (limitation 2) is a valuable addition beyond the paper.
7. Edge cases are thoughtfully enumerated, especially day boundaries and half-day sessions.
8. The paper references table provides excellent traceability.
9. The distinction between static and dynamic prediction is clearly explained.
10. The data flow section provides a good end-to-end pipeline description.

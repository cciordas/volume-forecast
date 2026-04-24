# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model for Intraday Volume

## Summary

Draft 2 successfully addresses all 5 major and 8 minor issues from critique 1.
The filter initialization is now explicit and correct (M1). The summation
bounds for a_eta and sigma_eta_sq are fixed to T-1 terms (M2/m3). The
innovation variance notation is unified under S_tau (M3). The smoother's
unchanged role in the robust extension is documented (M4). The dynamic VWAP
formula correctly implements Eq 41's recursive structure (M5). The Q function,
MAPE definition, phi identifiability note, shares outstanding data
requirement, and forward pass storage are all added.

However, one correction in draft 2 introduced a new bug: the cross-covariance
recursion was changed from filtered to smoothed covariance (responding to
critique 1's m1), but **critique 1's m1 was incorrect**. The paper's Eq A.20
uses filtered covariance, and I have verified this mathematically. This is a
major regression. I also identify 3 minor issues.

---

## Major Issues

### M1. Cross-covariance recursion: draft 2 switched to SMOOTHED covariance, but the paper uses FILTERED

**Spec location:** Algorithm 2, lines 234-236.

**Problem:** Draft 1 used Sigma_filtered[tau] in the cross-covariance recursion.
Critique 1 (m1) claimed this was wrong and should use Sigma_smooth[tau],
citing Eq A.20. Draft 2 accepted this and changed to Sigma_smooth[tau].

**Critique 1's m1 was incorrect.** The paper's Eq A.20 uses Sigma_{tau|tau}
(filtered covariance), not Sigma_{tau|N} (smoothed covariance). I verified
this through three independent checks:

**Check 1: Consistency with initialization.** Eq A.21 initializes:

    Sigma_{N,N-1|N} = (I - K_N C) A_{N-1} Sigma_{N-1|N-1}

This uses Sigma_{N-1|N-1} (filtered at N-1). If the recursion A.20 used
smoothed covariance, the initialization would also need to use smoothed
covariance at N-1. But at N-1, smoothed != filtered (the smoother adds
information from y_N). The initialization uses filtered, so the recursion
should too.

The spec's own code is internally inconsistent: line 228 uses
Sigma_filtered[N-1] for the initialization, but lines 235-236 use
Sigma_smooth[tau] for the recursion. These cannot both be correct if
they belong to the same formula family.

**Check 2: Equivalence with Shumway & Stoffer (1982).** The paper cites
Shumway and Stoffer (1982), who give a simpler non-recursive formula for
the lag-one cross-covariance (their Property 6.3 / Eq 6.56):

    Sigma_{tau, tau-1|N} = Sigma_{tau|N} L_{tau-1}^T

where Sigma_{tau|N} is the smoothed covariance and L_{tau-1} is the smoother
gain. This formula requires no recursion at all.

I verified algebraically that the paper's Eq A.20 with FILTERED covariance
reduces to this simpler formula:

Starting from A.20 with filtered:
    Sigma_{tau,tau-1|N} = Sigma_{tau|tau} L_{tau-1}^T
        + L_tau (Sigma_{tau+1,tau|N} - A_tau Sigma_{tau|tau}) L_{tau-1}^T

Substituting Sigma_{tau+1,tau|N} = Sigma_{tau+1|N} L_tau^T (the S&S formula
at one step later) and using the identity A_tau Sigma_{tau|tau} =
Sigma_{tau+1|tau} L_tau^T (which follows from the definition L_tau =
Sigma_{tau|tau} A_tau^T Sigma_{tau+1|tau}^{-1}), the second term becomes:

    L_tau (Sigma_{tau+1|N} L_tau^T - Sigma_{tau+1|tau} L_tau^T) L_{tau-1}^T
    = L_tau (Sigma_{tau+1|N} - Sigma_{tau+1|tau}) L_tau^T L_{tau-1}^T

And from the smoother covariance update:
    Sigma_{tau|N} = Sigma_{tau|tau} + L_tau (Sigma_{tau+1|N} - Sigma_{tau+1|tau}) L_tau^T

So:
    Sigma_{tau|tau} + L_tau (Sigma_{tau+1|N} - Sigma_{tau+1|tau}) L_tau^T = Sigma_{tau|N}

Therefore:
    Sigma_{tau,tau-1|N} = Sigma_{tau|N} L_{tau-1}^T

This confirms that A.20 with filtered covariance is equivalent to the S&S
non-recursive formula.

**Check 3: A.20 with SMOOTHED covariance gives wrong results.** Substituting
Sigma_{tau|N} for Sigma_{tau|tau} in A.20:

    Sigma_{tau|N} L_{tau-1}^T
        + L_tau (Sigma_{tau+1,tau|N} - A_tau Sigma_{tau|N}) L_{tau-1}^T

The correct answer is Sigma_{tau|N} L_{tau-1}^T. So this formula is correct
only if the second term is zero:

    L_tau (Sigma_{tau+1,tau|N} - A_tau Sigma_{tau|N}) = 0

Expanding: L_tau (Sigma_{tau+1|N} L_tau^T - A_tau Sigma_{tau|N}). Since
A_tau Sigma_{tau|tau} = Sigma_{tau+1|tau} L_tau^T but
A_tau Sigma_{tau|N} != Sigma_{tau+1|tau} L_tau^T (because Sigma_{tau|N} !=
Sigma_{tau|tau} in general), this term is NOT zero. Therefore A.20 with
smoothed covariance gives incorrect cross-covariance values.

**Recommendation:** Revert the cross-covariance recursion to use FILTERED
covariance:

```
FOR tau = N-1 DOWN TO 2:
    Sigma_cross[tau] = Sigma_filtered[tau] @ L[tau-1]^T
        + L[tau] @ (Sigma_cross[tau+1] - A_stored[tau+1] @ Sigma_filtered[tau]) @ L[tau-1]^T
END FOR
```

Alternatively (and preferably), replace the entire cross-covariance
computation with the simpler non-recursive formula from Shumway & Stoffer:

```
# Cross-covariance: non-recursive formula (Shumway & Stoffer 1982)
# Sigma_{tau, tau-1|N} = Sigma_{tau|N} L_{tau-1}^T
FOR tau = N DOWN TO 2:
    Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T
END FOR
```

This eliminates the recursion entirely, removes the filtered/smoothed
confusion risk, removes the separate initialization step (A.21 is no longer
needed), and is computationally simpler. Since the smoother already computes
Sigma_smooth[tau] and L[tau-1], no additional quantities are needed.

Add a note: "This non-recursive formula is mathematically equivalent to the
recursive Eq A.20 (with filtered covariance) in the paper, as proved by
substituting the smoother gain definition and the smoother covariance update.
See Shumway & Stoffer (1982)."

(Paper, Appendix A.2, Equations A.20-A.21; Shumway & Stoffer 1982)

---

## Minor Issues

### m1. Q function E_1 through E_4 terms are referenced but not defined

**Spec location:** Lines 368-378 (Q function for convergence monitoring).

**Problem:** The spec gives the overall Q formula (Eq A.10) as:

    Q = -E_1 - E_2 - E_3 - E_4 - (N/2)log(r) - ...

but defines E_1 through E_4 only by reference: "defined in Equations A.11-A.14
using the sufficient statistics." A developer implementing convergence
monitoring would need to read the paper to find these formulas.

The spec is intended to be self-contained. Since the M-step updates are
derived from the partial derivatives of Q, the E terms are closely related to
the M-step sums. But the mapping is not explicit.

**Recommendation:** Either:
(a) Add the E_1-E_4 formulas explicitly. They are (from A.11-A.14):

```
E_1 = (1/(2r)) * SUM_{tau=1}^{N}
    [y_tau^2 + C P_tau C^T - 2 y_tau C x_hat_tau + phi_tau^2
     - 2 y_tau phi_tau + 2 phi_tau C x_hat_tau]

E_2 = SUM_{tau=2}^{N} (1/(2 sigma_mu_sq))
    [P_tau^(2,2) + a_mu^2 P_{tau-1}^(2,2) - 2 a_mu P_{tau,tau-1}^(2,2)]

E_3 = SUM_{tau in D'} (1/(2 sigma_eta_sq))
    [P_tau^(1,1) + a_eta^2 P_{tau-1}^(1,1) - 2 a_eta P_{tau,tau-1}^(1,1)]
    (where D' = {kI+1 : k=1,...,T-1})

E_4 = (1/2) (x_hat[1] - pi_1)^T Sigma_1^{-1} (x_hat[1] - pi_1)
    + (1/2) tr(Sigma_1^{-1} Sigma_smooth[1])
```

Note that E_1 simplifies to (N/(2r)) * r_update_value, E_2 simplifies to
((N-1)/(2 sigma_mu_sq)) * sigma_mu_sq_update_value, and E_3 simplifies to
((T-1)/(2 sigma_eta_sq)) * sigma_eta_sq_update_value. So after the M-step,
E_1 = N/2, E_2 = (N-1)/2, E_3 = (T-1)/2, and Q reduces to a function of
just the log-determinant terms plus E_4. This can simplify the convergence
check implementation.

Or (b) note that an alternative convergence criterion is parameter-change
monitoring: ||theta^{j+1} - theta^j|| / ||theta^j|| < tol, which avoids
computing Q entirely.

(Paper, Appendix A.1, Equations A.11-A.14)

### m2. Data preparation step contradicts edge case handling for zero-volume bins

**Spec location:** Calibration procedure, line 731e vs. Edge Cases, point 1.

**Problem:** The data preparation step says:
> "Exclude any bins with zero volume (log(0) is undefined)"

But the edge case section offers three options for zero-volume bins:
(a) exclude and treat as missing (skip correction), (b) impute, or
(c) the paper excludes them.

Option (a) (skip correction) preserves the day structure: the tau index still
advances through all I bins per day, but the Kalman correction is skipped for
zero-volume bins (using the "IF y[tau] is observed" branch in Algorithm 1).

However, the data prep instruction "exclude any bins" is ambiguous: does it
mean removing the bin from the time series (which breaks N = T * I and the
tau mod I day-boundary detection), or marking it as unobserved?

If individual bins are excluded (removed from the series), the linearized
index tau no longer satisfies the day structure: tau mod I != 0 at day
boundaries. This breaks the transition matrix logic.

**Recommendation:** Clarify the data preparation step:
- For training data: exclude entire days that contain any zero-volume bin
  (to preserve the N = T * I structure). Alternatively, keep the day and
  mark zero-volume bins as unobserved (skip correction but advance tau).
- For online prediction: mark zero-volume bins as unobserved and skip the
  correction step (the "IF y[tau] is observed" branch). Never remove bins
  from the tau sequence.

Add a note: "The tau index must always advance through all I bins per day,
even if some bins are unobserved. The day-boundary detection (tau mod I == 0)
requires the full I-bin day structure. Removing individual bins from the time
series will corrupt the transition matrix logic."

### m3. Sigma_1 EM update computed via subtraction instead of direct assignment

**Spec location:** Algorithm 3, line 302.

**Problem:** The spec computes:

    Sigma_1 = P[1] - x_hat[1] @ x_hat[1]^T

Since P[1] = Sigma_smooth[1] + x_hat[1] @ x_hat[1]^T (by definition of the
second moment), this simplifies to:

    Sigma_1 = Sigma_smooth[1]

Computing it as P[1] - outer(x_hat[1]) is numerically less stable than
directly using Sigma_smooth[1]: the subtraction of two positive semi-definite
matrices of similar magnitude can lose positive definiteness due to
floating-point errors.

**Recommendation:** Replace line 302 with:

    Sigma_1 = Sigma_smooth[1]    # = P[1] - x_hat[1] @ x_hat[1]^T (Eq A.33)

This is mathematically identical but numerically safer and clearer. Add a
comment explaining the equivalence.

---

## Assessment of Critique 1 Corrections

| Critique 1 Issue | Status in Draft 2 | Verdict |
|---|---|---|
| M1: Filter init off-by-one | Fixed: tau=1 uses pi_1 directly, no transition | Correct |
| M2: a_eta summation bounds | Fixed: k=1 to T-1, T-1 terms | Correct |
| M3: W_tau precision/variance | Fixed: unified under S_tau notation | Correct |
| M4: Robust smoother unspecified | Fixed: explicit statement smoother unchanged | Correct |
| M5: Dynamic VWAP formula wrong | Fixed: implements Eq 41 recursive formula | Correct |
| m1: Cross-cov filtered vs smoothed | Changed to smoothed | **INCORRECT -- must revert** |
| m2: No log-likelihood formula | Fixed: Q function reference and description added | Partial (E terms still external) |
| m3: sigma_eta_sq off-by-one | Fixed: same as M2 | Correct |
| m4: MAPE undefined | Fixed: Eq 37 formula added with scale clarification | Correct |
| m5: I=26 derivation | Fixed: marked as researcher inference | Correct |
| m6: phi identifiability | Fixed: added note, no constraint needed | Correct |
| m7: shares outstanding data req | Fixed: added to data requirements | Correct |
| m8: forward pass storage | Fixed: K_stored and A_stored documented | Correct |

---

## Completeness Assessment

**Improvements over draft 1:**
- Filter initialization is now explicit and correct.
- Summation bounds are precise with T-1 terms.
- Notation is consistent (S_tau throughout, no W_tau confusion).
- Dynamic VWAP pseudocode correctly implements the paper's recursive formula.
- Data requirements, storage requirements, and convergence monitoring are
  documented.
- Researcher inferences are clearly marked.

**Remaining gaps:**
1. Cross-covariance recursion is now incorrect (M1 above -- must revert or
   replace with simpler formula).
2. Q function E terms not fully explicit (m1 above -- moderate impact on
   self-containment).
3. Zero-volume bin handling is ambiguous (m2 above -- could cause production
   bugs).

---

## Verdict

Draft 2 is a significant improvement over draft 1 on 12 of 13 critique
points. The one regression (M1: cross-covariance using smoothed instead of
filtered covariance) is critical because it affects the sufficient statistics
that drive all EM parameter updates. This must be fixed.

The recommended fix is straightforward: either revert to filtered covariance
in A.20, or (better) replace the entire cross-covariance computation with the
simpler non-recursive formula Sigma_{tau,tau-1|N} = Sigma_{tau|N} L_{tau-1}^T.
The latter is preferable because it is simpler, non-recursive, and uses only
quantities already computed by the smoother.

After fixing M1 and addressing the minor issues, the spec should be ready
for implementation.

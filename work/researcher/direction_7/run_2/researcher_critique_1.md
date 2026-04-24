# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model for Intraday Volume

## Summary

The draft is well-structured and covers the core algorithm in substantial
detail. The Kalman filter, RTS smoother, EM algorithm, robust extension, and
VWAP weight computation are all present and largely correct. Paper citations
are thorough throughout. However, I identified 5 major issues and 8 minor
issues that could lead to incorrect implementations if not resolved.

---

## Major Issues

### M1. Kalman Filter Algorithm 1: Day-boundary detection is off-by-one

**Spec location:** Algorithm 1, lines 95-96.

**Problem:** The spec says:
> IF (tau - 1) is a day boundary (i.e., (tau - 1) mod I == 0 AND tau > 1)

This means the AR(1) transition for eta is applied when stepping FROM tau-1
TO tau, where tau-1 is the last bin of a day. That is conceptually correct:
the day boundary transition should fire when we cross from day k to day k+1.

However, the paper's notation is different. In the paper (Section 2, after
Equation 5), A_tau is the transition matrix used in x_{tau+1} = A_tau x_tau + w_tau.
So A_tau governs the transition FROM tau TO tau+1. The day-boundary set D
contains tau = kI (the last bin of each day), and at those tau values,
a_eta^tau = a^eta. At all other tau, a_eta^tau = 1.

In Algorithm 1 of the paper, the prediction step at iteration tau computes
x_{tau+1|tau} using A_tau. But the spec's Algorithm 1 iterates tau from 1 to N
and at each tau performs the prediction step to get x_{tau|tau-1} (using the
transition from tau-1 to tau). This is a valid reformulation, but the
day-boundary condition needs to check whether TAU-1 is in D, i.e.,
(tau-1) mod I == 0.

The spec's condition `(tau - 1) mod I == 0 AND tau > 1` is actually correct
for this reformulation. However, the comment says "(tau - 1) is a day
boundary" which could confuse a developer: tau-1 = 0 is NOT a valid day
boundary (it's before the first observation). The condition should be:
tau-1 >= 1 AND (tau-1) mod I == 0. When tau = 1, there is no prediction step
from a previous state -- the filter should use pi_1 and Sigma_1 directly.

**The real issue:** The spec's filter loop starts the prediction step at tau=1,
but at tau=1 there is no previous state to predict FROM. The paper's
Algorithm 1 initializes at tau=0 with x_{0|0} = pi_1, Sigma_{0|0} = Sigma_1,
then the loop runs from tau = 0 to N-1, computing x_{tau+1|tau} and then
correcting with y_{tau+1}. The spec conflates initialization with the first
prediction step. A developer could easily implement a prediction step at
tau=1 that applies A to pi_1, which would be correct only if the very first
transition is handled properly.

**Recommendation:** Restructure Algorithm 1 to explicitly handle initialization
separately:
```
x_hat = pi_1
Sigma = Sigma_1
FOR tau = 1 TO N:
    # Prediction: transition from tau-1 to tau
    IF tau == 1:
        # First bin: no transition, use initial state directly
        x_pred = x_hat    # = pi_1
        Sigma_pred = Sigma  # = Sigma_1
    ELSE:
        # Build A based on whether tau-1 is the last bin of a day
        IF (tau - 1) mod I == 0:
            A = [[a_eta, 0], [0, a_mu]]
            Q = [[sigma_eta_sq, 0], [0, sigma_mu_sq]]
        ELSE:
            A = [[1, 0], [0, a_mu]]
            Q = [[0, 0], [0, sigma_mu_sq]]
        x_pred = A @ x_hat
        Sigma_pred = A @ Sigma @ A^T + Q
    # Correction step...
```

Actually, looking more closely at the paper's Algorithm 1: it uses the
convention x_{tau+1|tau} where tau starts from 1. The predict step computes
x_{tau+1|tau} = A_tau x_{tau|tau}, and the correct step updates to x_{tau+1|tau+1}.
So the loop runs tau = 1, 2, ..., and at each iteration produces the filtered
state for time tau+1. The initial state x_{1|0} would require A_0 applied to
the initial condition, but the paper sets x_{1|0} = pi_1 directly (Equation 17
shows pi_1 = x_hat_1, the smoothed estimate at tau=1, not a predicted value).

This is actually subtler than the spec acknowledges. The spec should clarify
whether pi_1 represents x_{1|0} (predicted before seeing y_1) or x_{1|1}
(filtered after y_1). The EM update pi_1 = x_hat_1 (Eq A.32) sets it to the
smoothed value at tau=1, which is x_{1|N}. This means pi_1 plays the role of
the prior mean for x_1, and the first iteration should predict x_pred = pi_1
(no transition applied), then correct with y_1.

**The spec's current formulation will work if the developer skips the
transition at tau=1**, but this is not explicit enough. Add a clear note.

(Paper, Algorithm 1; Equations 7-8; Appendix A.3, Equation A.32)

### M2. EM M-step: a_eta summation indices are ambiguous

**Spec location:** Algorithm 3, lines 277-279.

**Problem:** The spec writes:
> numerator_eta = SUM over tau in D: P_cross[tau+1]^(1,1)
> denominator_eta = SUM over tau in D: P[tau]^(1,1)

where D = {I, 2I, 3I, ...}. This means the numerator sums P_cross at indices
{I+1, 2I+1, 3I+1, ...}, which are the first bins of days 2, 3, 4, ....

The paper's Equation A.34 writes:

    (a^eta)^(j+1) = [SUM_{tau=kI+1} P^(1,1)_{tau,tau-1}] / [SUM_{tau=kI+1} P^(1,1)_{tau-1}]

where the sum is over tau = kI+1 for k = 1, 2, .... This is equivalent: tau
ranges over {I+1, 2I+1, ...}, so P_{tau,tau-1} at those points are
cross-moments at day boundaries. The spec is consistent with this.

However, the spec's clarification paragraph (lines 333-341) says "tau = kI+1
(the first bin of each new day) for the numerator" but also says "P^(1,1) at
'tau = kI+1' to 'N' for a related sum," which is confusing and could be read
as a continuous range from kI+1 to N. The denominator should sum P[tau]^(1,1)
specifically at tau in {I, 2I, 3I, ..., (T-1)*I} -- i.e., the LAST bin of
each day except the last day. The spec says "SUM over tau in D" for the
denominator, giving P[tau]^(1,1) at {I, 2I, ..., (T-1)*I, TI=N}, but
P[N]^(1,1) should not be included because there is no day-boundary transition
OUT of the last day.

**Verification against paper:** Equation A.34 denominates with
SUM_{tau=kI+1} P^(1,1)_{tau-1}, summing tau from kI+1 for k=1,...,T-1. Since
tau-1 = kI, this sums P^(1,1) at {I, 2I, ..., (T-1)*I}. So the denominator
has T-1 terms. The numerator also has T-1 terms: P_{tau,tau-1}^(1,1) at
tau in {I+1, 2I+1, ..., (T-1)I+1}.

The spec's "SUM over tau in D" where D = {I, 2I, 3I, ...} includes tau = TI = N
in the denominator, giving T terms instead of T-1. This is a bug if T*I = N
and the last element of D is N.

**Recommendation:** Explicitly define the summation range:
- Numerator: SUM for k = 1 to T-1 of P_cross[kI+1]^(1,1)
- Denominator: SUM for k = 1 to T-1 of P[kI]^(1,1)

This gives exactly T-1 terms in each sum, matching Equation A.34.

(Paper, Appendix A.3, Equation A.34)

### M3. Robust Kalman filter: W_tau is used inconsistently as both variance and precision

**Spec location:** Algorithm 4, lines 370-388.

**Problem:** The spec defines W_tau as precision on line 370:
> W_tau = (C @ Sigma_pred @ C^T + r)^{-1}   # scalar: precision

But then on line 374, the threshold is:
> threshold = lambda / (2 * W_tau)

If W_tau is precision (the inverse of the innovation variance), then
lambda / (2 * W_tau) = lambda * (innovation_variance) / 2. This matches
Equation 34 in the paper: the "width" of the dead zone is lambda / (2 * W),
where W is the precision.

However, on line 387, the Kalman gain is computed as:
> K = Sigma_pred @ C^T * W_tau   # 2x1 Kalman gain (note: W is precision here)

This is correct: K = Sigma_pred C^T (C Sigma_pred C^T + r)^{-1} =
Sigma_pred C^T * W_tau when W_tau is precision.

**But the comment on line 388 contradicts this:**
> # Equivalently: K = Sigma_pred @ C^T / (C @ Sigma_pred @ C^T + r)

This equivalent form divides by the innovation variance, which is
1/W_tau. So K = Sigma_pred C^T * W_tau = Sigma_pred C^T / (1/W_tau).
The comment is correct but having both forms with different notation for the
same quantity invites bugs.

**More critically:** The "Key insight on W_tau" paragraph (lines 410-416) says:
> The Lasso threshold is lambda / (2 * W_{tau+1}), which equals
> lambda * (C Sigma_{tau+1|tau} C^T + r) / 2.

This correctly expands lambda/(2*W) = lambda * (innovation_variance)/2. But
it uses the notation W_{tau+1} from the paper, while the algorithm uses W_tau
at the current iteration. The paper's indexing has W_{tau+1} because its loop
index is offset. In the spec's loop (where tau is the current observation
index), this should just be W_tau.

**Recommendation:** Pick one convention and use it consistently. Since the
observation is scalar, define:
- S_tau = C @ Sigma_pred @ C^T + r  (innovation variance, scalar)
- K = Sigma_pred @ C^T / S_tau
- threshold = lambda * S_tau / 2

This avoids the precision/variance confusion entirely and matches standard
Kalman filter textbook notation. Drop the W_tau notation or clearly define
it as S_tau^{-1} in one place only.

(Paper, Section 3.1, Equations 30, 33-34)

### M4. Robust EM (Algorithm 5): Smoother interaction with z_star is unspecified

**Spec location:** Algorithm 5, lines 418-449.

**Problem:** The spec says "In the E-step, use the robust Kalman filter
(Algorithm 4) instead of the standard filter." But the EM E-step requires
not just the forward filter but also the backward RTS smoother
(Algorithm 2). The smoother uses the filtered estimates and the transition
matrices from the forward pass.

The question is: does the smoother also need to account for the detected
outliers z_star? Specifically:

1. In the robust filter, the observation residual used for correction is
   e_clean = e_tau - z_star (the innovation minus the detected outlier).
   The filtered state x_{tau|tau} is thus based on the cleaned innovation.

2. The smoother (Algorithm 2) takes the filtered estimates as input and does
   not re-process observations. So the smoother itself does not need
   modification -- it operates on the state estimates, which were already
   cleaned by the robust filter.

3. However, the sufficient statistics for the M-step depend on both the
   smoothed states AND the observations. Specifically, the r update
   (Equation 35) and phi update (Equation 36) subtract z_star from the
   observation residuals. This means the M-step needs access to the z_star
   values from the forward pass.

The spec correctly shows the modified r and phi updates but does not
explicitly state that the smoother is unchanged. A developer might wonder
whether the smoother needs modification for the robust case.

**Recommendation:** Add an explicit statement: "The RTS smoother (Algorithm 2)
is applied without modification to the filtered estimates from the robust
Kalman filter. The smoother operates on state estimates, not on observations
directly, so the outlier detection does not affect the smoothing recursion.
The z_star values from the forward pass are stored and used only in the
M-step updates for r and phi."

(Paper, Section 3.2: only modifies Equations 35-36, implying smoother is unchanged)

### M5. Dynamic VWAP weight formula (Algorithm 6) has an undefined variable

**Spec location:** Algorithm 6, lines 464-485.

**Problem:** The dynamic VWAP weight computation references `predicted_total_daily`
on line 482:
> w[current_bin] = vol_hat[current_bin] / remaining_total
>                  * (1 - actual_cumulative / predicted_total_daily)

But `predicted_total_daily` is never defined in the function. This variable
should be the total predicted volume for the day (sum of all I bin forecasts),
but it's unclear whether this uses the static forecasts made at market open
or the dynamic forecasts that update as the day progresses.

Moreover, the formula doesn't match the paper's Equation 41. The paper
defines the dynamic weight recursively:

    w_{t,i}^(d) = (volume_hat_{t,i}^(d) / SUM_{j=i}^{I} volume_hat_{t,j}^(d))
                  * (1 - SUM_{j=1}^{i-1} w_{t,j}^(d))

The key difference: the paper's formula uses the sum of PREVIOUS weights
(SUM_{j=1}^{i-1} w_{t,j}^(d)), not a ratio of actual cumulative to predicted
total. These are not equivalent because the dynamic weights w_{t,j}^(d) are
themselves recursively defined and may differ from actual_volume/predicted_total.

The spec does correctly quote Equation 41 on lines 496-498, but the
pseudocode in the function body (lines 464-485) does not implement this
formula. The pseudocode mixes actual volumes with predicted volumes in a way
that doesn't correspond to the paper.

**Recommendation:** Rewrite the dynamic VWAP function to directly implement
Equation 41:
```
FUNCTION compute_vwap_weights_dynamic(volume_hat_dynamic[1..I]):
    # volume_hat_dynamic[i] = exp(y_hat_dynamic[i]) for each bin
    # Weights are computed recursively

    cumulative_weight = 0
    FOR i = 1 TO I-1:
        remaining_predicted = SUM(volume_hat_dynamic[i..I])
        w[i] = (volume_hat_dynamic[i] / remaining_predicted) * (1 - cumulative_weight)
        cumulative_weight = cumulative_weight + w[i]
    END FOR
    w[I] = 1 - cumulative_weight
    RETURN w[1..I]
END FUNCTION
```

Note that the dynamic aspect is that volume_hat_dynamic[i] is re-forecast
at each bin i using all information up to bin i-1 (via the Kalman filter
correction). The weight formula itself is deterministic given the forecasts.

(Paper, Section 4.3, Equation 41)

---

## Minor Issues

### m1. Cross-covariance recursion in Algorithm 2 may have an error

**Spec location:** Algorithm 2, lines 208-211.

The spec gives:
> Sigma_cross[tau] = Sigma_filtered[tau] @ L[tau-1]^T
>     + L[tau] @ (Sigma_cross[tau+1] - A_{tau+1} @ Sigma_filtered[tau]) @ L[tau-1]^T

The paper's Equation A.20 gives:
> Sigma_{tau,tau-1|N} = Sigma_{tau|N} L_{tau-1}^T
>     + L_tau (Sigma_{tau+1,tau|N} - A_tau Sigma_{tau|N}) L_{tau-1}^T

Note the paper uses Sigma_{tau|N} (smoothed covariance) in both terms, but
the spec uses Sigma_filtered[tau] (filtered covariance) in the first term.
These are different quantities. The correct recursion should use the SMOOTHED
covariance Sigma_smooth[tau], not the filtered covariance.

**Recommendation:** Replace Sigma_filtered[tau] with Sigma_smooth[tau] in the
cross-covariance recursion. Also verify the A matrix index: the paper uses
A_tau in the second term where tau is the transition from tau to tau+1,
while the spec uses A_{tau+1} which would be the transition from tau+1 to tau+2.
Clarify which convention is intended.

(Paper, Appendix A.2, Equation A.20)

### m2. Log-likelihood for convergence check is not specified

**Spec location:** Algorithm 3, lines 320-323.

The spec says "relative change in log-likelihood < tol" but does not provide
the log-likelihood formula. The paper gives the complete log-likelihood in
Equation A.8 (Appendix A.1). The expected log-likelihood Q(theta | theta^(j))
is given in Equation A.10. The convergence check should use Q, not the
observed log-likelihood, since Q is what the EM monotonically increases.

**Recommendation:** Add the Q function formula (Equation A.10) or at least a
clear reference, and specify that convergence is checked on Q(theta^(j+1) |
theta^(j)) vs Q(theta^(j) | theta^(j-1)). Alternatively, monitor the
parameter vector change: ||theta^(j+1) - theta^(j)|| / ||theta^(j)|| < tol.

(Paper, Appendix A.1, Equation A.10)

### m3. sigma_eta_sq M-step: denominator should be T-1, but spec's text and formula disagree

**Spec location:** Algorithm 3, lines 287-291.

The spec correctly says:
> sigma_eta_sq = (1 / (T - 1)) * SUM over tau in D: [...]

And the comment notes "T-1 because there are T-1 day transitions in T days."
But the summation "over tau in D" where D = {I, 2I, ..., TI} would have T
terms, not T-1. The same off-by-one issue as M2 applies here: the sum should
run over D' = {I, 2I, ..., (T-1)I}, giving T-1 terms, or equivalently over
the first bins of days 2 through T.

This is consistent with Equation A.36 in the paper, which sums from
tau = kI+1 for k = 1, ..., T-1.

**Recommendation:** Fix the summation range to exclude the last day boundary,
consistent with M2.

(Paper, Appendix A.3, Equation A.36)

### m4. MAPE definition is missing

**Spec location:** Validation section, lines 677-706.

The spec quotes MAPE results extensively but never defines MAPE. The paper
defines it in Equation 37:

    MAPE = (1/M) SUM_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of bins in the out-of-sample set, and volumes are
in the original (non-log) scale. This is important because a developer might
compute MAPE on log-volumes or on normalized volumes, getting different numbers.

**Recommendation:** Add the MAPE formula and clarify that it operates on
original-scale volumes (after exponentiating log-volume predictions and
multiplying by shares outstanding, or equivalently on normalized volumes
since shares outstanding cancels in the ratio).

(Paper, Section 3.3, Equation 37)

### m5. Bins per day I = 26 for NYSE is stated without derivation

**Spec location:** Parameters table, line 600.

The spec says "26 for NYSE 15-min bins (6.5 hours / 15 min)." This gives
6.5 * 60 / 15 = 26 bins. However, the paper (Section 4.1) says "the total
number of bins per day I varies from market to market, depending on the
trading hours of the stock exchange." It does not specify I = 26 for NYSE.

The computation is straightforward (9:30 AM to 4:00 PM = 6.5 hours = 26
fifteen-minute bins), but the spec should note this is a derived value, not
one stated in the paper. Also clarify whether the first bin starts at 9:30
or 9:45 (i.e., whether the bins are [9:30, 9:45), [9:45, 10:00), ... or
[9:30, 9:45], [9:45, 10:00], ...), as this affects whether there are 26 or
27 bins.

**Recommendation:** Mark as "Researcher inference" and clarify bin boundary
convention.

### m6. phi constraint: sum of phi should be documented

**Spec location:** Parameters table, line 595.

The seasonality vector phi is estimated as the average residual per bin
(Equation 24/A.39). There is no identifiability constraint documented: both
eta and phi contribute to the mean level of y. If you add a constant c to
phi and subtract c from eta, the observation equation is unchanged. The EM
update for phi (Equation 24) resolves this implicitly because eta is estimated
jointly, but a developer implementing this independently might add a
sum-to-zero constraint on phi (as is common in seasonal decomposition),
which would conflict with the EM updates.

**Recommendation:** Add a note that no explicit constraint on phi is needed --
the EM updates for eta and phi jointly determine the split between the daily
level and the seasonal pattern. However, note that this means the absolute
level of phi and eta is not uniquely determined in isolation; only their sum
eta + phi[i] is identifiable from the observation equation.

(Paper, Equation 24; identifiability is implicit in the model structure)

### m7. Shares outstanding data requirement is understated

**Spec location:** Calibration procedure, line 636.

The spec says "Normalize each bin's volume by shares outstanding for that day"
but the data requirements section doesn't list shares outstanding as a
required input. For a developer building this system, knowing that they need
daily shares outstanding data (which can change due to buybacks, issuances,
splits) is important for data acquisition.

**Recommendation:** Add "daily shares outstanding per security" to the data
requirements, with a note about handling corporate actions.

### m8. Missing guidance on handling the first EM iteration's smoother

**Spec location:** Algorithm 3 / Algorithm 2 interaction.

The cross-covariance initialization in Algorithm 2 (line 206) requires K_N
(the Kalman gain at the last time step) and A_N (the transition matrix at
the last step). These come from the forward pass. The spec should note that
the developer must store K_tau and A_tau from the forward pass for use in
the backward smoother, not just x_filtered and Sigma_filtered.

**Recommendation:** In Algorithm 1, add to the stored outputs: K[tau] and
A_tau (or at least the day-boundary flag for each tau, from which A_tau can
be reconstructed). In Algorithm 2's inputs, list these explicitly.

(Paper, Algorithm 2 line 2 uses L_tau which requires Sigma_{tau|tau-1}, not
just Sigma_{tau|tau})

---

## Completeness Assessment

**What's well done:**
- The three-component decomposition is clearly explained.
- Pseudocode for all five core algorithms is provided.
- The data flow diagram is excellent and gives a clear end-to-end picture.
- Parameter table is comprehensive with sensitivity assessments.
- Sanity checks are thoughtful and actionable.
- Edge cases cover the most important production concerns.
- Paper citations are thorough and mostly accurate.

**What's missing or underspecified:**
1. The MAPE metric definition (m4).
2. Storage requirements for the forward pass (m8).
3. Shares outstanding as an explicit data requirement (m7).
4. The Q function for convergence monitoring (m2).
5. No discussion of computational complexity. The state is 2D so this is
   O(N) per EM iteration, but worth stating explicitly.

---

## Verdict

The spec is a strong first draft. The two most critical issues are M5
(dynamic VWAP formula is wrong in the pseudocode) and M1 (filter
initialization ambiguity). M2/m3 are essentially the same off-by-one in the
a_eta summation bounds. M3 is a notational consistency issue that could cause
subtle sign errors. M4 is a missing clarification rather than an error.

After addressing these issues, the spec should be implementable by a
developer who has not read the paper.

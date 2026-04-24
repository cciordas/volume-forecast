# Heterogeneous Component Multiplicative Error Models for Forecasting Trading Volumes
**Authors:** Antonio Naimoli, Giuseppe Storti
**Year:** 2019
**DOI:** N/A (MPRA Paper No. 93802)

## Core Contribution
This paper introduces the Heterogeneous MIDAS Component Multiplicative Error Model
(H-MIDAS-CMEM) for modeling and forecasting high-frequency intra-daily trading volumes.
The model extends the Component MEM (CMEM) of Brownlees et al. (2011) by replacing the
simple GARCH-type long-run component with an additive cascade of MIDAS (Mixed Data
Sampling) polynomial filters operating at different frequencies — daily and sub-daily (hourly).
This heterogeneous specification is motivated by the Heterogeneous Market Hypothesis of
Muller et al. (1993), which posits that market participants operate at distinct frequencies, each
driving volume variation at a different speed.

The paper makes three main contributions: (1) it proposes the H-MIDAS-CMEM model and
establishes conditions for stationarity and ergodicity of the components; (2) it develops a
two-stage estimation procedure using the (Dynamic) Zero-Augmented Generalized F distribution
to handle the discrete probability mass at zero volumes; and (3) it demonstrates through an
empirical study on six Xetra-traded German stocks that the H-MIDAS-CMEM outperforms all
benchmark models in both in-sample fit and out-of-sample forecasting.

## Model/Algorithm Description
The H-MIDAS-CMEM decomposes intra-daily volume x_{t,i} (day t, interval i) multiplicatively
into four components:

    x_{t,i} = tau_{t,i} * g_{t,i} * phi_i * epsilon_{t,i}

**Periodic seasonal component (phi_i):** Captures the U-shaped intra-daily pattern via a
Fourier Flexible Form (Gallant, 1981) with polynomial trend of order Q and P sine/cosine
pairs. Selected by BIC; empirically Q=2, P=6 works well. Estimated by OLS in stage one.

**Long-run component (tau_{t,i}):** The model's key innovation. Specified as an additive
cascade of MIDAS filters:

    tau_{t,i} = m + theta_d * sum_k phi_k(omega_d) * YD^(k)_{t,i}
                  + theta_h * sum_{k,l} phi_{l,k}(omega_h) * YH^(l,k)_{t,i}

where YD^(k) is the rolling daily cumulative seasonally adjusted volume (k days back) and
YH^(l,k) is the rolling hourly cumulative volume. Weights phi_k follow the Beta weighting
scheme with shape parameters (omega_1, omega_2), constrained to sum to one. The daily
filter (slower decay) and hourly filter (faster decay) together reproduce long-memory-like
behavior. The restriction omega_{1,d} = omega_{1,h} = 1 reduces the Beta weighting to a
monotone decreasing scheme, used in practice for parsimony.

**Short-run non-periodic component (g_{t,i}):** A mean-reverting GARCH(r,s)-type process:

    g_{t,i} = omega* + alpha_1 * (y_{t,i-1} / tau_{t,i-1}) + alpha_0 * I(y_{t,i-1}=0)
              + beta_1 * g_{t,i-1} + beta_2 * g_{t,i-2}

The indicator alpha_0 handles zero-volume periods. The mean constraint E(g_{t,i})=1 is
enforced via expectation targeting: omega* is derived analytically from the other parameters
and the empirical zero probability pi.

**Innovation (epsilon_{t,i}):** Modeled as either the static Zero-Augmented Generalized F
(ZAF) distribution or the Dynamic ZAF (DZAF), where the zero probability pi_{t,i} follows
a logistic ACM(1,1) process. The ZAF nests Weibull, Generalized Gamma, and Log-Logistic
as special cases, providing a flexible fit for non-negative volumes with a point mass at zero.

Stationarity of g_{t,i} is guaranteed by the negative top Lyapunov exponent condition,
practically approximated by sum(alpha_j + beta_j) < 1. Stationarity of seasonally adjusted
volumes y_{t,i} is established via a random-coefficient AR(1) representation, following the
approach of Wang and Ghysels (2015).

## Key Parameters

| Parameter | Description | Typical Empirical Value |
|-----------|-------------|------------------------|
| Q, P | Fourier Flexible Form order (seasonal component) | Q=2, P=6 (BIC-selected) |
| alpha_1 | Short-run ARCH coefficient | ~0.29-0.36 |
| beta_1 | Short-run GARCH coefficient (lag 1) | ~0.24-0.40 |
| beta_2 | Short-run GARCH coefficient (lag 2) | ~0.11-0.31 |
| alpha_0 | Zero-volume adjustment in g_{t,i} | Significant only for DTE, CON |
| omega_2,d | Beta weight decay for daily MIDAS filter | Varies widely; omegas decrease when hourly filter added |
| omega_2,h | Beta weight decay for hourly MIDAS filter | Always > omega_2,d (faster decay) |
| K_d | Number of daily lags in MIDAS filter | 240-460 (BIC-selected per stock) |
| theta_d, theta_h | Slope coefficients for daily/hourly MIDAS filters | ~0.004-0.017 (daily), ~0.030-0.040 (hourly) |
| a, b, c | ZAF shape parameters | a~2.5-3.4, b~0.6-1.35, c~1.05-1.56 |
| pi | ZAF static zero probability | Close to empirical non-zero frequency |
| gamma_1 | ACM persistence for DZAF zero probability | <1 (stationarity condition) |

The daily MIDAS filter weights decay to 10^-2 in 7-37 trading days across stocks; the hourly
filter weights decay to 10^-2 in 4-7 two-hour-and-fifty-minute periods, confirming the
separation of time scales.

## Data Requirements
- High-frequency intra-daily trading volumes for individual stocks, tick-by-tick aggregated
  into equal intervals (10 minutes used, yielding 51 observations per 8.5-hour trading day)
- Filtered and cleaned using the Brownlees-Gallo (2006) procedure to remove erroneous ticks
- Data must cover a period long enough to estimate both daily and sub-daily MIDAS components
  (paper uses 1017 trading days: Jan 2009 - Dec 2012)
- The model handles stocks with zero-volume intervals (zero percentages ranged 0.014%-0.792%
  across the six Xetra stocks: BEI, CON, DTE, G1A, SZG, VOW)
- For out-of-sample evaluation, a rolling window of 500 days is used

## Results

**In-sample:** The H-MIDAS-CMEM achieves the lowest BIC values across all six stocks in
both ZAF and DZAF specifications. The DZAF distribution produces noticeably better
log-likelihoods and lower BIC values than ZAF in all cases. Residual diagnostics confirm
that the H-MIDAS-CMEM eliminates autocorrelation at lag 1 (unlike competing models) and
frequently captures residual dynamics at all tested lags.

**Out-of-sample (one-step-ahead, 10-minute horizon):** Relative to the MEM-ZAF benchmark:
- H-MIDAS-CMEM achieves 8.4-18.3% MSE reduction (most improvement for CON and G1A)
- H-MIDAS-CMEM achieves 4.4-7.3% MAE reduction
- H-MIDAS-CMEM enters the 75% Model Confidence Set (MCS) in almost all cases; only DTE
  shows exceptions for MAE and Slicing loss

**Multi-horizon forecasting:** The advantage of H-MIDAS-CMEM persists across all
forecasting horizons from market open (9:00) to close (17:30). The DZAF specification
offers negligible additional benefit over ZAF for out-of-sample prediction.

**Component analysis:** The daily MIDAS filter memory is substantially longer than that of
the hourly filter. Including the hourly component causes the daily filter's weights to
decay more slowly, increasing its effective memory. The g_{t,i} and tau_{t,i} components
exhibit clearly distinct autocorrelation patterns, validating the separation-of-scales design.

## Relationships

**Directly extends:**
- Brownlees, Cipollini, Gallo (2011) CMEM: replaces their daily GARCH-type long-run
  component with a multi-frequency MIDAS cascade; also extends the trend to vary
  intra-daily rather than being fixed per day

**Draws on:**
- Corsi (2009) HAR model: the additive cascade of aggregated lags at daily/weekly/monthly
  scales is the direct conceptual inspiration for the heterogeneous MIDAS structure
- Ghysels, Santa-Clara, Valkanov (2007) MIDAS framework: provides the Beta polynomial
  weighting scheme used in tau_{t,i}
- Hautsch, Malec, Schienle (2014): source of the ZAF and DZAF distributions used
  for estimation
- Muller et al. (1993) Heterogeneous Market Hypothesis: theoretical motivation for
  multi-frequency component structure
- Wang and Ghysels (2015): stochastic properties derivation strategy for the MIDAS
  long-run component
- Engle (2002) MEM: the overarching model class
- Engle and Russell (1998) ACD model: foundational irregular-spacing model

**Compared against (all outperformed):**
- Standard MEM (constant trend)
- CMEM of Brownlees et al. (2011) (daily GARCH long-run component)
- HAR-CMEM (Corsi-style daily/weekly/monthly additive structure in trend)
- MIDAS-CMEM (single-frequency daily MIDAS, a restricted version of the proposed model)

## Implementation Notes

**Two-stage estimation:** Stage one estimates seasonal Fourier coefficients by OLS. Stage
two maximizes the ZAF or DZAF log-likelihood over the remaining parameters. Standard
errors account for first-stage uncertainty via a two-stage GMM sandwich formula (Newey
and McFadden, 1994; Engle and Sokalska, 2012).

**Identification constraints:**
- Set v = (pi * xi)^-1 to enforce E(epsilon_{t,i}) = 1
- Use expectation targeting to set omega* such that E(g_{t,i}) = 1
- Constrain omega_{1,d} = omega_{1,h} = 1 (monotone decreasing Beta weights) for parsimony
- Constrain theta_d > 0, theta_h > 0, m > 0 to guarantee positivity of tau_{t,i}
- Sum(alpha_j + beta_j) < 1 guarantees stationarity of g_{t,i}

**Grid search for K_d:** BIC minimization over K_d in {200, 220, ..., 500} and GARCH
orders r, s in {1, 2} is computationally tractable and needed; optimal K_d varies widely
across stocks (240-460 days).

**Forecasting strategy:** Parameters re-estimated daily on a 500-day rolling window.
One-step-ahead predictions use conditional mean: x-hat_{t,i} = tau_{t,i} * g_{t,i} * phi_i.
Multi-step predictions iterate the conditional expectation formula for g_{t,i+h}.

**VWAP slicing application:** The Slicing loss function (Brownlees et al., 2011) provides
an economically meaningful evaluation criterion by measuring the cross-entropy between
actual and predicted intra-daily volume proportions, directly linked to VWAP replication
error. Both static (using only prior-day information) and dynamic (updated within the day)
slicing strategies are evaluated.

**Open questions:** The stationarity conditions for DZAF-error models (where errors are
independently but not identically distributed) remain an open problem; the paper notes this
explicitly and leaves a formal extension to future work.

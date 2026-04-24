# Forecasting Intraday Trading Volume: A Kalman Filter Approach
**Authors:** Ran Chen, Yiyong Feng, Daniel P. Palomar
**Year:** 2016
**DOI:** N/A (working paper / preprint from HKUST)

## Core Contribution

This paper introduces a state-space model for forecasting intraday trading volume using
the Kalman filter, with closed-form expectation-maximization (EM) solutions for model
calibration. The approach decomposes log-volume into three additive components — a daily
average component, an intraday periodic (seasonal) component, and an intraday dynamic
component — and models their evolution using a linear Gaussian state-space representation.
The logarithmic transformation of raw volume is central to the method: it converts the
multiplicative, positiveness-constrained CMEM-style model into a tractable linear additive
one, and it also reduces the right-skewness of the distribution, making Gaussian noise
assumptions more defensible.

The paper also proposes a robust extension that handles outliers in real-time (non-curated)
market data by augmenting the observation equation with a sparse noise term and applying
Lasso (L1) regularization in the Kalman correction step. Empirical evaluation covers 30
securities (12 ETFs and 18 stocks) across U.S., European, and Asian exchanges over
January 2013 to June 2016, demonstrating substantial improvements over both the rolling
means (RM) baseline and the state-of-the-art Component Multiplicative Error Model (CMEM).

## Model/Algorithm Description

The model operates on 15-minute intraday bins. Raw volume is normalized by daily
outstanding shares, and the natural log is taken. The log-volume observation y_{t,i} for
day t and bin i is decomposed as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where eta_t is the (log) daily average component, phi_i is the (log) intraday periodic
component (the U-shaped intraday seasonal pattern), mu_{t,i} is the (log) intraday dynamic
component, and v_{t,i} is Gaussian observation noise.

This is cast into a standard state-space form with hidden state x_tau = [eta_tau, mu_tau]^T:

    x_{tau+1} = A_tau * x_tau + w_tau     (state transition)
    y_tau     = C * x_tau + phi_tau + v_tau   (observation)

The state transition matrix A_tau is time-varying: at day boundaries (tau = kI) the daily
component eta transitions with coefficient a^eta and receives process noise; within a day
a^eta = 1 and the noise variance for eta is zero (i.e., eta is piecewise constant within
a day). The intraday dynamic component mu always transitions with scalar coefficient a^mu.

Three algorithms work together:

1. Kalman Filter (Algorithm 1): Recursive predict-correct loop producing one-bin-ahead
   or multi-bin-ahead forecasts. The "dynamic" prediction mode updates estimates after
   each observed bin; the "static" prediction mode skips correction steps and forecasts
   all bins of the next day using information up to the prior day's close.

2. Kalman Smoother (Algorithm 2): Backward-pass smoother (Rauch-Tung-Striebel form)
   used during calibration to obtain full-data posterior estimates of the hidden states.

3. EM Algorithm (Algorithm 3): Iteratively estimates all model parameters by alternating
   between an E-step (running forward filter + backward smoother to compute sufficient
   statistics x_hat_tau, P_tau, P_{tau,tau-1}) and an M-step (closed-form updates for
   all parameters, including the seasonality vector phi). Convergence is rapid and robust
   to initial parameter choice.

The robust variant (Section 3) adds a sparse noise term z_tau to the observation equation.
The modified Kalman correction step solves a Lasso-penalized quadratic problem at each
bin, which has a closed-form soft-thresholding solution. The threshold adapts dynamically
via the innovation variance W_tau = (C * Sigma_{tau+1|tau} * C^T + r)^{-1}. Parameters
r and phi are re-estimated in the EM M-step using the inferred z_tau* values.

VWAP execution is implemented in two strategies:
- Static VWAP: slice weights determined from static volume forecasts before market open.
- Dynamic VWAP: weights revised at each bin using one-bin-ahead dynamic forecasts,
  redistributing remaining execution proportionally.

## Key Parameters

| Parameter       | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| a^eta           | Day-to-day AR coefficient for the daily average component            |
| a^mu            | Bin-to-bin AR coefficient for the intraday dynamic component         |
| (sigma^eta)^2   | Process noise variance for the daily component (at day transitions)  |
| (sigma^mu)^2    | Process noise variance for the intraday dynamic component            |
| r               | Observation noise variance                                           |
| phi = [phi_1..phi_I] | Intraday seasonality vector (I values, one per bin per day)     |
| pi_1, Sigma_1   | Initial state mean and covariance                                    |
| lambda          | Lasso regularization parameter (robust model only)                   |
| N               | Training window length (number of bins)                              |

All parameters except lambda are estimated by EM with closed-form M-step updates.
Optimal N and lambda are selected by cross-validation on a held-out validation period
(January 2015 to May 2015 in the empirical study). No recommended numerical values
for parameters are given; they are data-driven.

## Data Requirements

- Intraday volume data at 15-minute granularity (bin level).
- Volume is normalized by daily outstanding shares (to correct for splits and scale
  changes). Bins with zero volume are excluded (the model assumes non-zero volumes).
- The empirical study uses Bloomberg data for 30 securities across NYSE, NASDAQ,
  NYSEArca, EPA (France), LON (UK), ETR (Germany), Amsterdam, TYO (Japan), and HKEX
  (Hong Kong), covering January 2013 to June 2016 (excluding half-day sessions).
- The number of bins per day I varies by exchange (depending on local trading hours).
- Out-of-sample evaluation: June 2015 to June 2016 (D=250 days).
- Cross-validation set: January 2015 to May 2015.
- In addition to volume, the last transaction price in each bin is used for VWAP
  replication evaluation.

## Results

Volume prediction (MAPE, out-of-sample, dynamic prediction mode):
- Robust Kalman Filter average MAPE: 0.46 across 30 securities.
- Standard Kalman Filter average MAPE: 0.47.
- CMEM average MAPE: 0.65 (dynamic mode).
- RM baseline average MAPE: 1.28.

This yields a 64% improvement over RM and 29% improvement over dynamic CMEM for the
robust Kalman filter.

For static prediction:
- Robust Kalman Filter average MAPE: 0.61.
- Standard Kalman Filter: 0.62.
- CMEM: 0.90.
- RM: 1.28.

VWAP tracking error (basis points, out-of-sample, dynamic VWAP strategy):
- Robust Kalman Filter average: 6.38 bps.
- Standard Kalman Filter: 6.39 bps.
- CMEM: 7.01 bps.
- RM: 7.48 bps.

This is a 15% improvement over RM and 9% improvement over CMEM.

Robustness simulation (artificially contaminated data, 10% of bins perturbed with
outliers at small/medium/large scales): the robust Kalman filter degrades much more
slowly than the standard Kalman filter, CMEM (which fails entirely at medium/large
outlier levels), or RM.

EM convergence (synthetic experiments): parameters converge within a few iterations and
are insensitive to initial values, unlike CMEM which is brittle to initialization.

## Relationships

This paper builds directly on Brownlees, Cipollini, and Gallo (2011), adopting their
three-component volume decomposition idea (daily, periodic, dynamic) but reformulating
it in log space as a linear state-space model rather than a multiplicative error model.
The key departure is the use of the Kalman filter and EM in place of CMEM's nonlinear
estimation, which avoids positiveness constraints, numerical instability, and
initialization sensitivity.

Satish et al. (2014) used a similar decomposition with ARMA models; this paper claims
to outperform that approach as well. Rolling means (RM) serve as the simplest baseline.

The robust extension draws on Mattingley and Boyd (2010) for the efficient Lasso-in-
Kalman formulation and on Tibshirani (1996) for the Lasso itself. The EM framework
follows Shumway and Stoffer (1982).

The paper does not directly engage with microstructure or order flow models (e.g.,
PIN-based volume models), nor with machine learning approaches to volume forecasting.
It positions itself squarely within the time-series / signal-processing tradition applied
to execution-quality problems.

## Implementation Notes

- The state dimension is 2 (eta and mu), keeping the Kalman filter computationally
  lightweight. The observation is scalar, so all matrix inversions reduce to scalar
  divisions (W_tau = scalar, Kalman gain = 2x1 vector).
- The seasonality phi is estimated as the average residual y_{t,i} - C*x_hat_{t,i}
  over all T training days for each bin i, which is a simple mean across days.
- Within a trading day, eta is held constant (A_tau diagonal entry = 1, noise = 0),
  so the Kalman filter effectively only propagates mu during intraday steps and updates
  eta at each day boundary.
- The Lasso threshold lambda/(2*W_tau) is time-varying because W_tau depends on the
  current predictive variance. This means the effective outlier rejection width adapts
  to the model's current uncertainty.
- Cross-validation is needed to select N (training window length) and lambda. The paper
  does not report typical values of N; practitioners should expect this to vary by
  asset and market.
- The CMEM is noted to fail (unable to find a solution) under medium/large outlier
  conditions, which represents a practical reliability risk for production systems.
- Code or data are not publicly released in this paper.
- An open question is how to handle zero-volume bins (explicitly excluded by assumption).
  Extending the model to handle intermittent or missing observations would require
  modifications to the standard Kalman equations.
- Future work identified in the paper includes applying the model to ultra-high-frequency
  data (finer time grids) and incorporating additional covariates (e.g., volatility,
  spread) as exogenous inputs to the state-space model.

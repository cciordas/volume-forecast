# Forecasting Intraday Volume: Comparison of Two Early Models
**Authors:** Szűcs, Balázs Árpád
**Year:** 2017 (accepted 2016)
**DOI:** 10.1016/j.frl.2016.11.018

## Core Contribution
This paper provides a direct, controlled comparison of two influential intraday volume forecasting
models — Bialkowski, Darolles and Le Fol (2008) (BDF) and Brownlees, Cipollini and Gallo (2011)
(BCG) — evaluated on the same dataset using identical error measures. Prior to this work, the two
models had only been tested on different samples with different metrics, making cross-model
comparison impossible.

The paper finds that both models outperform the naive U-method benchmark, and that the BDF model
is the clear winner: it produces more accurate forecasts under both MSE and MAPE criteria across
the full 11-year sample and across multiple subperiods. Crucially, the BDF model is also faster
to estimate by several orders of magnitude, making it far more practical for real-time use.

## Models Compared

### Benchmark: U-method
A simple average of the same intraday bin across the prior L days:

    y_hat_{p+1} = (1/L) * sum_{l=1}^{L} y_{p+1 - m*l}

where m = 26 (number of 15-minute bins per trading day) and L = 20 (days in estimation window).
This formula encodes the well-known intraday U-shape of volume.

### BDF Model (Bialkowski, Darolles, Le Fol 2008)
Uses an additive decomposition: X = K + e, where X is a (P x N) matrix of turnovers.

- Common component K: extracted using large-dimensional factor analysis (Bai 2003), then forecast
  via the U-method.
- Specific component e: forecast using either an AR(1) model (BDF_AR) or a SETAR model
  (BDF_SETAR). The SETAR variant applies a threshold indicator I(z) to switch between two linear
  regimes.

This is a cross-sectional + time series model: it uses all shares jointly to estimate the common
U-shape, then models each share's idiosyncratic deviation individually.

### BCG Model (Brownlees, Cipollini, Gallo 2011)
Uses a multiplicative decomposition: x_{ti} = eta_t * phi_i * mu_{ti} * eps_{ti}, where:

- eta_t: daily component, modeled as an ARMA-like recursion in daily turnover.
- phi_i: intraday periodic component (the U-shape), modeled via a Fourier series with K harmonics.
- mu_{ti}: intraday non-periodic component, modeled as an intraday ARMA recursion.
- eps_{ti}: i.i.d. innovation with mean 1.

The full variant (BCG_3, used here) adds: an opening-bin dummy in the mu equation, asymmetric
return effects in both eta and mu equations, and a second lag in the mu equation. Parameters are
estimated by GMM. The paper tests four estimation strategies (BCG_0 through BCG_3) to address
severe numerical difficulties; BCG_3 (grid search with fallback to prior-day parameters) gives
the best results.

## Key Parameters

| Parameter | Description | Value Used |
|---|---|---|
| Estimation window (L) | Days used to fit model parameters | 20 days |
| Intraday bins (m) | 15-minute intervals per trading day | 26 |
| Forecast horizon | Days ahead to forecast | 1 day |
| Fourier harmonics (K) | Terms in BCG periodic component | 4 (reduced from max 25) |
| BCG total parameters | After K=4 reduction | 13 (vs. 34 at full K) |

Parameters are re-estimated daily on a rolling 20-day window. Within each day, one-step-ahead
forecasts are updated every 15 minutes as actuals arrive.

## Data Requirements
- Universe: 33 DJIA constituent stocks (30 NYSE, 3 NASDAQ); 3 tickers excluded for short history.
- Period: October 10, 2001 to July 13, 2012 (~11 years, ~2,648 trading days).
- Frequency: 1-minute data aggregated to 15-minute bins, yielding 26 observations per day.
- Volume representation: turnover = volume / total shares outstanding (TSO), requiring TSO data
  from an external source (Bloomberg in this study).
- Total observations: ~2.29 million.
- Liquidity requirement: every stock must have non-zero volume in every 15-minute interval.

## Results

**Full sample (33 shares):**

| Model     | MSE      | MSE*     | MAPE  |
|-----------|----------|----------|-------|
| U-method  | 1.02E-03 | 3.65E-05 | 0.503 |
| BDF_AR    | 6.49E-04 | 2.30E-05 | 0.403 |
| BDF_SETAR | 6.60E-04 | 2.38E-05 | 0.399 |
| BCG_3     | 6.77E-04 | 2.53E-05 | 0.402 |

- By MSE: BDF_AR is best overall and beats all other models on 31/33 individual shares.
- By MAPE: BDF_SETAR is best overall and beats all other models on 26–30/33 individual shares.
- BCG_3 clearly beats the U-method (28/33 shares by MSE; 33/33 by MAPE) but trails both BDF
  variants.

**Robustness across 5 equal subperiods:** BDF dominance holds in 4 of 5 subperiods. In the first
subperiod only, BCG_3 wins on MAPE.

**Extreme errors (Q95):** BDF_AR produces the fewest extreme squared errors; BDF_SETAR produces
the fewest extreme absolute percentage errors across most samples.

**Estimation time:** BDF model runs in ~2 hours. BCG_0 (fully re-estimated daily with grid search)
took ~60 machine-days. BCG_3's fallback strategy reduces this but remains far slower than BDF.

## Relationships
- Directly compares Bialkowski et al. (2008) and Brownlees et al. (2011), which had previously
  only been assessed on non-overlapping samples.
- BDF model builds on Bai (2003) large-dimensional factor analysis for extracting the common
  intraday volume component.
- Both models extend the naive U-method (a practitioner benchmark) by separately modeling the
  common periodic shape and the idiosyncratic or non-periodic residual.
- Related work not directly compared: Humphery-Jenner (2011) on optimal VWAP under noise,
  Manchaladore (2010) on wavelet decomposition for intraday volume, and Satish et al. (2014) on
  predicting intraday volume percentages.
- Both models are motivated by VWAP execution strategies (Madhavan 2002, Kissell et al. 2004),
  where an accurate intraday volume forecast directly reduces market impact costs.

## Implementation Notes
- The BCG model has significant numerical fragility: the GMM objective is not smooth enough for
  reliable daily optimization. The paper documents four estimation variants and recommends BCG_3
  (grid search + fallback) as the most stable, but even this required 60 machine-days for the
  full sample under naive implementation.
- The K=4 Fourier harmonics choice for the BCG periodic component is a practical compromise; the
  paper does not rigorously justify this choice beyond noting it reduces parameter count from 34
  to 13 and should suffice for 26-point U-shape approximation.
- BCG estimation requires careful initialization: initial values for return-related asymmetry
  terms are set to zero, second-lag initial value is set equal to the first lag — these are
  assumptions not specified in the original paper.
- For the BDF model, factor analysis (Bai 2003) on a (P x N) turnover matrix is the main
  computational step; this is well-specified and fast.
- One-step-ahead forecasting within the day (updating every 15 minutes as actuals arrive) is
  straightforward for both models since only the intraday non-periodic component needs updating
  intraday; daily and periodic parameters are fixed until the next day's re-estimation.
- The BCG model was originally validated on ETFs; applying it to individual stocks (as done here)
  is a generalization, though the authors note intraday volume patterns are similar across both
  asset classes.
- Future work suggested: comparison using additional error measures, and inclusion of more recent
  models (Humphery-Jenner 2011, Satish et al. 2014, Manchaladore 2010).

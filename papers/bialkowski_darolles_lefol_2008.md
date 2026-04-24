# Improving VWAP Strategies: A Dynamic Volume Approach
**Authors:** Jedrzej Bialkowski, Serge Darolles, Gaelle Le Fol
**Year:** 2008
**DOI:** 10.1016/j.jbankfin.2007.09.013 (Journal of Banking & Finance 32, 1709–1722)

## Core Contribution

This paper introduces a two-component factor model for intraday volume that decomposes
traded turnover into a market-wide common component and a stock-specific idiosyncratic
component. The common component captures the well-known U-shaped seasonal pattern shared
across all stocks, while the specific component captures stock-level deviations from that
pattern and is modelled dynamically with ARMA or SETAR time-series models. The
decomposition is estimated by principal components analysis (PCA) on a rolling window of
CAC40 constituents.

The practical motivation is VWAP (Volume Weighted Average Price) order execution. The
paper demonstrates that a trader who accurately predicts the intraday volume profile can
replicate the end-of-day VWAP regardless of price movements, as long as their order size
is small relative to market volume. The proposed dynamic VWAP execution algorithm updates
volume forecasts intraday as new information arrives, reducing tracking error by more than
10% on average versus the classical historical-average benchmark, with reductions
exceeding 50% for some stocks.

## Model/Algorithm Description

**Volume decomposition.** For stock i at intraday interval t, turnover x_{i,t} (shares
traded divided by float) is expressed as:

    x_{i,t} = lambda_i' * F_t + e_{i,t} = c_{i,t} + e_{i,t}

where F_t is an r-dimensional vector of latent common factors, lambda_i are factor
loadings, c_{i,t} is the common (market) component, and e_{i,t} is the specific
(idiosyncratic) component. The factor dimension r is selected using the Bai & Ng (2002)
information criterion. Estimation follows Bai (2003) by principal components, which
remains consistent under large N and large T even with serially correlated errors.

**Common component forecast.** The common component is treated as a stable seasonal
shape. Its forecast for the next day is the cross-day historical average over the
preceding L days:

    c_hat_{i,t+1} = (1/L) * sum_{l=1}^{L} c_{i,t+1 - k*l}

where k = 25 (the number of 20-minute intervals per trading day) so that averaging is
always done at the same time-of-day across L previous days.

**Specific component dynamics.** Two time-series models are fitted to e_{i,t}:

- ARMA(1,1): e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}
- SETAR (Self-Exciting Threshold AR): a two-regime model
      e_{i,t} = (phi_{11}*e_{i,t-1} + phi_{12}) * I(e_{i,t-1} <= tau)
              + (phi_{21}*e_{i,t-1} + phi_{22}) * (1 - I(e_{i,t-1} <= tau))
              + epsilon_{i,t}
  where the threshold tau discriminates between calm and turbulent volume regimes.

Estimation is two-step: (1) extract e_{i,t} via PCA factor analysis, (2) fit ARMA or
SETAR to the residuals by maximum likelihood.

**VWAP execution strategies.** Three implementations are considered:

1. Theoretical execution: uses one-step-ahead predictions for each interval, requires
   end-of-day total volume (not implementable in real time; serves as an upper bound).
2. Static execution: predicts all 25 intervals at the start of the day; performs poorly
   because long-horizon ARMA forecasts collapse to zero and the dynamic component
   contributes nothing.
3. Dynamic execution: starts with a full-day forecast at open, then re-forecasts the
   remaining intervals after each observed interval using updated information. This is
   the operationally viable strategy.

## Key Parameters

| Parameter | Meaning                                      | Value used        |
|-----------|----------------------------------------------|-------------------|
| r         | Number of common factors (estimated by Bai-Ng IC) | Estimated from data |
| k         | Intervals per trading day                    | 25 (20-min bars)  |
| L         | Rolling window length for common-component average | 20 trading days |
| delta_t   | Interval length                              | 20 minutes        |
| tau       | SETAR threshold on e_{i,t-1}                 | Estimated per stock |
| h         | Historical window for PCA estimation         | 1 month (approx. 500 intraday obs.) |

The trading day covers 9:20–17:20 (continuous auction only; pre-opening trades excluded).

## Data Requirements

- Tick-by-tick trade volume and prices for all CAC40 constituents (39 stocks) from
  EURONEXT historical database.
- Sample period: September 2003 to August 2004 (approximately 250 trading days).
- Data aggregated into 20-minute bars; volume measured as turnover (traded shares /
  float shares), adjusted for splits and dividends.
- End-of-day VWAP computed from daily transaction records for each company.
- The 24th and 31st of December 2003 excluded as partial trading days.
- In-sample estimation window: September 2 to December 16, 2003 (75 days).
- Out-of-sample evaluation: rolling 20-day estimation window, 50 prediction days.

## Results

**Volume prediction accuracy (MAPE across CAC40 portfolio):**

| Model            | Mean MAPE | Std    | Q95    |
|------------------|-----------|--------|--------|
| PCA-SETAR        | 0.0752    | 0.0869 | 0.2010 |
| PCA-ARMA         | 0.0829    | 0.0973 | 0.2330 |
| Classical average| 0.0905    | 0.1050 | 0.2490 |

**VWAP execution cost (MAPE, in-sample):**

| Model              | Mean   | Std    | Q95    |
|--------------------|--------|--------|--------|
| PC-SETAR           | 0.0706 | 0.0825 | 0.2030 |
| PC-ARMA            | 0.0772 | 0.0877 | 0.2173 |
| Classical approach | 0.1140 | 0.1358 | 0.3702 |

**VWAP execution cost (MAPE, out-of-sample):**

| Model                          | Mean   | Std    | Q95    |
|--------------------------------|--------|--------|--------|
| PC-SETAR theoretical           | 0.0770 | 0.0942 | 0.2432 |
| PC-ARMA theoretical            | 0.0833 | 0.0956 | 0.2498 |
| PC-SETAR dynamic               | 0.0898 | 0.0954 | 0.2854 |
| PC-ARMA dynamic                | 0.0922 | 0.0994 | 0.2854 |
| Classical approach             | 0.1006 | 0.1171 | 0.3427 |

Key findings:
- SETAR outperforms ARMA in nearly all stocks; ARMA beats SETAR for only 3 out of 39
  stocks (and the margin is negligible).
- Dynamic PCA-SETAR reduces out-of-sample tracking error by roughly 10% on average
  versus classical, and by up to 50% for high-volatility stocks (e.g., CAP GEMINI,
  EADS). Improvements are largest precisely where the classical approach fails most.
- The gap between theoretical and dynamic execution represents the value of intraday
  information: on average approximately 1 bp, but reaching 4–8 bp for volatile stocks.
- The static execution strategy is confirmed to be worse than classical, because
  multi-step ARMA forecasts decay to zero and the dynamic component is neutralised.

## Relationships

- **Darolles & Le Fol (2003)** is the direct predecessor: that working paper introduced
  volume decomposition into common and specific parts for arbitrage detection. The
  present paper extends it to high-frequency intraday data and operationalises the
  decomposition for VWAP execution.
- **Lo & Wang (2000)** proposed the original factor model for trading volume (one-factor
  CAPM analogy). This paper generalises Lo & Wang by allowing r > 1 factors and using
  Bai (2003) consistent estimation.
- **McCulloch (2004)** used cross-stock average volume as the seasonal baseline. The PCA
  decomposition here can be viewed as an extension that extracts multiple factors rather
  than a single market average.
- **Konishi (2002)** proposed an optimal static slice for VWAP trades. This paper's
  dynamic approach is positioned as an improvement over such static methods.
- **Hobson (2006)** concluded that volume-profile refinements yield no material VWAP
  benefit; this paper directly contradicts that finding with empirical evidence of
  significant error reduction.
- **Bai (2003) / Bai & Ng (2002)** provide the statistical foundations for large-N,
  large-T factor estimation and factor-count selection used in the PCA step.
- Related work on intraday volume seasonality includes Biais, Hillion & Spatt (1995)
  and Gourieroux, Jasiak & Le Fol (1999) for the French market.

## Implementation Notes

- The PCA is re-estimated on a rolling 1-month window (approximately 500 observations
  for 39 stocks x 25 intervals). Factor loadings are thus time-varying but refreshed
  only daily.
- The common component forecast is computed the night before (using the previous day's
  estimated loadings), so no look-ahead bias is introduced.
- The specific component is forecast on a 20-minute delay during the day; the ARMA/SETAR
  parameters are estimated on residuals from the previous 20-day window.
- Static VWAP execution is explicitly shown to be dominated by both the classical
  approach and dynamic execution; it should not be used in practice.
- The SETAR threshold tau is estimated stock by stock; the paper does not provide a
  universal recommended value.
- An important open limitation: to "beat" the VWAP (rather than merely track it), a
  bivariate volume-price model is required. The current model is purely volume-driven
  and agnostic to price dynamics.
- The model assumes the trader's order size is small relative to market volume (no
  significant price impact). For large orders, a multi-day VWAP strategy or
  participation-limit extension would be needed.
- Code or software is not provided; the paper describes a two-step ML estimation
  procedure that is straightforward to implement with standard PCA and ARMA/SETAR
  routines.

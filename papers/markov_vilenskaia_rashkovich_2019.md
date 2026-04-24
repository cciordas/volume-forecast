# Quintet Volume Projection
**Authors:** Vladimir Markov, Olga Vilenskaia, Vlad Rashkovich
**Year:** 2019
**DOI:** N/A (Bloomberg L.P. working paper / Journal of Trading submission)

## Core Contribution

This paper presents the "Quintet" — an ensemble of five coordinated sub-models for intra-day equity volume prediction. Rather than building a single complex model, the authors decompose the prediction problem into interpretable components (total daily volume prior, intra-day u-curve, close auction volume, and two Bayesian update models for liquid and illiquid securities), then tie them together so their outputs remain mutually consistent.

The paper also introduces the Asymmetrical Logarithmic Error (ALE) metric as a replacement for symmetric metrics such as RMSE and MAPE. ALE penalizes overestimation of volume at twice the rate of underestimation, reflecting the asymmetric execution risk: overestimating volume drives up participation rate, causing excess market impact.

## Model/Algorithm Description

The quintet consists of five sub-models, applied sequentially and consistently:

**Model 1 — Historical Daily Volume Prior (Volume Prior)**
The prior for total daily volume is the 20-day geometric mean (GM) of log-daily volume, which equals the median of the log-normal distribution and is robust to outlier days. The prior is then adjusted by two add-ons:
- An ARMA(1,1) component fit on de-meaned log-volume residuals (y_t = X_t - mu_t), capturing serial autocorrelation (AR) and response to volume shocks (MA). Fitted coefficients are near-universal across S&P 500 names: phi ~0.7, theta ~-0.3.
- A special-day adjustment via linear regression on dummy variables (earnings, option expirations, index rebalancing, overnight price gaps), calibrated with ALE, yielding a multiplier applied to the geometric average prior.

**Model 2 — Intra-day Volume Profile (U-curve)**
The u-curve u(t) is the fraction of total daily volume traded in each intra-day bin; the c-curve c(t) is its cumulative sum. A plain estimator uses the 180-day historical average. This is enhanced by Functional Data Analysis (FDA) functional regression, modeling c(t) as a function of scalar predictors such as overnight price gap (normalized by 20-day volatility) and total daily volume percentile. Higher overnight gaps or higher volume days shift the u-curve from a U-shape toward an inverted J-shape (more front-loaded volume).

**Model 3 — Close Auction Volume**
Base prediction uses the 20-day geometric average of closing auction volume. A seasonal adjustment multiplier is estimated by linear regression on option/futures expiration dummies (triple witching days in the U.S., quarterly expirations). An ARMA component offers marginal improvement (within 5%) and is generally omitted for simplicity. Real-time auction imbalance data, when available within ~30 minutes of close, can be incorporated via Bayesian updating.

**Model 4 — Intra-day Bin Model (Liquid Securities)**
For liquid stocks, per-bin volume observations x(j) = log(v(j) / u_hat(j)) provide intra-day evidence about total daily volume. Bayesian updating (conjugate normal-gamma prior) blends the prior mu_0 with the growing set of intra-day observations. Early in the day (unknown variance), the unknown-mean-and-variance form (Student-t posterior) is used with an effective prior sample size kappa_0. Once sufficient observations exist to estimate variance, the known-variance form (Gaussian posterior) is used:
  mu(n) = [ (n * x_bar / Sigma^2) + (mu_0 / sigma_0^2) ] / [ n/Sigma^2 + 1/sigma_0^2 ]

**Model 5 — Historical Cumulative Model (Illiquid Securities)**
For illiquid stocks with sparse or erratic bin volumes, the cumulative intra-day volume V(i) is used instead: z(i) = log(V(i) / c_hat(i)). Bayesian updating employs only a single observation z(n) per day, weighted by its historical dispersion Omega^2(n) estimated over the last M days. This is more stable than bin-by-bin inference when many bins have zero volume.

**Final Prediction**
Remaining daily volume = exp(mu(t)) * [1 - c_hat(t)]. Volume over any interval [t1, t2] is estimated using the updated daily volume estimate and the c-curve. Practical outputs include expected participation rate for POV algorithms and estimated order completion time given a target participation rate.

## Key Parameters

| Parameter | Description | Recommended Value |
|---|---|---|
| N | Look-back window for geometric average prior and ARMA de-meaning | 20 days |
| phi | AR(1) coefficient in ARMA(1,1) on log-volume residuals | ~0.7 (S&P 500) |
| theta | MA(1) coefficient in ARMA(1,1) on log-volume residuals | ~-0.3 (S&P 500) |
| kappa_0 | Effective prior sample size for Bayesian update | 0.3–0.8 * N_prior; e.g. 0.5 * 20 = 10 for liquid U.S. names in 10-min bins |
| N_prior | Number of historical daily volume observations used as prior | 20 |
| Bin size | Intra-day bin width | 10 minutes (for U.S. equities) |
| U-curve history | Look-back for historical average u-curve | 180 days |
| M | Days used to estimate dispersion Omega^2(n) in illiquid model | Not specified; historical window |
| ALE weights | Overestimation penalty multiplier | 2x (underestimation = 1x) |

## Data Requirements

- Daily closing equity volumes (minimum 20 days for prior, 180 days for u-curve)
- Intra-day bin volume time series (10-minute resolution demonstrated)
- Overnight price gaps (open vs. prior close, normalized by 20-day realized volatility)
- Closing auction volume history (separate from continuous session volume)
- Event calendars: earnings announcement dates, option/futures expiration dates, index rebalancing dates
- Test sample: representative draws from S&P 500 (50 names), S&P Midcap 400 (100 names), Russell 2000 (100 names); July 2015 through December 2016

## Results

- The 20-day geometric mean outperforms the arithmetic mean under ALE because it matches the median of the log-normal distribution and is not inflated by fat-tail outlier days.
- ARMA(1,1) improves ALE over the plain geometric average for total daily volume; higher-order ARIMA models yield no meaningful additional improvement.
- Functional regression on the u-curve confirms that high overnight-gap days and high-volume days both shift the intra-day profile from U-shaped to inverted J-shaped (volume concentrated at open).
- Close auction volume prediction improves marginally with ARMA (within 5%) but the gain does not justify added complexity; the geometric average with seasonal dummy adjustments is recommended.
- Bayesian inference automatically balances prior reliability against intra-day signal strength, adapting to both liquid (bin-based) and illiquid (cumulative-based) regimes without manual parameter tuning per stock.
- ALE is preferred over RMSE (sensitive to fat tails, symmetric), MAPE (symmetric), and R^2 (only valid in linear regression context).

## Relationships

- Directly extends Brownlees, Cipollini & Gallo (2011) — the paper in this collection that introduced a dynamic intra-day volume model with periodic and dynamic components. The Quintet adopts a similar decomposition but replaces the GAS/MEM framework with Bayesian inference and adds close-auction and special-day sub-models.
- Closely related to Calvori, Cipollini & Gallo (2014) — also in this collection — which uses a GAS model for intra-day volume shares under a Dirichlet distribution. The Quintet uses log-normal/conjugate-prior Bayesian methods instead of score-driven updating, and explicitly models illiquid securities separately.
- Extends Satish, Saxena & Palmer (2014) — also in this collection — which introduced a three-component weighted model with per-bin ARMA. The Quintet's bin model (Model 4) is conceptually similar but embedded within a fully Bayesian framework.
- Relates to Bialkowski, Darolles & Le Fol (2008) — in this collection — which uses an autoregressive model for normalized volume including time-of-day shape and day-of-week factors. The Quintet generalizes these seasonality adjustments to event-based dummies and functional regression.
- References Markov, Mazur & Saltz (2011) for the use of volume prediction within schedule-based trading algorithms, establishing the practical motivation for the model.
- The close auction section builds on Stone, Kingsley & Kan (2015) "12% rule" for auction allocation strategy.

## Implementation Notes

- The quintet design encourages modular implementation: each of the five sub-models can be built and validated independently, then composed at prediction time.
- The switch between Model 4 (liquid) and Model 5 (illiquid) should be triggered by a stability criterion for the u-curve, e.g. fraction of zero-volume bins or variance of the per-bin u-curve estimates over recent history.
- For the Bayesian bin model (Model 4), the transition from unknown-variance (Student-t, formula 6/20) to known-variance (Gaussian, formula 4/21) should be triggered when n is large enough to estimate Sigma^2 reliably; the paper does not specify an exact threshold.
- kappa_0 is described as bin-size-, market-cap- and country-dependent; for 10-minute bins on liquid U.S. names, kappa_0 = 0.5 * N_prior = 10 is recommended.
- The Grubbs filter (Grubbs 1950) is used to remove outlier log-volume observations before fitting the log-normal; this should be applied to both daily and bin-level series.
- ALE (Equation 7) is an asymmetric L1 norm in log space: errors in log-volume are squared (like L2) but weighted 2x for overestimation. Note this differs from typical asymmetric quantile loss — it weights by the sign of the error, not a fixed quantile.
- The paper uses cross-sectional functional regression for u-curve adjustments aggregated across the index; in production, per-symbol functional regression coefficients are recommended as they are noted to be stock-dependent.
- Open question: the paper does not specify how to handle days with no prior intra-day observations (pure prior day), nor does it detail how the close auction Bayesian update on real-time imbalance data integrates with Models 3–5.

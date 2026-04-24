# Relative Volume as a Doubly Stochastic Binomial Point Process
**Authors:** James McCulloch
**Year:** 2007
**DOI:** 10.1080/14697680600969735

## Core Contribution

This paper introduces the doubly stochastic binomial point process as a formal model
for relative intra-day cumulative volume — that is, the fraction of total daily trades
executed by time t within the trading day. The key insight is that if intra-day trade
arrivals follow a Cox (doubly stochastic Poisson) process, then dividing the intra-day
cumulative count by the final daily trade count produces a process whose conditional
distribution is binomial, directed by the self-normalized integrated intensity
Lambda(t)/Lambda(T) of the underlying Cox process.

A second contribution is a practical statistical framework: rescaling volume to the
unit interval [0, 1] allows empirical data from stocks with vastly different daily trade
counts to be pooled in a single 2-D histogram. From this histogram, moments of the
self-normalized integrated intensity are estimated and used to discriminate between
competing structural models of intra-day trading intensity. The paper demonstrates on
NYSE data that stock-specific intensities are best modeled via time-scaling by trade
count rather than amplitude-scaling, providing indirect confirmation of Ane and Geman
(2000) — that trade count serves as a stochastic market clock.

## Model/Algorithm Description

Trade arrivals are modeled as a Cox process N(t) with stochastic intensity lambda(t)
and integrated intensity Lambda(t) = integral_0^t lambda(s) ds. The central object is
the random relative counting measure:

    R(t; K) = N(t) / N(T),    N(T) = K

where T is end of day and K is the final trade count. Using initial enlargement of
filtration (the natural filtration of the Cox process enlarged by knowledge of the
terminal count N(T) = K), the paper proves:

    P(R(t;K) = a | Lambda(T)) = C(K, aK) * [Lambda(t)/Lambda(T)]^{aK}
                                            * [1 - Lambda(t)/Lambda(T)]^{K-aK}

This is a binomial distribution parameterized by the self-normalized integrated
intensity p(t) = Lambda(t)/Lambda(T), which plays the role of the "success probability"
for each of the K trades being executed by time t.

The unconditional distribution of R(t; K) is then a binomial mixture:

    P(R(t;K) = a) = integral_0^1 Binomial(aK; K, s) dPhi_t(s)

where Phi_t(s) = P(Lambda(t)/Lambda(T) <= s) is the marginal CDF of the
self-normalized integrated intensity — the mixing distribution. From this mixture
representation, the moments of the mixing distribution (lambda_1, lambda_2, ...) can
be recovered from the observed moments of the relative trade count distribution using
closed-form recursions. The framework is extended to handle pooled data across stocks
with different trade counts K by introducing precomputed scaling constants
V_n = sum_K z_K / K^n, where z_K is the empirical trade-count distribution.

Two structural models for the integrated intensity of stock i relative to a baseline
Lambda_0 are considered and tested:
- Model (i): Lambda_i(t) = alpha_i * Lambda_0(t)  (amplitude scaling)
- Model (ii): Lambda_i(t) = Lambda_0(alpha_i * t)  (time scaling)

Model (i) predicts that the variance of Lambda(t)/Lambda(T) is the same across all
trade counts; this is empirically rejected. Model (ii) (time scaling) is supported by
the data and is adopted.

## Key Parameters

- K: final daily trade count for a given stock-day; governs resolution of the binomial
  distribution and appears in the moment-extraction recursions.
- alpha_i = E[N_i(T)]: stock-specific expected daily trade count; serves as the
  time-scaling constant in Model (ii).
- Lambda_0: baseline integrated intensity with E[Lambda_0(T)] = 1; shared across all
  stocks under Model (ii).
- V_n = sum_K z_K / K^n: precomputed constants summarizing the empirical trade-count
  distribution; used to pool moment estimates across heterogeneous stocks.
- Histogram resolution: 391 time bins (one per minute, 9:30–16:00 plus initial state)
  x 253 or 1261 relative-volume bins (prime numbers chosen to avoid rounding artifacts
  on bin boundaries).

## Data Requirements

- NYSE TAQ database: tick-by-tick trade records for all stocks trading from
  1 June 2001 to 31 August 2001 (60 trading days after excluding two anomalous days).
- Total sample: 203,158 relative trade count sample paths across all stocks.
- Each sample path: the sequence of cumulative trade counts N(t) at each minute
  t in {9:31, ..., 16:00}, together with the final count N(T) = K.
- Stocks are split into eight trade-count bands of approximately 30,000 paths each
  for the analysis of intensity moments by trade-count stratum.
- Minimum threshold: stocks with at least 50 trades per day are included in the
  histogram; lower-count stocks are discussed separately (market-open effect).

## Results

- The conditional distribution of relative volume is exactly binomial, directed by
  the self-normalized integrated intensity — this is an analytic result, not an
  empirical approximation.
- The mean of the self-normalized integrated intensity E[Lambda(t)/Lambda(T)]
  reproduces the classic U-shaped intra-day seasonality of NYSE trading volume,
  but the shape varies by trade-count band: low-count stocks show a stronger
  opening spike because a single opening print represents a large fraction of their
  daily activity.
- The variance of Lambda(t)/Lambda(T) scales approximately as 1/sqrt(K), where K
  is the final trade count. This trade-count dependence rejects Model (i)
  (amplitude scaling) and supports Model (ii) (time scaling of the market clock).
- After rescaling variances by sqrt(K), the variance curves for different trade-count
  bands collapse approximately onto a single curve, consistent with a common
  baseline intensity Lambda_0 time-scaled by alpha_i.
- No sampling artifacts are found when comparing the 391x253 and 391x1261
  histograms, validating the coarser grid for practical use.

## Relationships

- Builds directly on Cox (1955) doubly stochastic Poisson process and on the
  theory of initial enlargement of filtrations (Jeulin 1980; Jacod 1985; Yor 1985).
- Motivated by and related to financial market microstructure models of trade
  arrivals using Cox processes: Engle and Russell (1998) autoregressive conditional
  duration model, Gourieroux et al. (1999), Rydberg and Shephard (2000).
- The time-scaling conclusion directly corroborates Ane and Geman (2000), who
  showed that trade count proxies the stochastic market clock (subordinator) and
  that asset returns are normal when time is measured by trade count.
- The binomial mixture structure is a special case of the general mixture theory
  for doubly stochastic point processes in Daley and Vere-Jones (1988).
- The U-shaped intra-day seasonality finding connects to Admati and Pfleiderer
  (1988) and Brock and Kleidon (1992) on theoretical explanations for this pattern.
- The VWAP application (using E[R(t;K)] as the expected volume participation curve)
  links to Berkowitz et al. (1988), who introduced VWAP as a transaction-cost
  benchmark.
- A later companion paper by McCulloch (2012) extends this framework explicitly
  toward VWAP execution, building on the stochastic model introduced here.

## Implementation Notes

- The 2-D histogram (time x relative volume) is the central data structure. It
  estimates the unconditional finite-dimensional distributions of R(t; K) and is
  straightforward to build from tick data.
- Prime numbers of bins (251 interior + 2 boundary = 253) are recommended to avoid
  floating-point rounding artifacts when mapping k/n fractions to bin indices.
- Moments of the mixing distribution are recovered from observed moments via the
  closed-form recursions in Section 4. These are numerically stable given enough
  data but require care when K is small (denominators involve K-1, K^2-3K+2, etc.).
- The V_n constants (Section 4.2) must be computed from the empirical trade-count
  distribution before extracting moments from pooled data; they are straightforward
  scalar summaries of that distribution.
- The paper does not specify a parametric form for the mixing distribution Phi_t(s)
  or the baseline intensity Lambda_0; it works entirely with nonparametric moment
  estimates. Fitting a parametric model (e.g., beta distribution for Phi_t, or a
  deterministic seasonality curve for E[Lambda_0]) is left as an open extension.
- The variance scaling Var[Lambda(t)/Lambda(T)] ~ 1/sqrt(K) suggests a possible
  connection to self-similar or fractional processes (Hurst exponent), noted as a
  direction for future research.
- For VWAP scheduling, the expected relative volume curve E[R(t;K)] = E[Lambda(t)/
  Lambda(T)] is the key output: it gives the fraction of daily volume expected by
  each minute of the day and can be read directly from the first moment of the
  2-D histogram.

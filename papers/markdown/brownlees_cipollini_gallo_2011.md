# **Intra-daily Volume Modeling and Prediction for Algorithmic Trading**

CHRISTIAN T. BROWNLEES

*Department of Finance, Stern School of Business, NYU*

FABRIZIO CIPOLLINI

*Dipartimento di Statistica "G. Parenti", Università di Firenze, Italy*

GIAMPIERO M. GALLO

*Dipartimento di Statistica "G. Parenti", Università di Firenze, Italy*

### ABSTRACT

The explosion of algorithmic trading has been one of the most prominent recent trends in the financial industry. Algorithmic trading consists of automated trading strategies that attempt to minimize transaction costs by optimally placing orders. The key ingredient of many of these strategies are intra-daily volume proportions forecasts. This work proposes a dynamic model for intra-daily volumes that captures salient features of the series such as time series dependence, intra-daily periodicity and volume asymmetry. Moreover, we introduce loss functions for the evaluation of proportion forecasts which retains both an operational and information theoretic interpretation. An empirical application on a set of widely traded index Exchange Traded Funds shows that the proposed methodology is able to significantly outperform common forecasting methods and delivers more precise predictions for Volume Weighted Average Price trading. (*JEL*: C22, C51, C53, G12)

KEYWORDS: forecasting, GMM, multiplicative error models, traded volumes, VWAP, ultra-high-frequency data

We would like to thank two anonymous referees and especially the Editor, Torben G. Andersen, for their comments and concerns. Thanks are also due to Robert Almgren, Bruno Biais, Claude Courbois, Rob Engle, Thierry Foucault, Terry Hendershott, Eric Ghysels, Farhang Farazmand, Dennis Kristensen, Albert Menkveld, Giovanni Urga, and seminar participants in the Columbia Econometrics Workshop and at the Goldman Sachs GSET Strategists Weekly seminar; conference participants in the Chicago London Conference, Dec. 5–6, 2008; the First FBF-IDEI-R Conference on Investment Banking and Financial Markets, Toulouse, March 26–27, 2009; The SOFIE 1st European Conference Geneva, 16–18 June, 2009; and (EC)<sup>2</sup> -Conference Aarhus, 18–19 December, 2009. Financial support from the Italian MIUR is gratefully acknowledged. The usual disclaimer applies. Address correspondence to Giampiero M. Gallo, Dipartimento di Statistica "G. Parenti", Università di Firenze, Viale G.B. Morgagni, 59 - 50134 Firenze – Italy, or e-mail: gallog@ds.unifi.it

doi: 10.1093/jjfinec/nbq024 Advance Access publication July 5, 2010

 c The Author 2011. Published by Oxford University Press. All rights reserved. For permissions, please e-mail: journals.permissions@oxfordjournals.org.

The last few years have witnessed a widespread development of automated order execution systems, typically known in the financial industry as *algorithmic* (or *algo*) *trading*. Such algorithms aim at enhancing order execution by strategically submitting orders: computer-based pattern recognition allows for instantaneous information processing and for subsequent action taken with limited (if any) human judgement and intervention. The impact of such an innovation has been quite substantial: Chordia, Roll, and Subrahmanyam (2008) recognize that algo trading is the main determinant of the increase in volume and the reduction in the average trade size observed over the recent years. In October 2006, the NYSE has boosted a mixed system of electronic and face-to-face auctions which brings automated trades to more than 70% of total trades according to some 2009 estimates.

One of the industry's concerns is to split orders in order to seek better price execution, while retaining transparency of the procedures *vis-à-vis* the client. The *Volume Weighted Average Price* (VWAP) (cf. Madhavan 2002) of an asset is a wellestablished benchmark to fulfill this goal: it is calculated at the end of the trading day as an average of intra-daily transaction prices weighted by the corresponding share of traded volume to total daily volume. In the simplest of the scenarios, customers can be guaranteed that their order will be executed at the VWAP and the algorithm can be tuned to spread that order during the day with the goal of achieving an average execution price close to the VWAP. Such order execution procedure is fairly popular and is commonly named *VWAP trading*. From an econometric point of view, it is interesting to note that the strategy needs to be based on accurate predictions of intra-daily volume proportions.

This paper deals with modeling and forecasting intra-daily volume time series with an application to VWAP trading. Despite the widespread use of ultrahigh-frequency data in econometrics for more than 15 years, there has been little investigation of the features of intra-daily volumes and there is no well-established econometric methodology for forecasting them. The main contribution of this paper is to turn a descriptive analysis of the patterns of intra-daily volume time series into the design of a novel dynamic specification which replicates the empirical regularities. In particular, daily average dynamics are strongly persistent, whereas intra-daily patterns are characterized by clustering around a stable intra-daily Ushaped periodic component. Combined with the fact that volumes are nonnegative quantities, such evidence suggests a *Component Multiplicative Error Model* (CMEM), an extension of the Multiplicative Error Model (MEM) (Engle 2002). We do not choose a specific distribution for the error term, preferring to work with a semiparametric specification, and we estimate all parameters jointly by Generalized Method of Moments (GMM). We evaluate volume forecast performance in the perspective of algorithmic trading proposing a loss function, the *Slicing Loss* function, suitable for proportions.

The empirical analysis is carried out on modeling volume turnover of three liquid Exchange Traded Funds (ETFs) which replicate the behavior of major U.S. stock indices, SPDR S&P 500 (SPY—S&P 500), Diamonds (DIA—Dow Jones), and PowerShares QQQ (QQQQ—NASDAQ), between 2002 and 2006. We use different metrics to measure precision gains in forecasting volumes, volume proportions, and VWAP. Results show that our model significantly outperforms a simple benchmark (common among practitioners), signaling that order execution can be enhanced by an econometric model molded after stylized facts.

Our paper fits in the large literature on intra-daily trading activity modeling. In particular, our model extends the logic of the component GARCH model for intra-daily volatility suggested by Engle, Sokalska, and Chanda (2007) and has some connections with P-GARCH models introduced by Bollerslev and Ghysels (1996). Bialkowski, Darolles, and Le Fol (2008) concentrate on volume dynamics of several stocks, isolating common and idiosyncratic components. Gouriéroux, Jasiak, and Le Fol (1999) fix a volume level and model the time needed to trade it. Andersen (1996) models volumes together with volatility as a function of an underlying latent information flow. The paper is also related to a recent strand of literature which shows how econometric methodology can be applied to the analysis and reduction of transaction costs, such as Taylor (2002), Engle (2002), Bikker et al. (2008), and Härdle, Hautsch, and Mihoci (2009) among others.

The paper is organized as follows. We start from stylized facts (Section 1) to motivate the Component MEM (Section 2). Section 2.3 gives details on the estimation procedure. The empirical application is divided up between model estimation and diagnostics (Section 2.4) and volume forecasting and VWAP forecast comparisons (Section 3). Concluding remarks follow (Section 4).

### **1 THE EMPIRICAL REGULARITIES OF INTRA-DAILY VOLUMES**

We chose to analyze ETFs, innovative financial products which allow straightforward trading in market averages as if they were stocks, while avoiding the possible idiosyncrasies of single shares. In the present framework, we count on a dataset consisting of regularly spaced intra-daily volumes and transaction price data for three popular equity index ETFs, SPY (S&P 500 ETF), DIA (Dow Jones ETF), and QQQQ (Nasdaq ETF), over a sample period between January 2002 and December 2006, excluding days with empty bins. The frequency of the intra-daily data is 15 min (26 intra-daily bins). Volumes are computed as the sum of all transaction volumes exchanged within each intra-daily bin: in order to correct for some possible trending behavior in volume, we divide the original series (multiplied by 100) by the number of daily outstanding shares.<sup>1</sup> The last recorded transaction price before the end of each bin is used as the reference price. The ultra-high-frequency data used in the analysis are extracted from the TAQ while shares outstanding are taken from the CRSP. Details on the series handling and management are documented in Brownlees and Gallo (2006).

<sup>1</sup>Andersen (1996) proposes two flexible procedures to remove the trend in volumes; while not needed in the present case, an appropriate further adjustment of the data may be advisable with other series.

![](_page_3_Figure_3.jpeg)

**Figure 1** SPY turnover data: original turnover data (top); daily averages (center), intra-daily component (bottom). January 2002 to December 2006.

While similar evidence also holds for the other tickers (for which we report summary descriptive statistics only), let us focus on the empirical regularities of the SPY turnover series (1247 trading days and 32,422 observations) by starting with a graphic appraisal of the 15 min turnover (top panel of Figure 1). As with most financial time series, it clearly exhibits clustering of trading activity. Such a pattern is retained by taking daily averages (cf. second panel of Figure 1), which we interpret as evidence that the overall series clusters around a lower-frequency component. Dividing each observation by the corresponding daily average, we obtain the intra-daily pattern (bottom panel of Figure 1, also reproduced as the

![](_page_4_Figure_3.jpeg)

**Figure 2** SPY turnover data: intra-daily component (top); intra-daily periodic component (center); intra-daily nonperiodic component (bottom). January 2002 to December 2006.

top panel of Figure 2 for ease of reference). Allowing for the possible presence of periodic and nonperiodic components, we compute averages by time of day (26 bins—center panel of Figure 2) which exhibit a U shape in line with other intradaily financial time series (e.g., average durations, the trading activity is higher at the opening and closing of the markets and is lower around midday). The ratio between these two series shows the nonperiodic pattern in the bottom panel of Figure 2.

The dynamic features of these three series (daily, intra-daily periodic, and intra-daily nonperiodic) are summarized in the corresponding correlograms (left panel of Figure 3 for raw data; right panel for the daily averages). Focusing on the

![](_page_5_Figure_3.jpeg)

**Figure 3** SPY turnover data: autocorrelation function of the original turnover (left panel) and of the daily averages (right panel).

original series divided by daily averages, we show marked periodicity (correlogram in the left panel of Figure 4); adjusting this series by time-of-day averages, we remove periodicity but some short-lived dependence is retained (correlogram in the right panel of Figure 4).

The autocorrelations of the components of the 15 min series for all three tickers (Table 1) confirm the graphical analysis of SPY. The overall time series displays relatively high levels of persistence which are also slowly decaying. The autocorrelations do not decrease by daily averaging. By dividing the overall turnover by its daily average (intra-daily component), a substantial part of dependence in the series is removed. Finally, once the intra-daily periodic component is removed,

![](_page_5_Figure_7.jpeg)

**Figure 4** SPY turnover data: autocorrelation function of the intra-daily component (left panel) and of the intra-daily nonperiodic component (right panel).

**Table 1** Autocorrelations at selected lags of the turnover time series components. The table reports the lag 1 (*ρ*ˆ<sup>1</sup> ) and lag 26 (*ρ*ˆ<sup>1</sup> day) autocorrelations of the intra-daily frequency components (overall, intra-daily and intra-daily nonperiodic) and lag 1 (*ρ*ˆ<sup>1</sup> day) and lag 5 (*ρ*ˆ<sup>1</sup> week) autocorrelations of the daily frequency component (daily).

|      | Overall |            | Daily      |             |      | Intra-daily | Intra-daily<br>nonperiodic |            |  |
|------|---------|------------|------------|-------------|------|-------------|----------------------------|------------|--|
|      | ρˆ1     | ρˆ1<br>day | ρˆ1<br>day | ρˆ1<br>week | ρˆ1  | ρˆ1<br>day  | ρˆ1                        | ρˆ1<br>day |  |
| SPY  | 0.74    | 0.59       | 0.84       | 0.76        | 0.46 | 0.37        | 0.26                       | 0.00       |  |
| DIA  | 0.60    | 0.41       | 0.72       | 0.60        | 0.35 | 0.26        | 0.18                       | 0.02       |  |
| QQQQ | 0.69    | 0.53       | 0.77       | 0.66        | 0.53 | 0.45        | 0.29                       | 0.01       |  |

the resulting series shows significant low-order correlations only. Interestingly, the magnitudes of the various autocorrelations of the series are remarkably similar across the assets.

# **2 A MULTIPLICATIVE ERROR MODEL FOR INTRA-DAILY VOLUMES**

Based on the empirical regularities discussed in Section 1, we decompose the dynamics of intra-daily volumes in one daily and two intra-daily (one periodic and one dynamic) components. Let us first establish the notation used in what follows. Days are denoted with *t* ∈ {1, . . . , *T*}; each day is divided into *I* equally spaced intervals (referred to as bins) indexed by *i* ∈ {1, . . . , *I*}. In order to simplify the notation, we may label observations indexed by the double subscript *t i* with a single progressive subscript *τ* = *I* × (*t* − 1) + *i*. Correspondingly, we denote the total number of observations by *N* (*T* × *I*).

The nonnegative quantity under analysis relative to bin *i* of day *t* is denoted as *xt i* or, alternatively, as *xτ*. F*t i*−<sup>1</sup> indicates the information about *xt i* available before forecasting it. Usually, we will assume F*<sup>t</sup>* <sup>0</sup> = F*t*−<sup>1</sup> *<sup>I</sup>* but it may possible to include additional pieces of market opening information into F*<sup>t</sup>* <sup>0</sup>.

We adopt the following convention: if **x**1, . . . , **x***<sup>K</sup>* are (*m*, *n*) matrices, then (**x**1; . . . ; **x***K*) represents the (*mK*, *n*) matrix obtained stacking the **x***<sup>t</sup>* matrices columnwise.

### **2.1 Model Definition**

Following the logic of MEM and in view of the motivation provided by the stylized facts in Section 1, we propose a CMEM

$$x_{ti} = \eta_t \phi_i \mu_{ti} \varepsilon_{ti}$$
.

The multiplicative innovation term  $\varepsilon_{ti}$  is assumed i.i.d., nonnegative, with mean 1 and constant variance  $\sigma^2$ :

$$\varepsilon_{ti}|\mathcal{F}_{ti-1}\sim(1,\sigma^2).$$
 (1)

The conditional expectation of  $x_{ti}$  is the product of three multiplicative elements:

- $\eta_t$ , a daily component;
- $\phi_i$ , an intra-daily periodic component aimed to reproduce the time-of-day pattern;
- $\mu_{ti}$ , an intra-daily dynamic (nonperiodic) component.

In order to simplify the exposition, we assume a relatively simple specification for the components. If needed, the formulation proposed can be trivially generalized, for instance by including asymmetric effects, other predetermined variables, more lags, and so on (see the empirical application in Section 2.4).

The daily component is structured as

$$\eta_t = \alpha_0^{(\eta)} + \beta_1^{(\eta)} \eta_{t-1} + \alpha_1^{(\eta)} x_{t-1}^{(\eta)}$$
 (2)

where  $x^{(\eta)}$  is what we name the *standardized daily volume*,

$$x_t^{(\eta)} = \frac{1}{I} \sum_{i=1}^{I} \frac{x_{ti}}{\phi_i \, \mu_{ti}},\tag{3}$$

that is the daily average of the intra-daily volumes normalized by the intra-daily components  $\phi_i$  and  $\mu_{t\,i}$ .

The intra-daily dynamic component is formulated as

$$\mu_{ti} = \alpha_0^{(\mu)} + \beta_1^{(\mu)} \mu_{ti-1} + \alpha_1^{(\mu)} x_{ti-1}^{(\mu)} \tag{4}$$

where  $x^{(\mu)}$  is the *standardized intra-daily volume*,

$$x_{ti}^{(\mu)} = \frac{x_{ti}}{\eta_t \, \phi_i}.\tag{5}$$

Furthermore,  $\mu_{t\,i}$  is constrained to have unconditional expectation equal to 1 in order to make the model identifiable, allowing us to interpret it as a pure intra-daily dynamic component. This implies  $\alpha_0^{(\mu)}=1-\beta_1^{(\mu)}-\alpha_1^{(\mu)}$ . The starting conditions for the system we use are  $\eta_0=x_0^{(\eta)}=\frac{1}{5}\sum_{t=1}^5\sum_{i=1}^Ix_{t\,i}$  (that is the daily volume average over the first week of the sample) and  $\mu_{1\,0}=x_{1\,0}^{(\mu)}=1$ .

The intra-daily nonperiodic component can be initialized at each day t with the latest quantities available, namely,

$$\mu_{t\,0} = \mu_{t-1\,I} \qquad x_{t\,0}^{(\mu)} = x_{t-1\,I}^{(\mu)}.$$

In synthesis, the system nests the daily and the intra-daily dynamic components by alternating the update of the former (from  $\eta_{t-1}$  to  $\eta_t$ ) and of the latter (from  $\mu_{t0} = \mu_{t-1}I$  to  $\mu_{t1}$ ). Time-varying  $\eta_t$  adjusts the mean level of the series, whereas the intra-daily component  $\phi_i \mu_{ti}$  captures bin-specific departures from such an average level.

Note that defining  $x_{ti}^{(\mu)}$  as in Equation (5) implies  $x_{ti}^{(\mu)} = \mu_{ti} \varepsilon_{ti}$ . Combining this with Equation (1), one obtains

$$E(x_{ti}^{(\mu)}|\mathcal{F}_{ti-1}) = \mu_{ti} \qquad V(x_{ti}^{(\mu)}|\mathcal{F}_{ti-1}) = \mu_{ti}^2 \sigma^2$$
 (6)

that coincide with the properties of the corresponding quantity in the usual MEM (Engle 2002). A similar consideration can be made for  $x_t^{(\eta)}$ . In fact, definition (3) implies  $x_t^{(\eta)} = \eta_t \bar{\varepsilon}_t$ , where  $\bar{\varepsilon}_t = I^{-1} \sum_{i=1}^{I} \varepsilon_{ti}$ , and thus

$$E(x_t^{(\eta)}|\mathcal{F}_{t-1\,I}) = \eta_t \qquad V(x_t^{(\eta)}|\mathcal{F}_{t-1\,I}) = \eta_t^2 \sigma^2 / I.$$
 (7)

On this base,  $x_t^{(\eta)}$  and  $x_{ti}^{(\mu)}$  are adjusted versions of the observed  $x_{ti}$ 's that can be interpreted as transmitting an innovation content through the respective equations.

The intra-daily periodic component  $\phi_i$  can be specified in various ways, but here we retain a parsimonious parameterization of  $\phi_i$  via a Fourier (sine/cosine) representation:

$$\phi_{i+1} = \exp\left\{\sum_{k=1}^{K} \left[\delta_{1k}\cos\left(fki\right) + \delta_{2k}\sin\left(fki\right)\right]\right\}$$
(8)

where  $f = 2\pi/I$ , K is the integer part of I/2,  $\delta_{2K} = 0$  if I is even, and  $i = 0, \ldots, I-1$ . Besides the good approximation properties of the Fourier series, what also makes this basis of functions appealing is the fact that the sine/cosine terms are orthogonal and bounded functions (as opposed, for instance, to splines), and this significantly eases the nonlinear numerical optimization required for the estimation (also see White (2006) as well as Gallant (1981) and Gallant (1984)). Moreover, the number of terms in Equation (8) may be considerably reduced if the periodic intra-daily pattern is sufficiently smooth since few low-frequency harmonics may be enough. Alternatively, shrinkage-type estimation may allow flexibility and parsimony of the estimated periodic component (cf. Brownlees and Gallo 2010).

### 2.2 Discussion

**2.2.1** Correspondence of the CMEM to the descriptive analysis. The daily average  $\overline{x}_{t} = I^{-1} \sum_{i=1}^{I} x_{ti}$  represents a proxy of the daily component  $\eta_t$ . In fact, by taking its expectation conditionally on the previous day, we have

$$E(\overline{x}_{t.}|\mathcal{F}_{t-1I}) = \eta_t \frac{1}{I} \sum_{i=1}^{I} \phi_i E(\mu_{ti}|\mathcal{F}_{t-1I}). \tag{9}$$

From Equation (4), let us write the conditional expectations during the day in terms of the forecast for the first bin in *t*:

$$E(\mu_{ti}|\mathcal{F}_{t-1I}) = 1 + (\alpha_1^{(\mu)} + \beta_1^{(\mu)})^{(i-1)} [E(\mu_{t1}|\mathcal{F}_{t-1I}) - 1], \quad i = 2, ..., I.$$

Correspondingly, Equation (9) becomes

$$E(\overline{x}_{t.}|\mathcal{F}_{t-1\,I}) = \eta_t \frac{1}{I} \sum_{i=1}^{I} \phi_i + \frac{\eta_t \left[ E(\mu_{t\,1}|\mathcal{F}_{t-1\,I}) - 1 \right]}{I} \sum_{i=1}^{I} \phi_i \left( \alpha_1^{(\mu)} + \beta_1^{(\mu)} \right)^{(i-1)}$$

which can be approximated by the first term  $\eta_t \overline{\phi}$ , although the contribution of the second term may be substantial if the terminal value of the day before brings the forecast of the first bin to be much above or below 1.<sup>2</sup>

Once the daily averages are computed, the ratio  $x_{ti}^{(I)} = x_{ti}/\overline{x}_{t}$  can be used as a proxy for the whole intra-daily component  $\phi_i \mu_{ti}$  since

$$x_{ti}^{(I)} = \frac{x_{ti}}{\overline{x}_{t.}} \simeq \frac{\eta_t \phi_i \mu_{ti} \varepsilon_{ti}}{\eta_t \overline{\phi}} = \frac{\phi_i \mu_{ti} \varepsilon_{ti}}{\overline{\phi}}.$$
 (10)

The bin average of the quantities into Equation (10), namely  $\overline{x}_{.i}^{(I)} = T^{-1} \sum_{t=1}^{T} x_{ti}^{(I)}$ , represents a proxy for the intra-daily periodic component  $\phi_i$ . In fact,

$$\overline{x}_{.i}^{(I)} = \frac{1}{T} \sum_{t=1}^{T} x_{ti}^{(I)} \simeq \frac{\phi_i}{\overline{\phi}} \frac{1}{T} \sum_{t=1}^{T} \mu_{ti} \varepsilon_{ti}.$$
 (11)

By taking its expectation conditionally on the starting information  $\mathcal{F}_{0I}$ , we have

$$E(\overline{x}_{.i}^{(I)}|\mathcal{F}_{0I}) \simeq \frac{\phi_i}{\overline{\phi}} \frac{1}{T} \sum_{t=1}^{T} E(\mu_{ti}|\mathcal{F}_{0I}) \simeq \frac{\phi_i}{\overline{\phi}}.$$
 (12)

The last approximation can be motivated by the fact that, after a few days, the conditional expectations of the  $\mu_{ti}$ 's match the unconditional one.

<sup>&</sup>lt;sup>2</sup> We also remark that the log formulation of the intra-daily periodic component guarantees  $\prod_{i=1}^{I} \phi_i = 1$ , but not  $\overline{\phi} = 1$ . This is a minor issue since  $\overline{\phi}$  is a constant (once the periodic component is fixed) scaling factor for the daily component and is quite close to 1 in the applications considered.

Finally, the residual quantity  $x_{ti}^{(I)}/\overline{x}_{.i}^{(I)}=x_{ti}/\overline{x}_{.i}$  can be justified as a proxy for the intra-daily nonperiodic component since

$$\frac{x_{ti}^{(I)}}{\overline{x}_{i}^{(I)}} \simeq \frac{\phi_{i}\mu_{ti}\varepsilon_{ti}/\overline{\phi}}{\phi_{i}/\overline{\phi}} = \mu_{ti}\varepsilon_{ti}.$$
(13)

**2.2.2 CMEM and component GARCH.** Although the CMEM of Section 2.1 has some relationships with the component GARCH model suggested by Engle, Sokalska, and Chanda (2007) for modeling intra-daily volatility, our proposal differs under a number of aspects. The main distinguishing feature lies in the evolution of the daily and intra-daily components: exploiting the scheme proposed, all parameters of the model can be estimated jointly, instead to resorting to a multistep procedure.

**2.2.3 CMEM and periodic GARCH.** Our CMEM shares some features with the P-GARCH model (Bollerslev and Ghysels 1996) as well. By grouping the intradaily components  $\phi_i$  and  $\mu_{ti}$  and referring to Equation (4) for the latter, the combined component can be written as

$$\phi_i \, \mu_{t\,i} = \alpha_{0\,i}^{(\mu)} + \beta_{1\,i}^{(\mu)} \, \mu_{t\,i-1} + \alpha_{1\,i}^{(\mu)} \, \chi_{t\,i-1}^{(\mu)}, \tag{14}$$

where

$$\alpha_{0i}^{(\mu)} = \alpha_{0}^{(\mu)} \phi_{i} \qquad \alpha_{1i}^{(\mu)} = \alpha_{1}^{(\mu)} \phi_{i} \qquad \beta_{1i}^{(\mu)} = \beta_{1}^{(\mu)} \phi_{i}. \tag{15}$$

In practice, the coefficients defined in Equation (15) are periodic: their pattern is governed by  $\phi_i$  but each is rescaled by a (possibly) different value, as it would be in a P-GARCH-like formulation. We adopt a considerable simplification by imposing the same periodic pattern to all coefficients. In this respect, we are inspired by the results in Martens, Chang, and Taylor (2002) where a relatively parsimonious formulation is adopted, based on an intra-daily periodic component scaling the dynamic (GARCH-like) component of the variance. The corresponding forecasts of the intra-daily volatility are only marginally worse than a more computationally expensive P-GARCH. Martens, Chang, and Taylor (2002) provide also empirical evidence in favor of the exponential formulation of the periodic intra-daily component and support its representation in a Fourier form (even if in their application they consider only the first four harmonics). This notwithstanding, we substantially depart from their approach since we add an explicit dynamic structure for the daily component, taking the intra-daily component to be a corresponding scale factor, and we estimate all CMEM parameters jointly.

#### 2.3 Inference

Let us now illustrate how to conduct inference regarding the model specified in Section 2.1. We group the main parameters of interest into the *p*-dimensional vector  $\boldsymbol{\theta} = (\boldsymbol{\theta}^{(\eta)}; \boldsymbol{\phi}; \boldsymbol{\theta}^{(\mu)})$ , where the three subvectors refer to the corresponding components of the model. Relative to these, the variance of the error term,  $\sigma^2$ , represents a nuisance parameter.

Since the model is specified in a semiparametric way (see Equation (1)), we focus our attention on the GMM (Newey and McFadden 1994, Wooldridge 1994) as an estimation strategy not requiring the adoption of a density function for the innovation term.

Rather than by GMM, MEMs are often estimated by QMLE by maximizing the log-likelihood of the specification based on a Gamma distribution assumption for the innovation term (see Engle and Gallo 2006). The first-order conditions for the conditional mean parameters are in fact the same for the two estimators. However, the portion of the Gamma log-likelihood due to the Gamma dispersion parameter is not defined or overflows numerically when, respectively, zeros or inliers<sup>3</sup> are present in the data. In contrast, our GMM approach is robust to such features which are common in these datasets, especially when dealing with a higher number of intra-daily bins or illiquid assets.

#### 2.3.1 Efficient GMM inference. Let

$$u_{\tau} = \frac{x_{\tau}}{\eta_t \, \phi_i \, u_{\tau}} - 1,\tag{16}$$

where we simplified the notation by suppressing the reference to the dependency of  $u_{\tau}$  on the parameters  $\boldsymbol{\theta}$ , on the information  $\mathcal{F}_{\tau-1}$ , and on the current value of the dependent variable  $x_{\tau}$ .  $u_{\tau}$  is a conditionally homoskedastic martingale difference, given that its conditional expectation is 0 and its conditional variance is  $\sigma^2$ . As a consequence, let us consider any (M,1) vector  $\mathbf{G}_{\tau}$  depending deterministically on the information  $\mathcal{F}_{\tau-1}$  and write  $\mathbf{G}_{\tau}u_{\tau} \equiv \mathbf{g}_{\tau}$ . We have

$$E(\mathbf{g}_{\tau}|\mathcal{F}_{\tau-1}) = \mathbf{0} \quad \forall \, \tau \Rightarrow E(\mathbf{g}_{\tau}) = \mathbf{0},$$
 (17)

by the law of iterated expectations;  $\mathbf{g}_{\tau}$  is also a martingale difference.

Assuming that the absolute values of  $u_{\tau}$  and  $\mathbf{G}_{\tau}u_{\tau}$  have finite expectations, the uncorrelatedness of  $\mathbf{G}_{\tau}$  and  $u_{\tau}$  provides an instrument role to the former.  $\mathbf{G}_{\tau}$  may depend on nuisance parameters collected into the vector  $\boldsymbol{\psi}$ . In order for us to concentrate on estimating  $\boldsymbol{\theta}$ , we assume for the moment that  $\boldsymbol{\psi}$  is a known constant, postponing any further discussion about its role and how to draw inference about it to the end of this section and to Section 2.3.2.

<sup>&</sup>lt;sup>3</sup> Inliers are observations that are anomalous by being too small (in this context, too close to 0). Although relevant in general, in our empirical application, this is not a problem.

If M = p, we have as many equations as the dimension of  $\theta$ , thus leading to the moment criterion

$$\overline{\mathbf{g}} = \frac{1}{N} \sum_{\tau=1}^{N} \mathbf{g}_{\tau} = \mathbf{0}. \tag{18}$$

Under correct specification of the  $\eta_t$ ,  $\phi_i$ , and  $\mu_{ti}$  equations and some regularity conditions, the GMM estimator  $\hat{\theta}_N$ , obtained solving Equation (18) for  $\theta$ , is consistent (Wooldridge 1994, th. 7.1). Furthermore, under additional regularity conditions, we have asymptotic normality of  $\hat{\theta}_N$ , with asymptotic covariance matrix (Wooldridge 1994, th. 7.2)

$$Avar(\widehat{\boldsymbol{\theta}}_N) = \frac{1}{N} (\mathbf{S}' \mathbf{V}^{-1} \mathbf{S})^{-1}, \tag{19}$$

where

$$\mathbf{S} = \lim_{N \to \infty} \frac{1}{N} \sum_{\tau=1}^{N} E\left(\nabla_{\boldsymbol{\theta}} \cdot \mathbf{g}_{\tau}\right)$$
 (20)

$$\mathbf{V} = \lim_{N \to \infty} \frac{1}{N} V \left( \sum_{\tau=1}^{N} \mathbf{g}_{\tau} \right) = \lim_{N \to \infty} \left[ \frac{1}{N} \sum_{\tau=1}^{N} E \left( \mathbf{g}_{\tau} \mathbf{g}_{\tau}' \right) \right]. \tag{21}$$

The last expression for V comes from the fact that  $g_T$  is a martingale difference since this is a sufficient condition for making these terms serially uncorrelated; moreover, the same condition leads to simplifications in the assumptions needed for the asymptotic normality, by virtue of the martingale CLT.

The martingale difference structure of  $u_{\tau}$  gives also a simple formulation for the *efficient* choice of the instrument  $G_{\tau}$ , associated with the "smallest" asymptotic variance among the GMM estimators generated by  $\overline{\mathbf{g}}$  functions structured as in Equation (18). Such efficient choice is

$$\mathbf{G}_{\tau}^{*} = -E(\nabla_{\theta} u_{\tau} | \mathcal{F}_{\tau-1}) V(u_{\tau} | \mathcal{F}_{\tau-1})^{-1}. \tag{22}$$

Inserting  $E\left(\mathbf{g}_{\tau}\mathbf{g}_{\tau}'\right)$  into Equation (21) and  $E\left(\nabla_{\boldsymbol{\theta}'}\mathbf{g}_{\tau}\right)$  into Equation (20), we obtain

$$E\left(\mathbf{g}_{\tau}\mathbf{g}_{\tau}^{\prime}\right) = -E\left(\nabla_{\boldsymbol{\theta}^{\prime}}\mathbf{g}_{\tau}\right) = \sigma^{2}E\left(\mathbf{G}_{\tau}^{*}\mathbf{G}_{\tau}^{*\prime}\right),$$

so that

$$\mathbf{V} = -\mathbf{S} = \sigma^2 \lim_{N \to \infty} \frac{1}{N} \sum_{\tau=1}^{N} E\left(\mathbf{G}_{\tau}^* \mathbf{G}_{\tau}^{*\prime}\right)$$

and Equation (19) specializes to

$$Avar(\widehat{\boldsymbol{\theta}}_N) = \frac{1}{N} (\mathbf{S}' \mathbf{V}^{-1} \mathbf{S})^{-1} = -\frac{1}{N} \mathbf{S}^{-1} = \frac{1}{N} \mathbf{V}^{-1}.$$
 (23)

Considering the analytical structure of  $u_{\tau}$  in the model (Equation (16)), we have

$$\nabla_{\boldsymbol{\theta}} u_{\tau} = -\mathbf{a}_{\tau}(u_{\tau}+1),$$

where

$$\mathbf{a}_{\tau} = \eta_t^{-1} \nabla_{\theta} \eta_t + \mu_{\tau}^{-1} \nabla_{\theta} \mu_{\tau} + \phi_i^{-1} \nabla_{\theta} \phi_i \tag{24}$$

so that Equation (22) becomes

$$\mathbf{G}_{\tau}^* = \mathbf{a}_{\tau} \sigma^{-2}$$
.

Substituting it into  $\mathbf{g}_{\tau} = \mathbf{G}_{\tau}u_{\tau}$  and this, in turn, into Equation (18), we obtain that the GMM estimator of  $\boldsymbol{\theta}$  in the CMEM solves the MM equation

$$\frac{1}{N} \sum_{\tau=1}^{N} \mathbf{a}_{\tau} u_{\tau} = \mathbf{0},\tag{25}$$

which does not depend on the nuisance parameter  $\sigma^2$  and, therefore, inference relative to the main parameter  $\theta$  does not depend on the estimation of  $\sigma^2$ .

The asymptotic variance matrix of  $\widehat{\boldsymbol{\theta}}_N$  is

$$\operatorname{Avar}(\widehat{\boldsymbol{\theta}}_{N}) = \frac{\sigma^{2}}{N} \left[ \lim_{N \to \infty} \frac{1}{N} \sum_{\tau=1}^{N} E(\mathbf{a}_{\tau} \mathbf{a}_{\tau}') \right]^{-1}$$
 (26)

that can be consistently estimated by

$$\widehat{\text{Avar}}(\widehat{\boldsymbol{\theta}}_N) = \widehat{\sigma}_N^2 \left[ \sum_{\tau=1}^N \mathbf{a}_{\tau} \mathbf{a}_{\tau}' \right]^{-1}$$
 (27)

where  $\hat{\sigma}_N^2$  is a consistent estimator of  $\sigma^2$  (Section 2.3.2) and  $\mathbf{a}_{\tau}$  is here evaluated at  $\hat{\boldsymbol{\theta}}_N$ .

**2.3.2 Inference on**  $\sigma^2$ **.** A straightforward estimator for the second moment  $\sigma^2$  of  $u_{\tau}$  (cf. Equation (16)), which is not compromised by zeros in the data, is

$$\hat{\sigma}_N^2 = \frac{1}{N} \sum_{\tau=1}^{N} \hat{u}_{\tau}^2$$
 (28)

where  $\hat{u}_{\tau}$ 's are computed from the value of  $\hat{\boldsymbol{\theta}}_{N}$ .

## 2.4 Empirical Application: In Sample Volume Analysis

The empirical application focuses on the same SPY, DIA, and QQQQ tickers previously examined, dividing each trading day in equally spaced bins of 15 min. We make the model more realistic by inserting some additional features which would have encumbered the notation presented before. We allow for a specific behavior in the first bin of the day to accommodate the possible accumulation of news during nontrading periods: a higher level of turnover at the beginning of the day is not necessarily linked to market activity in the last bin of the previous day (see Figure 2 - center panel). The insertion of a dummy variable for the first bin into the dynamic intra-daily component (in Equation (4) with a coefficient  $v_1^{(\mu)}$ ) marks the break in trading. In order to allow for possible differences in the dynamics related to the sign of returns, we also add asymmetric effects to the right-hand side of the daily and intra-daily dynamic components (Equations (2) and (4), respectively)

$$\gamma_1^{(\eta)} x_{t-1}^{-(\eta)}$$
 and  $\gamma_1^{(\mu)} x_{t-1}^{-(\mu)}$ ,

where

$$x_t^{-(\eta)} = x_t^{(\eta)} I(r_{t.} < 0)$$
 and  $x_{ti}^{-(\mu)} = x_{ti}^{(\mu)} I(r_{ti} < 0)$ 

denote the asymmetric versions of the standardized daily and intra-daily volumes, respectively, as defined in Section 2.1.  $r_t$ . is the total return in day t and  $r_{ti}$  is the return in bin i of day t (we assume that the return distribution has a zero median). The inclusion of asymmetric effects can be motivated by the well-documented empirical finding that bad news have more impact on subsequent volatility than good news (leverage effect—Nelson 1991, Glosten, Jagannanthan, and Runkle 1993, Rabemananjara and Zakoïan 1993) and the recognized existence of a common latent component (commonly interpreted as information flow) behind both volatilities and volumes (see Andersen 1996, Hautsch 2008). Consequently, we may expect negative returns to have an additional impact on subsequent volumes as well. Finally, we allow for a second lag in the intra-daily dynamic component.

Summarizing, by mixing the previous ingredients, we consider the following CMEMs:

**base:** CMEM with a dummy at bin 1, lag-1 dependence, and no asymmetric effects;

asym: base CMEM with lag-1 asymmetric effects (daily and intra-daily);intra2: base CMEM with a second lag in the intra-daily dynamic component;asym-intra2: intra2 CMEM with lag-1 asymmetric effects (daily and intra-daily).

The parameter estimates of the daily and intra-daily dynamic components are reported in Table 2.  $\alpha_0^{(\mu)}$  lacks standard errors because it is estimated via expec-

<sup>&</sup>lt;sup>4</sup>We thank Torben G. Andersen for pointing this out to us.

**Table 2** Parameter estimates for CMEM on turnover data. Sample period 2002–2006; intra-daily bins taken at 15 min. Standard errors are reported in parentheses.  $pers(\mu)$  indicates estimated persistence of the dynamic intra-daily component.

| Ticker | Specification | $\alpha_0^{(\eta)}$   | $\alpha_1^{(\eta)}$   | $\gamma^{(\eta)}$                     | $\beta^{(\eta)}$      | $\alpha_0^{(\mu)}$ | $\nu_1^{(\mu)}$  | $\alpha_1^{(\mu)}$    | $\alpha_2^{(\mu)}$ | $\gamma^{(\mu)}$                       | $\beta^{(\mu)}$       | σ     | $pers(\mu)$ |
|--------|---------------|-----------------------|-----------------------|---------------------------------------|-----------------------|--------------------|------------------|-----------------------|--------------------|----------------------------------------|-----------------------|-------|-------------|
| SPY    | Base          | 1.811<br>(0.369)      | 0.474<br>(0.037)      |                                       | 0.483<br>(0.040)      | 0.230              | 0.914<br>(0.207) | 0.340<br>(0.007)      |                    |                                        | 0.395<br>(0.013)      | 0.622 | 0.735       |
|        | Asym          | 1.665<br>(0.348)      | 0.431 $(0.037)$       | 0.041 $(0.010)$                       | 0.511 $(0.039)$       | 0.227              | 0.968 $(0.213)$  | 0.317                 |                    | 0.036 $(0.006)$                        | 0.400 $(0.013)$       | 0.622 | 0.735       |
|        | Intra2        | 1.925<br>(0.458)      | 0.504<br>(0.052)      | , ,                                   | 0.452<br>(0.057)      | 0.034              | 1.056<br>(0.172) | 0.354 (0.008)         | -0.235 $(0.010)$   |                                        | 0.807<br>(0.012)      | 0.622 | 0.900       |
|        | Asym-intra2   | 1.795 $(0.432)$       | $0.461 \atop (0.050)$ | $\underset{(0.012)}{0.041}$           | $0.480 \\ (0.054)$    | 0.034              | 1.084 $(0.170)$  | $0.338 \atop (0.008)$ | -0.233 $(0.010)$   | $\underset{\left(0.004\right)}{0.024}$ | $0.808 \atop (0.012)$ | 0.621 | 0.899       |
| DIA    | Base          | 1.951<br>(0.420)      | 0.419<br>(0.035)      |                                       | 0.536<br>(0.039)      | 0.199              | 0.897<br>(0.215) | 0.283<br>(0.008)      |                    |                                        | 0.484<br>(0.013)      | 0.834 | 0.767       |
|        | Asym          | 1.881<br>(0.387)      | 0.361 $(0.035)$       | $0.06 \atop (0.012)$                  | 0.567 $(0.038)$       | 0.200              | 0.934 $(0.222)$  | 0.265 $(0.008)$       |                    | 0.03 $(0.008)$                         | 0.484 $(0.013)$       | 0.833 | 0.751       |
|        | Intra2        | 1.766 $(0.479)$       | 0.452 $(0.045)$       |                                       | 0.511 $(0.049)$       | 0.041              | 1.237 $(0.223)$  | 0.309 $(0.009)$       | -0.188 $(0.011)$   |                                        | 0.791 $(0.013)$       | 0.833 | 0.888       |
|        | Asym-intra2   | 1.749 $(0.449)$       | 0.397 $(0.044)$       | $0.057 \atop (0.014)$                 | 0.538 $(0.047)$       | 0.041              | 1.235<br>(0.222) | 0.295 $(0.009)$       | -0.187 $(0.010)$   | 0.022 $(0.005)$                        | 0.793 $(0.013)$       | 0.832 | 0.889       |
| QQQQ   | Base          | 1.537<br>(0.369)      | 0.466<br>(0.036)      |                                       | 0.507<br>(0.038)      | 0.218              | 1.134<br>(0.220) | 0.359<br>(0.007)      |                    |                                        | 0.380<br>(0.012)      | 0.538 | 0.739       |
|        | Asym          | 1.466<br>(0.350)      | 0.423 $(0.036)$       | 0.045 $(0.008)$                       | 0.530 $(0.037)$       | 0.222              | 1.145 $(0.224)$  | 0.351<br>(0.008)      |                    | 0.012 $(0.006)$                        | 0.376 $(0.012)$       | 0.538 | 0.733       |
|        | Intra2        | 1.815<br>(0.460)      | 0.485 $(0.052)$       |                                       | 0.484 $(0.055)$       | 0.026              | 1.085<br>(0.157) | 0.372<br>(0.007)      | -0.258 $(0.01)$    |                                        | 0.818 $(0.011)$       | 0.536 | 0.905       |
|        | Asym-intra2   | $1.746 \atop (0.440)$ | $0.448 \atop (0.051)$ | $\underset{\left(0.01\right)}{0.041}$ | $0.503 \atop (0.054)$ | 0.027              | 1.119 $(0.162)$  | $0.365 \atop (0.008)$ | -0.256 $(0.01)$    | $0.011 \\ (0.003)$                     | $0.815 \atop (0.011)$ | 0.536 | 0.902       |

tation targeting by imposing  $E(\mu_{\tau}) = 1$  (as detailed in Section 2.1); the remaining parameters are jointly estimated via GMM as detailed in Section 2.3.

The parameter estimates are remarkably similar across assets, suggesting some common behavior in the volume dynamics. In the present context,  $\alpha_1^{(\eta)}$  is much larger than the customary values encountered in typical GARCH(1,1) estimated on daily returns. As expected, the coefficient of the daily asymmetric effect  $(\gamma^{(\eta)})$  is positive and always significant; moreover, its inclusion tends in general to increase  $\beta^{(\eta)}$  slightly. The level of persistence of the daily component  $(\alpha_1^{(\eta)} + \gamma^{(\eta)}/2 + \beta^{(\eta)})$  is in general around 0.95.

The level of persistence in the intra-daily component (column  $\operatorname{pers}(\mu)$ , evaluated as the largest eigenvalue of the companion matrix built from  $\alpha_l^{(\eta)} + \gamma_l^{(\eta)}/2 + \beta_l^{(\eta)}$  estimates, l=1,2) is relatively high and increases remarkably when  $\alpha_2^{(\mu)}$  is present. This last parameter is negative and with a relatively large magnitude, but it is such that the Nelson and Cao (1992) nonnegativity condition for the corresponding component is satisfied in all cases. The coefficients of the intra-daily asymmetric effects are positive and significant, even if smaller in magnitude than the corresponding daily values.

Table 3 displays some diagnostics on the estimated residuals. Summarizing, the CMEM makes a relatively good job in modeling turnover, in particular in the presence of a second lag in the intra-daily dynamic component. For such models, Ljung–Box statistics at lag 1 reveal that only the DIA ticker shows some marginally significant autocorrelation in residuals. Statistics computed for the 1 day, or 26 bin, window appear more problematic, as only the DIA produces insignificant statistics. To be noted that the very large number of observations makes the statistic sensitive to even small departures from the uncorrelatedness assumption. This appears to be confirmed by Figure 5 (relative to SPY residuals) for which single autocorrelations are substantially within the confidence bands whereas the global statistic is significant. On the other hand, squared residuals are by far less problematic for all formulations taken into account. The table and the figure reveal that the contribution of the lag-2 autoregressive term in improving the diagnostics is more relevant than the one of the asymmetric terms.

Finally, in order to provide some idea about the impact of different intra-daily frequencies, we estimated the same formulations on the tickers just considered but with bins of 30 min instead of 15 (Table 4). Focusing on the asym-intra2 version, the intra-daily persistence tends to diminish for larger bins, in particular as consequence of smaller  $\beta^{(\mu)}$  values, whereas intra-daily asymmetric effects tend to increase slightly. Also, the coefficients of the daily component tend to adjust as an effect of the different sampling frequencies. The persistence is about the same, but this comes from larger  $\beta^{(\eta)}$ 's and smaller  $\alpha_1^{(\eta)}$ 's. The doubling of the  $\alpha_0^{(\eta)}$  coefficients is just due to the fact that by halving the number of the intra-daily bins (from 26 to 13) the average level tends to double.

**Table 3** Residuals analysis. Sample period 2002–2006. The columns  $Q_l$  report the values of the Ljung–Box statistics for the null of no autocorrelation between 1 and the l-th lag (the corresponding p-values are in smaller font underneath).

| Ticker | Model       |                | Resid                 | uals $\hat{\epsilon}_{ti}$ |                  | Squared residuals $\hat{e}_{ti}^2$ |                       |                       |                       |  |  |
|--------|-------------|----------------|-----------------------|----------------------------|------------------|------------------------------------|-----------------------|-----------------------|-----------------------|--|--|
|        |             | $\hat{\rho}_1$ | $\hat{\rho}_{1\;day}$ | $Q_1$                      | $Q_{1\;day}$     | $\hat{\rho}_1$                     | $\hat{\rho}_{1\;day}$ | $Q_1$                 | $Q_{1\;\mathrm{day}}$ |  |  |
| SPY    | Base        | 0.012          | 0.014                 | 4.847<br>0.028             | 179.291<br>0.000 | 0.000                              | 0.006                 | $0.004 \\ 0.948$      | 36.406<br>0.084       |  |  |
|        | Asym        | 0.013          | 0.014                 | 5.584 $0.018$              | 177.625 $0.000$  | 0.000                              | 0.006                 | $0.000 \\ 0.988$      | 39.038<br>0.048       |  |  |
|        | Intra2      | -0.007         | 0.013                 | $\frac{1.582}{0.208}$      | 47.936 $0.005$   | -0.005                             | 0.005                 | 0.937 $0.333$         | 37.450<br>0.068       |  |  |
|        | Asym-intra2 | -0.006         | 0.013                 | $\frac{1.354}{0.245}$      | 49.87<br>0.003   | -0.005                             | 0.005                 | $0.970 \atop 0.325$   | 39.536<br>0.043       |  |  |
| DIA    | Base        | 0.011          | 0.004                 | 3.604<br>0.058             | 147.463<br>0.000 | -0.003                             | 0.006                 | 0.251<br>0.616        | 17.850<br>0.881       |  |  |
|        | Asym        | 0.010          | 0.004                 | 3.331<br>0.068             | 153.42 $0.000$   | -0.003                             | 0.005                 | 0.247 $0.619$         | 17.725<br>0.886       |  |  |
|        | Intra2      | -0.013         | 0.005                 | 5.328<br>0.021             | 36.335<br>0.086  | -0.008                             | 0.007                 | $\frac{1.869}{0.172}$ | 11.787 $0.992$        |  |  |
|        | Asym-intra2 | -0.012         | 0.005                 | 5.004 $0.025$              | 36.103 $0.090$   | -0.008                             | 0.008                 | $\frac{1.887}{0.170}$ | 12.233<br>0.990       |  |  |
| QQQQ   | Base        | 0.019          | 0.002                 | 11.647<br>0.001            | 239.935<br>0.000 | 0.011                              | -0.003                | 4.062<br>0.044        | 52.965<br>0.001       |  |  |
|        | Asym        | 0.020          | 0.002                 | 12.411 $0.000$             | 250.289          | 0.012                              | -0.003                | 4.997<br>0.025        | 55.825<br>0.001       |  |  |
|        | Intra2      | -0.001         | 0.000                 | 0.04 $0.841$               | 61.145 $0.000$   | 0.006                              | -0.004                | 1.262 $0.261$         | $34.250 \atop 0.129$  |  |  |
|        | Asym-intra2 | -0.001         | 0.000                 | 0.046<br>0.830             | 59.348<br>0.000  | 0.005                              | -0.004                | 0.864<br>0.353        | 32.806<br>0.168       |  |  |

**Table 4** Parameter estimates for CMEM on turnover data. Sample period 2002–2006; intra-daily bins taken at 30 min. Standard errors are reported in parentheses.  $pers(\mu)$  indicates estimated persistence of the dynamic intra-daily component.

| Ticker | Specification | $\alpha_0^{(\eta)}$ | $\alpha_1^{(\eta)}$ | $\gamma^{(\eta)}$ | $\beta^{(\eta)}$ | $\alpha_0^{(\mu)}$ | $\nu_1^{(\mu)}$  | $\alpha_1^{(\mu)}$ | $\alpha_2^{(\mu)}$ | $\gamma^{(\mu)}$ | $\beta^{(\mu)}$  | σ     | $pers(\mu)$ |
|--------|---------------|---------------------|---------------------|-------------------|------------------|--------------------|------------------|--------------------|--------------------|------------------|------------------|-------|-------------|
| SPY    | Asym-intra2   | 3.011<br>(0.786)    | 0.348<br>(0.040)    | 0.045<br>(0.011)  | 0.597<br>(0.044) | 0.055              | 0.999<br>(0.161) | 0.299<br>(0.011)   | -0.172 $(0.015)$   | 0.037<br>(0.005) | 0.723<br>(0.024) | 0.532 | 0.813       |
| DIA    | Asym-intra2   | 2.968<br>(0.804)    | 0.314<br>(0.037)    | 0.062<br>(0.013)  | 0.625<br>(0.040) | 0.065              | 1.125<br>(0.2)   | 0.268<br>(0.011)   | -0.138 (0.015      | 0.043<br>(0.007) | 0.697<br>(0.025) | 0.678 | 0.795       |
| QQQQ   | Asym-intra2   | 2.945<br>(0.816)    | 0.336<br>(0.041)    | 0.044<br>(0.009)  | 0.619<br>(0.044) | 0.046              | 1.081<br>(0.167) | 0.323<br>(0.011)   | -0.184<br>(0.016)  | 0.021<br>(0.005) | 0.722<br>(0.025) | 0.47  | 0.824       |

![](_page_19_Figure_3.jpeg)

**Figure 5** SPY turnover data: autocorrelation function of the residuals (left panels) and of the squared residuals (right panels). Base model (top panels); asym-intra2 model (bottom panels).

#### 3 INTRA-DAILY VOLUME FORECASTING FOR VWAP TRADING

# 3.1 VWAP Trading

The concept of daily VWAP was introduced by Berkowitz, Logue, and Noser (1988) as an average of intra-daily transaction prices weighted by the corresponding traded volume relative to the total volume traded during the day (*full VWAP* in Madhavan 2002). In the original paper, the difference between the price of a trade and the recorded VWAP was used to measure the market impact of that trade. As such, VWAP is used to evaluate execution performance, given that it is a very transparent measure, easily calculated at the end of the day with tick-by-tick data.

VWAP trading is thus defined as a procedure for splitting a certain number of shares into smaller size orders during the day, which will be executed at different prices with the net result of an average price that is close to the VWAP.<sup>5</sup> In what follows, the trade to be executed is treated as exogenously determined

<sup>&</sup>lt;sup>5</sup>Whether the VWAP benchmark is proposed on an agency base or on a guaranteed base (in exchange for a fee) is a technical aspect which does not have any bearings in what we discuss.

(cf. Bertsimas and Lo 1998, Almgren and Chriss 2000, Engle and Ferstenberg 2007). In order to implement the replicating strategy, we assume that we are price takers and no effort will be put in predicting prices while we concentrate on the fact that accurate intra-daily volume proportion forecasting is the base for VWAP replication.

Let the VWAP for day t be defined as

$$VWAP_{t} = \frac{\sum_{j=1}^{J_{t}} v_{t}(j) \ p_{t}(j)}{\sum_{j=1}^{J_{t}} v_{t}(j)}.$$

where  $p_t(j)$  and  $v_t(j)$  denote, respectively, the price and volume of the j-th transaction of day t and  $J_t$  is the total number of trades of day t. For a given partition of the trading day into I bins, it is possible to express the numerator of the VWAP as

$$\sum_{j=1}^{J_t} v_t(j) p_t(j) = \sum_{i=1}^{I} \left( \sum_{j \in \mathcal{J}_i} v_t(j) \right) \bar{p}_{ti}$$
$$= \sum_{i=1}^{I} x_{ti} \bar{p}_{ti},$$

where  $\bar{p}_{ti}$  is the VWAP of the *i*-th bin and  $\mathcal{J}_i$  denotes the set of indices of the trades belonging to the *i*-th bin. Hence,

$$\mathsf{VWAP}_t = \frac{\sum_{i=1}^I x_{t\,i} \; \bar{p}_{t\,i}}{\sum_{i=1}^I x_{t\,i}} = \sum_{i=1}^I w_{t\,i} \; \bar{p}_{t\,i} = \mathbf{w}_t' \mathbf{\bar{p}}_t$$

where  $w_{ti}$  is the intra-daily proportion of volumes traded in bin i on day t, that is  $w_{ti} = x_{ti} / \sum_{i=1}^{l} x_{ti}$ . Let  $\mathbf{y} = (y_1, \dots, y_l)$ , an order slicing strategy over day t with the same bin intervals. We can define the Average Execution Price as the quantity

$$\mathsf{AEP}_t = \sum_{i=1}^I y_i \bar{p}_{ti} = \mathbf{y}' \bar{\mathbf{p}}_t.$$

The choice variable being the vector  $\mathbf{y}$ , we can solve the problem of minimizing the distance between the two outcomes in a mean square error sense, namely

$$\min_{\mathbf{y}} \delta_t = (\mathbf{w}_t' \bar{\mathbf{p}}_t - \mathbf{y}' \bar{\mathbf{p}}_t)^2,$$

where solving the minimization problem leads to the first-order conditions

$$\frac{\mathrm{d}\delta_t}{\mathrm{d}\mathbf{y}} = \mathbf{0} \Rightarrow -2\bar{\mathbf{p}}_{\mathbf{t}}(\mathbf{w}_t - \mathbf{y})'\bar{\mathbf{p}}_t = \mathbf{0},\tag{29}$$

which has a meaningful solution for  $\mathbf{y} = \mathbf{w}_t$ , that is, when the order slicing sequence for each subperiod in the day reproduces exactly the overall relative volume for that subperiod.

The implication of Equation (29) is that the VWAP replication problem can be cast as an intra-daily volume proportion forecasting problem: the better we can predict the intra-daily volume proportions, the better we can track VWAP.

# 3.2 VWAP Replication Strategies

Following Bialkowski, Darolles, and Le Fol (2008), we consider two types of VWAP replication strategies: Static and Dynamic. The Static VWAP replication strategy assumes that the order slicing is set before the market opening and it is not revised during the trading day. In the Dynamic VWAP replication strategy scenario, on the other hand, order slicing is revised at each new subperiod as new intra-daily volumes are observed.

Let  $\hat{x}_{ti|t-1}$  be shorthand notation for the prediction of  $x_{ti}$  conditionally on the previous day full information set  $\mathcal{F}_{t-1}$ . The *Static VWAP* replication strategy is implemented using slices with weights given by

$$\widehat{w}_{t\,i|t-1} = \frac{\widehat{x}_{t\,i|t-1}}{\sum_{i=1}^{I} \widehat{x}_{t\,i|t-1}} \quad i = 1,..,I \; ,$$

that is, the proportion of volumes for bin i is given by predicted volume in bin i divided by the sum of the volume predictions.

Let  $\hat{x}_{ti|i-1}$  be shorthand notation to denote the prediction of  $x_{ti}$  conditionally on  $\mathcal{F}_{ti-1}$ . The *Dynamic VWAP* replication strategy is implemented using slices with weights given by

$$\widehat{w}_{t\,i|i-1} = \begin{cases} \frac{\widehat{x}_{t\,i|i-1}}{\sum_{j=1}^{I} \widehat{x}_{t\,j|i-1}} \left(1 - \sum_{j=1}^{i-1} \widehat{w}_{t\,j|j-1}\right) \ i = 1, \dots, I-1 \\ \left(1 - \sum_{i=1}^{I-1} \widehat{w}_{t\,i|i-1}\right) \qquad \qquad i = I \end{cases}$$

that is, for each intra-daily bin from 1 to I-1 the predicted proportion is given by the proportion of one-step ahead volumes with respect to the sum of the remaining predicted volumes multiplied by the slice proportion left to be traded. On the last period of the day I, the predicted proportion is equal to the remaining part of the slice that needs to be traded.

#### 3.3 Forecast Evaluation

We evaluate out-of-sample performance from different perspectives: intra-daily volumes, intra-daily volume proportions, and daily VWAP prediction.

.

A natural way to assess volume predictive ability is to consider the mean square prediction error of the volume forecasts, defined as

$$MSE^{vol} = \sum_{t=1}^{T} \sum_{i=1}^{I} (x_{ti} - \widehat{x}_{ti|.})^{2},$$

where *<sup>x</sup>*b*t i*|· denotes the volume from some VWAP replication and volume forecasting strategy. Although such a metric provides insights as to which model gives a more realistic description of volume dynamics, it does not necessarily provide useful information as to the performance of the models for VWAP trading.

The evaluation of the accuracy of intra-daily volume proportion forecasts calls for a suitable loss function, given the bounded interval within which they are constrained. As no well-established criteria to measure proportion predictions ability are present in the literature, we propose the following *Slicing* loss function:

$$L^{\text{slicing}} = -\sum_{t=1}^{T} \sum_{i=1}^{I} w_{ti} \log \widehat{w}_{ti},$$
 (30)

which we motivate from both an "operational" as well as an information theoretic perspective. In the spirit of Christoffersen (1998) in the context of Value at Risk, we ask ourselves which properties proportion forecast ought to have under correct specification. Assume that a broker is interested in trading *n* shares of the asset<sup>6</sup> each day. If the intra-daily volume proportion predictions are correct, the observed intra-daily volumes *nwt i*, *i* = 1, ..., *I* behave like a sample from a multinomial distribution with parameters *<sup>w</sup>*b*t i*, *<sup>i</sup>* <sup>=</sup> 1, ..., *<sup>I</sup>*, and *<sup>n</sup>*; that is

$$(n w_{t1},...,n w_{tI}) \sim \mathsf{Mult}(\widehat{w}_{t1},...,\widehat{w}_{tI},n).$$

This suggests that an appropriate loss function for the evaluation of such forecasts is the negative of the multinomial predictive log-likelihood

$$L^{\mathsf{Mult}} = -\sum_{t=1}^{T} \left( \log \frac{n!}{(n \, w_{t\, 1})! \dots (n \, w_{t\, I})!} + \sum_{i=1}^{I} n \, w_{t\, i} \log \widehat{w}_{t\, i} \right)$$

An alternative evaluation strategy consists of computing the distance between the actual and predicted intra-daily volume proportions as the discrepancy between two discrete distributions. Using the Kullback–Leibler measure, we get

$$L^{KL} = \sum_{t=1}^{T} \sum_{i=1}^{I} (w_{ti} \log w_{ti} - w_{ti} \log \widehat{w}_{ti}).$$

<sup>6</sup>We are implicitly assuming for simplicity's sake that the actual intra-daily proportions *nwt i* are all integer.

![](_page_23_Figure_3.jpeg)

**Figure 6** Slicing loss function for I = 3 and  $(w_1, w_2, w_3) = (0.3, 0.3, 0.4)$ .

Interestingly, both the Multinomial and Kullback–Leibler losses provide equivalent rankings among competing forecasting methods in that the comparison is driven by the common term  $-\sum_{t=1}^{I}\sum_{i=1}^{I}w_{t\,i}\log\widehat{w}_{t\,i}$ . Figure 6 shows a picture of the Slicing loss function in the case of three intra-daily bins when the actual proportions  $\mathbf{w}_t$  are (0.3,0.3,0.4). The Slicing loss function is defined over the I-1 dimensional simplex described by  $\sum_{i=1}^{I}\widehat{w}_{t\,i}=1$  and has a minimum in correspondence to the true values; the value of the loss function goes to infinity on the boundaries of the simplex (when the actual proportions are in the interior of the simplex) and is asymmetric.

Finally, we also consider *VWAP tracking errors* MSE as in Bialkowski, Darolles, and Le Fol (2008) defined as

$$\mathsf{MSE}^{\mathsf{VWAP}} = \sum_{t=1}^{T} \left( \frac{\mathsf{VWAP}_t - \widehat{\mathsf{VWAP}_t}}{\mathsf{VWAP}_t} 100 \right)^2,$$

where VWAP<sub>t</sub> is the VWAP of day t and  $\widehat{\text{VWAP}}_t$  is the realized average execution price obtained using some VWAP replication strategy and volume forecasting method. Both VWAP<sub>t</sub> and  $\widehat{\text{VWAP}}_t$  are computed using the last recorded price of the i-th bin as a proxy for the average price of the same interval. The VWAP tracking error for day t can be seen as an average of slicing errors within each bin

weighed by the relative deviation of the price associated to that bin with respect to the VWAP:

$$\mathsf{MSE}^{\mathsf{VWAP}} = \sum_{t=1}^{T} \left( \sum_{i=1}^{I} (w_{t\,i} - \widehat{w}_{t\,i}) \; \frac{\bar{p}_{t\,i}}{\mathsf{VWAP}_t} \right)^2 100^2$$

Note that the deviations of the prices from the daily VWAP add an extra source of noise which can spoil the correct ranking of slice forecasts. In light of this, we recommend evaluating the precision of the forecasts by means of the Slicing loss function.

# **3.4 Empirical Application: Out-of-Sample VWAP Prediction**

Our empirical application consists of a volume, volume proportion, and VWAP tracking exercise of the tickers SPY, DIA, and QQQQ between January 2005 and December 2006 (502 days, 6526 observations). Model misspecification and parameter instability are always a possible concern in forecasting: to this end, we adopt parameter estimates for our CMEM specifications which are recursively updated each week using data from 2002, producing turnover predictions under both Static and Dynamic VWAP replication strategies. In order to assess the usefulness of the proposed approach, we use (periodic) Rolling Means (RM) as a simple benchmark, that is the predicted volume for the *i*-th bin is obtained as the mean over the last 40 days at the same bin. The Rolling Means are implemented within the Static VWAP replication approach.

Figure 7 displays sequences of volume predictions for the SPY ticker on January 31, 2005. The forecast sequences are produced using the asym-intra2 specification starting from different intra-daily bins until the end of the day. We use this picture to offer some informal remarks on the potential differences in slicing decisions generated by the static and dynamic VWAP replication schemes. The periodic intra-daily pattern appears to dominate the intra-daily evolution of turnover forecasts but intra-daily bursts of activity can alter the intra-daily volume profile: thus, it is not straightforward to see large differences in the order slicing in the static VWAP replication case between our models and the benchmark. On the other hand, in the dynamic VWAP replication case, activity bursts can lead to quite improved order slicing decisions.

Table 5 reports the volume MSE, slicing loss, and VWAP tracking MSE together with asterisks denoting the significance of a Diebold–Mariano test of equal predictive ability with respect to RM using the corresponding loss functions. In terms of volume and volume proportion predictions, the CMEM dynamic VWAP replication performs best and significantly outperforms the benchmark, followed by the CMEM static VWAP replication which generally outperforms the benchmark as well. The ranking of the CMEM specifications reflects the in-sample estimation results with models with richer intra-daily dynamics and asymmetric

![](_page_25_Figure_3.jpeg)

**Figure 7** SPY: sequences of multi-step-ahead volume predictions. Asym-intra2 specification starting from different intra-daily bins (1, 7, 13, and 20) until the end of the day. January 31st, 2005.

terms performing the best. Results also point out that it is the inclusion of an extra intra-daily lag rather than asymmetric effects which delivers the most of the out-of-sample gains. Recalling the words of caution about its limits, the VWAP tracking MSE evidence is substantially in line with the volume MSE and slicing loss results. In the static case, our models do systematically better than the benchmark in the QQQQ and SPY case, although significance of outperformance is lost. In the dynamic case, our CMEMs are always able to significantly beat the benchmark, with the richest specifications obtaining the best results.

# **4 CONCLUSIONS**

In this paper, we propose a dynamic model with different components capturing the behavior of traded volumes (relative to outstanding shares) viewed from daily and (periodic and nonperiodic) intra-daily time perspectives. The parameters of this Component Multiplicative Error Model can be estimated in one step by the Generalized Method of Moments. The application to three major ETFs shows that both the static and the dynamic VWAP replication strategies generally outperform

**Table 5** Out-of-sample volume, slicing, and VWAP tracking forecasting results. For each ticker, specification, and VWAP replication strategy, the table reports the values of the volume, slicing, and VWAP tracking error loss functions.

|             |           | SPY       |          |           | DIA       |          | QQQQ      |           |          |  |
|-------------|-----------|-----------|----------|-----------|-----------|----------|-----------|-----------|----------|--|
|             | Volume    | slicing   | VWAP     | Volume    | slicing   | VWAP     | Volume    | slicing   | VWAP     |  |
|             |           |           |          |           | Static    |          |           |           |          |  |
| RM          | 43.179    | 3.2031    | 1.116    | 46.167    | 3.2096    | 1.334    | 53.296    | 3.1844    | 1.799    |  |
| Base        | 41.623*** | 3.1976*** | 1.114    | 45.124*** | 3.1982*** | 1.350    | 51.740*** | 3.1801*** | 1.785    |  |
| Asym        | 41.537*** | 3.1976*** | 1.114    | 45.039*** | 3.1982*** | 1.351    | 51.571*** | 3.1802*** | 1.783    |  |
| Intra2      | 41.623*** | 3.1972*** | 1.101    | 45.124*** | 3.1974*** | 1.335    | 51.740*** | 3.1792*** | 1.784    |  |
| Asym-intra2 | 41.537*** | 3.1971*** | 1.097    | 45.039*** | 3.1974*** | 1.335    | 51.571*** | 3.1793*** | 1.785    |  |
| ,           |           |           |          |           | Dynamic   |          |           |           |          |  |
| Base        | 36.958*** | 3.1922*** | 1.101*** | 41.843*** | 3.1924*** | 1.222*** | 46.514*** | 3.1745*** | 1.772**  |  |
| Asym        | 36.939*** | 3.1919*** | 1.099*** | 41.818*** | 3.1923*** | 1.217*** | 46.459*** | 3.1746*** | 1.767**  |  |
| Intra2      | 36.924*** | 3.1888*** | 1.074*** | 41.710*** | 3.1886*** | 1.222*** | 46.193*** | 3.1708*** | 1.739*** |  |
| Asym-intra2 | 36.912*** | 3.1887*** | 1.072*** | 41.696*** | 3.1886*** | 1.219*** | 46.148*** | 3.1708*** | 1.737*** |  |

Asterisks denote the significance (\* 10%, \*\* 5%, and \*\*\* 1%) of a Diebold–Mariano test of equal predictive ability with respect to RM using the corresponding loss functions.

a commonly used naïve method of rolling means for intra-daily volumes in an out-of-sample forecasting exercise.<sup>7</sup>

While focused here on specific assets, the analysis can be extended to explore what other features are useful in the model for accommodating other asset classes. For individual companies, for example, we may want to see if some stock specific characteristics (e.g., market capitalization, debt-to-equity ratio or percentage of holdings by institutional investors) have a bearing on the characteristics of the estimated dynamics.

The CMEM can be used in other contexts in which intra-daily bins are informative of some periodic features (e.g., volatility, number of trades, average durations) together with overall dynamics having components at different frequencies. The periodic component can be more parsimoniously specified by recurring to some shrinkage estimation as in Brownlees and Gallo (2010). Multivariate extensions are also possible (following Cipollini, Engle, and Gallo 2009) by retrieving a richer price–volume dynamics in order to establish a relationship between volume and volatility that can be related to the flow of information at different frequencies, separating it from (possibly common) periodic components. Whether this has implications about how to trade dynamically to achieve an even closer approximation to VWAP is thus an open question.

*Received June 18, 2009; revised June 1, 2010; accepted June 8, 2010*.

### **REFERENCES**

Almgren, R., and N. Chriss. 2000. Optimal Execution of Portfolio Transactions. *Journal of Risk* 3: 5–39.

Andersen, T. G. 1996. Return Volatility and Trading Volume: An Information Flow Interpretation of Stochastic Volatility. *The Journal of Finance* 51: 169–204.

Berkowitz, S. A., D. E. Logue, and E. A. J. Noser. 1988. The Total Cost of Transactions on the NYSE. *The Journal of Finance* 43: 97–112.

Bertsimas, D., and A. W. Lo. 1998. Optimal Control of Execution Costs. *Journal of Financial Markets* 1: 1–50.

Bialkowski, J., S. Darolles, and G. Le Fol. 2008. Improving VWAP Strategies: A Dynamic Volume Approach. *Journal of Banking and Finance* 32: 1709–1722.

Bikker, J., L. Spierdijk, R. Hoevenaars, and P. J. van der Sluis. 2008. Forecasting Market Impact Costs and Identifying Expensive Trades. *Journal of Forecasting* 27: 21–39.

<sup>7</sup>While statistically significant, we leave as an open question whether our procedure may generate a significant economic gain in practice due to the improved volume forecast. This goes beyond the scope of this paper as a simulation should be based on a joint model of price and volume determination with market microstructure foundations. On the other hand, evaluating the procedure on an actual exchange would be limited by the lack of access to an algorithmic trading system and in any case by proprietary limitations in diffusing the results.

- Bollerslev, T., and E. Ghysels. 1996. Periodic Autoregressive Conditional Heteroskedasticity. *Journal of Business and Economic Statistics* 14: 139–151.
- Brownlees, C. T., and G. M. Gallo. 2006. Financial Econometric Analysis at Ultra-High Frequency: Data Handling Concerns. *Computational Statistics and Data Analysis* 51: 2232–2245.
- Brownlees, C. T., and G. M. Gallo. 2010. Shrinkage Estimation of Semiparametric Multiplicative Error Models. *International Journal of Forecasting* (forthcoming).
- Chordia, T., R. Roll, and A. Subrahmanyam. 2008. Why Has Trading Volume Increased? Technical report, UCLA.
- Christoffersen, P. F. 1998. Evaluation of Interval Forecasts. *International Economic Review* 39: 841–862.
- Cipollini, F., R. F. Engle, and G. M. Gallo. 2009. "Semiparametric Vector MEM." WP200903, Università degli Studi di Firenze, Dipartimento di Statistica "G. Parenti".
- Engle, R. F. 2002. New Frontiers for ARCH Models. *Journal of Applied Econometrics* 17: 425–446.
- Engle, R. F., and R. Ferstenberg. 2007. Execution Risk. *Journal of Portfolio Management* 33: 34–45.
- Engle, R. F., and G. M. Gallo. 2006. A Multiple Indicators Model for Volatility Using Intra-daily Data. *Journal of Econometrics* 131: 3–27.
- Engle, R. F., R. Ferstenberg, and J. Russell. 2006. Measuring and Modeling Execution Cost and Risk. Technical Report FIN-06-044, Stern, NYU.
- Engle, R. F., M. E. Sokalska, and A. Chanda. 2007. Forecasting Intraday Volatility in the US Equity Market. Multiplicative Component GARCH. Technical Report, North American Winter Meetings of the Econometric Society.
- Gallant, A. R. 1981. On the Bias in Flexible Functional Forms and an Essentially Unbiased Form: The Fourier Flexible Form. *Journal of Econometrics* 15: 211–245.
- Gallant, A. R. 1984. The Fourier Flexible Form. *American Journal of Agricultural Economics* 66: 204–208.
- Glosten, L. R., R. Jagannanthan, and D. E. Runkle. 1993. On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *The Journal of Finance* 48: 1779–1801.
- Gouriéroux, C. S., J. Jasiak, and G. Le Fol. 1999. Intra-day Market Activity. *Journal of Financial Markets* 2: 193–226.
- Härdle, W. K., N. Hautsch, and A. Mihoci. 2009. Modelling and Forecasting Liquidity Supply Using Semiparametric Factor Dynamics. Technical Report, 2009-044, CRC 649, Humboldt Universität.
- Hautsch, N. 2008. Capturing Common Components in High-Frequency Financial Time Series: A Multivariate Stochastic Multiplicative Error Model. *Journal of Economic Dynamics and Control* 32: 3978–4015.

- Madhavan, A. 2002. VWAP Strategies, pages 32–38. Investment Guides Series. Institutional Investor Inc.
- Martens, M., Y.-C. Chang, and S. J. Taylor. 2002. A Comparison of Seasonal Adjustment Methods when Forecasting Intraday Volatility. *Journal of Financial Research* 25: 283–299.
- Nelson, D. B. 1991. Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica* 59: 347–370.
- Nelson, D. B., and C. Q. Cao. 1992. Inequality Constraints in the Univariate GARCH Model. *Journal of Business and Economic Statistics* 10: 229–235.
- Newey, W. K., and D. McFadden. 1994. "Large Sample Estimation and Hypothesis Testing". In R. F. Engle and D. McFadden (eds.), *Handbook of Econometrics*, volume 4, chapter 36, pages 2111–2245. Amsterdam: Elsevier.
- Rabemananjara, R., and J. M. Zakoïan. 1993. Threshold ARCH Models and Asymmetries in Volatility. *Journal of Applied Econometrics* 8: 31–49.
- Taylor, N. 2002. The Economic and Statistical Significance of Spread Forecasts: Evidence from the London Stock Exchange. *Journal of Banking and Finance* 26: 795–818.
- White, H. 2006. "Approximate Nonlinear Forecasting Methods." In G. Elliott, C. W. J. Granger, and A. Timmermann (eds.), *Handbook of Economic Forecasting* , volume 1, chapter 9, pages 459–512. Amsterdam: Elsevier.
- Wooldridge, J. M. 1994. "Estimation and Inference for Dependent Processes." In R. F. Engle and D. McFadden (eds.), *Handbook of Econometrics*, volume 4, chapter 45, pages 2639–2738. Amsterdam: Elsevier.
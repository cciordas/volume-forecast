![](_page_0_Picture_0.jpeg)

### **Quantitative Finance**

![](_page_0_Picture_2.jpeg)

**ISSN: 1469-7688 (Print) 1469-7696 (Online) Journal homepage:<http://www.tandfonline.com/loi/rquf20>**

## **Relative volume as a doubly stochastic binomial point process**

**James McCulloch**

**To cite this article:** James McCulloch (2007) Relative volume as a doubly stochastic binomial point process, Quantitative Finance, 7:1, 55-62, DOI: [10.1080/14697680600969735](http://www.tandfonline.com/action/showCitFormats?doi=10.1080/14697680600969735)

**To link to this article:** <http://dx.doi.org/10.1080/14697680600969735>

![](_page_0_Picture_8.jpeg)

Full Terms & Conditions of access and use can be found at <http://www.tandfonline.com/action/journalInformation?journalCode=rquf20>

![](_page_1_Picture_2.jpeg)

# Relative volume as a doubly stochastic binomial point process

#### JAMES McCULLOCH\*

Department of Finance, University of Technology Sydney, Quay Street, Haymarket, Sydney 2000, Australia

(Received 4 November 2005: in final form 21 August 2006)

Relative intra-day cumulative volume is intra-day cumulative volume divided by final total volume. If intra-day cumulative volume is modeled as a Cox (doubly stochastic Poisson) point process, then using initial enlargement of filtration with the filtration of the Cox process enlarged by knowledge of final volume, it is shown that relative intra-day volume conditionally has a binomial distribution and is a novel generalization of a binomial point process: the doubly stochastic binomial point process. Re-scaling the intra-day traded volume to a relative volume between 0 (no volume traded) and 1 (daily trading completed) allows empirical intra-day volume distribution information for all stocks to be used collectively to estimate and identify the random intensity component of the doubly stochastic binomial point process and closely related Cox point process.

Keywords: Doubly stochastic binomial point process; Relative volume; Cox process; Initial enlargement of filtration; NYSE, New York Stock Exchange; VWAP

#### 1. Introduction

The Cox (doubly stochastic Poisson) point process (Cox 1955) has been used to model trade by trade market behaviour by a number of financial market researchers, including Engle and Russell (1998), Gouriéroux *et al.* (1999) and Rydberg and Shephard (2000).

If intra-day trade arrival is modeled as a Cox point process then relative volume is a doubly stochastic binomial point process.

The conditional marginal distribution of the doubly stochastic binomial point process, R(t;K) = N(t)/N(T); N(T) = K, is derived from the underlying Cox process by a straightforward application of the theory of initial enlargement of filtrations developed by Jeulin (1980), Jacod (1985) and Yor (1985). This distribution is shown to be equivalent to the conditional distribution of the underlying Cox process under a change of measure due to the natural filtration of the Cox process enlarged by knowledge of the final count of the Cox process, N(T) = K. If  $\Lambda(t)$  is the integrated intensity (see section 2 for definition) of the underlying Cox process, then the

$$P(R(t;K) = a \mid \Lambda(T)) = {K \choose aK} \left[ \frac{\Lambda(t)}{\Lambda(T)} \right]^{aK} \left[ 1 - \frac{\Lambda(t)}{\Lambda(T)} \right]^{K-aK},$$

$$K \in \mathbb{N}, \quad a \in \left\{ \frac{1}{K}, \frac{2}{K}, \dots, \frac{K-1}{K}, 1 \right\}, \quad t \in [0, T].$$

The result that the doubly stochastic binomial point process is conditionally binomially distributed implies that the unconditional distributions of R(t; K) are a binomial mixture (Daley and Vere-Jones 1988), where the mixing distributions are the finite dimensional (fi-di) distributions of the self-normalized integrated intensity of the underlying Cox process,

$$\Phi_t(s) = P\left(\frac{\Lambda(t)}{\Lambda(T)} \le s\right), \quad s \in [0, 1],$$

$$P(R(t; K) = a) = \int_0^1 \text{Binomial}(aK; K, s) \, d\Phi_t(s).$$

Re-scaling the intra-day traded volume to a relative volume between 0 (no volume traded) and 1 (daily trading completed) allows the use of stocks with different final trade counts to conveniently estimate the empirical unconditional (binomial mixture) fi-di distributions

conditional marginal distribution of R(t; K) can be formulated as follows:

<sup>\*</sup>Corresponding author. Email: james.mcculloch@uts.edu.au

of the doubly stochastic binomial point process in a twodimension histogram. This convenient statistical modeling is a major advantage of the doubly stochastic binomial point process analysis proposed in this paper.

Using the binomial mixture model, the histogram estimates of the fi-di distributions of R(t;K) are used to estimate the moments of the fi-di distributions of the self-normalized integrated intensity  $\Lambda(t)/\Lambda(T)$  of the underlying Cox process. It is readily shown that the expectation of the relative volume process R(t;K) is equivalent to the expectation of the self-normalized integrated intensity for stocks of all final trade counts:

$$\mathbb{E}[R(t;K)] = \mathbb{E}\left[\frac{\Lambda(t)}{\Lambda(T)}\right], \quad \forall K \in \mathbb{N}.$$

By definition, this is a model of the 'U'-shaped deterministic intra-day trading seasonality characteristic of equity markets (NYSE). [For further discussion of intra-day market seasonality, see Admati and Pfleiderer (1988), Brock and Kleidon (1992), and Coppejans et al. (2001).] The actual seasonality is readily estimated from the 2-D histogram. This is useful for Volume Weighted Average Price (VWAP) traders, who require a stochastic model of relative intra-day cumulative volume to implement risk-optimal VWAP trading strategies [VWAP price as a quality of execution measurement was first introduced by Berkowitz et al. (1988)].

Examination of the higher moments of the self-normalized integrated intensity,  $\Lambda(t)/\Lambda(T)$ , permits an analysis of simple models of the integrated intensity  $\Lambda(t)$  of the underlying Cox process. The integrated intensity is the compensator (Brémaud 1981) of the random counting measure N(t) of the Cox process, so  $\Lambda(t) = \mathbb{E}[N(t)]$  and, in particular,  $\Lambda(T) = \mathbb{E}[N(T)]$ . Since stocks have different final trade counts, the integrated intensity for stock i,  $\Lambda_i(t)$ , can be simply modeled using a baseline integrated intensity  $\mathbb{E}[\Lambda_0(T)] = 1$  in two ways:

- (i) The integrated intensity for stock i is modeled as a baseline integrated intensity scaled by a stock-specific constant,  $\alpha_i = \mathbb{E}[N(T)]$ , the expectation of final trade count,  $\Lambda_i(t) = \alpha_i \Lambda_0(t)$ ; or
- (ii) The integrated intensity for stock i is modeled as a baseline integrated intensity where the time parameter is multiplied by a stock-specific constant,  $\alpha_i = \mathbb{E}[N(T)]$ , the expectation of final trade count,  $\Lambda_i(t) = \Lambda_0(\alpha_i t)$ .

In the first case, the self-normalized integrated intensity is the same for all stocks irrespective of final trade count and therefore all higher moments of the self-normalized integrated intensity will be the same irrespective of trade count. This is empirically tested by dividing stocks into trade count bands and calculating the variances of the self-normalized integrated intensities for different trade counts. The variances of the self-normalized integrated intensities for stocks of different trade count are different and case (i) is rejected (see section 7 for details).

This is an interesting insight since it suggests that the integrated intensities of different stocks are unique only up to a time scaling proxied by final trade count. Since the integrated intensity can be viewed as a random market clock (Barndorff-Nielsen and Shephard 1998) or subordinator, it is particularly interesting that this subordinator is scaled by trade count since this appears to confirm the results of Ané and Geman (2000), who also showed that the market stochastic clock or subordinator can be proxied by trade count.

#### 2. Trade arrival as a Cox process

A point process on the time index is a set of discrete events that can be ordered in time,  $t_1 < t_2 < \cdots < t_m <$ . If the time ordering is strict then there are no co-occurring points and the point process is *simple*. A time indexed point process is formally defined by its *random counting measure*  $N(\cdot)$ . This is a random measure defined on a probability space  $(\Omega, \mathcal{F}, P)$  that maps subsets of a Borel algebra of non-negative real numbers onto the set of non-negative integers:

$$N(B,\omega) \to I, \quad I \in \mathbb{Z}^+, B \in \mathcal{B}(\mathbb{R}^+), \omega \in \Omega.$$

The value of the random counting measure on an interval including zero  $N([0, t], \omega)$  is generally of interest and the notation will be truncated to N(t) where this is clear. Likewise, the formal probability space notation will be suppressed wherever it is not explicitly required.

Trade arrivals can be modeled as a point process with the cumulative trade count defined by the random counting measure N(t). In particular, trade arrivals can be modeled as a Cox process, a simple point process directed by a stochastic intensity. The following limit [Random integrated intensities exist that do not have an integral representation (the intensity process does not exist), see Segall and Kailath (1975). In all cases considered in this paper, an intensity process will be defined for the point process.] defines the *intensity process* of the Cox process:

$$\lambda(t) = \lim_{\Delta \to 0^+} \frac{P[N(t+\Delta) - N(t) > 0 \mid \mathcal{H}_t^{\Lambda}]}{\Delta}.$$

The intensity process of a Cox process,  $\lambda(t)$ , is a non-negative Lebesque integrable stochastic process adapted to the filtration  $\mathcal{H}_t^{\Lambda}$  at time t. This filtration may be larger than the natural filtration (internal history) of the point process,  $\mathcal{F}_t^{\mathrm{N}} \subseteq \mathcal{H}_t^{\Lambda}$ . The integral of the intensity process is the *integrated intensity*:

$$\Lambda(t) = \int_0^t \lambda(s) \, \mathrm{d}s. \tag{1}$$

The integrated intensity is a  $\mathcal{H}_t^{\Lambda}$ -compensator of the random counting measure N(t) (Brémaud 1981):

$$N(t) - \Lambda(t) = M_t$$
, a  $\mathcal{H}_t$ -martingale.

Conditional on a realization of the random integrated intensity, the Cox process is a Poisson point process (see Daley and Vere-Jones (2002), definition 6.2.I):

$$P(N(t) = k \mid \Lambda(t)) = \frac{\Lambda(t)^k}{k!} e^{-\Lambda(t)}.$$

#### 3. The doubly stochastic binomial point process

The intra-day trade count can be scaled to between 0 and 1 by the simple expedient of dividing the intra-day count (N(t) = aK) by the final trade count (N(T) = K). This defines the self-normalized trade count process R(t; K), which is formally named the *random relative counting measure*:

$$R(t;K) = \frac{N(t)}{N(T)} = \frac{aK}{K} = a, \quad a \in \left\{0, \frac{1}{K}, \dots, \frac{K-1}{K}, 1\right\}.$$

This section formalizes the relationship of the random relative counting measure R(t;K) and the self-normalized integrated intensity,  $\Lambda(t)/\Lambda(T)$ . It is unsurprising that the relationship between these measures is described by a binomial point process directed by the self-normalized integrated intensity. This point process is related to a binomial point process in a way analogous to the relationship between a Cox point process and the Poisson point process.

Theorem 3.1 The distribution of the doubly stochastic binomial point process conditional on the integrated intensity measure of the underlying Cox process at time T,  $\Lambda(T)$  and final trade count N(T) = K is a binomial distribution. The doubly stochastic binomial point process is also shown to be semi-martingale:

$$P(R(t;K) = a \mid \Lambda(T)) = P(N(t) = aK \mid N(T) = K, \Lambda(T))$$

$$= {K \choose aK} \left[ \frac{\Lambda(t)}{\Lambda(T)} \right]^{aK} \left[ 1 - \frac{\Lambda(t)}{\Lambda(T)} \right]^{K-aK},$$

$$K \in \mathbb{N}, a \in \left\{ 0, \frac{1}{K}, \dots, \frac{K-1}{K}, 1 \right\}, \ t \in [0, T].$$

$$(2)$$

*Proof* The proof is a straightforward application of the theory of initial enlargement of filtrations. For further reading, see Jeulin (1980), Jacod (1985), Yor (1985), Föllmer and Imkeller (1993), Amendinger (1999) and Gasbarra *et al.* (2004) with a summary on financial applications by Baudoin (2002).

Let  $(\Omega, \mathcal{F}, \mathbb{F}, P)$  be a filtered probability space with filtration  $\mathbb{F} = (\mathcal{F}_t)_{t \geq 0}$  satisfying the usual conditions. Assume  $\xi$  is a random variable on  $(\Omega, \mathcal{F})$  taking values in a Polish space  $(X, \mathcal{X})$ . The  $\sigma$ -algebra of  $\xi$  is non-trivial with  $\sigma(\xi) \not\subseteq \mathcal{F}_0$ . The following is subject to some non-restrictive formal conditions to exclude pathological cases that do not arise in the context of this paper (Jacod 1985).

Let  $t \in \mathbb{R}^+$  and  $\Phi_t(x)$  be the conditional distribution of random variable  $\xi$  given information  $\mathcal{F}_t$ :

$$\Phi_t(x) = P(\xi \in dx \mid \mathcal{F}_t), \quad x \in X.$$

Then by Jacod (1985) there exists a process  $(\pi_t(x))_{t\geq 0}$  defined by the normalization of the conditional distribution with the unconditional distribution  $\Phi(x)$  of the random variable  $\xi$  which is a  $(P, \mathcal{F})$ -martingale with  $\pi_0(x) = 1$ :

$$\Phi(x) = P(\xi \in dx),$$

$$\pi_t(x) = \frac{\Phi_t(x)}{\Phi(x)}, \quad \mathbb{E}[\pi_t(x)|\pi_s(x)] = \pi_s(x), \quad \forall s < t, \quad \pi_0(x) = 1.$$

The martingale  $\pi_t(x)$  is the (see Baudoin (2002), p. 47) density process for the measure change to the disintegrated probability measure  $P^x(.) = P(. \mid \xi = x)$ :

$$P^{x} = \pi_{t}(x)P.$$

This disintegrated probability measure is precisely what is required to calculate the probability of  $P(N(t) = aK \mid N(T) = K, \Lambda(T))$  given the probability of  $P(N(t) = aK \mid \Lambda(T))$ .

Note that the distribution of the Cox process conditional on  $\Lambda(t)$  is the same as the distribution conditioned on  $\Lambda(T)$ :

$$P(N(t) = aK \mid \Lambda(T)) = P(N(t) = aK \mid \Lambda(t))$$
$$= \frac{\Lambda(t)^{aK}}{(aK)!} e^{-\Lambda(t)}.$$

Then using the disintegrated probability measure and replacing  $\xi = x$  with N(T) = K:

$$P(R(t;K) = a \mid \Lambda(T)) = P(N(t) = aK \mid N(T) = K, \Lambda(T))$$

$$= P^{K}(N(t) = aK \mid \Lambda(T)),$$

$$P^{K}(\cdot) = P(\cdot \mid N(T) = K)$$

$$= \pi_{t}(K)P(N(t) = aK \mid \Lambda(T))$$

$$= \pi_{t}(K)\frac{\Lambda(t)^{aK}}{(aK)!}e^{-\Lambda(t)}.$$

Therefore, finding the conditional distribution of R(t; K) reduces to finding the change of measure martingale  $\pi_t(x)$ .

The distribution of the Cox process N(T) conditional on the  $\sigma$ -algebra  $\mathcal{F}_t$  is equivalent to the distribution of N(T) conditional on the outcome of N(t):

$$\begin{split} \Phi_t(x) &= P(N(T) = x \mid \Lambda(T), \mathcal{F}_t) \\ &= P(N(T) = x \mid \Lambda(T), N(t)) \\ &= \exp[-(\Lambda(T) - \Lambda(t))] \frac{(\Lambda(T) - \Lambda(t))^{x - N(t)}}{(x - N(t))!}. \end{split}$$

The distribution of the Cox process N(T) is given by:

$$\Phi(x) = P(N(T) = x \mid \Lambda(T))$$
$$= \exp[-\Lambda(T)] \frac{\Lambda(T)^{x}}{x!}.$$

So, the change of measure martingale for the Cox process is:

$$\pi_t(x) = \frac{\Phi_t(x)}{\Phi(x)} = \exp[\Lambda(t)] \frac{(\Lambda(T) - \Lambda(t))^{x - N(t)}}{\Lambda(T)^x} \frac{x!}{(x - N(t))!}$$

Therefore, the distribution of R(t; K) conditional on  $\Lambda(T)$  is:

$$P(R(t;K) = a \mid \Lambda(T)) = P(N(t) = aK \mid N(T) = K, \Lambda(T))$$

$$= P^{K}(N(t) = aK \mid \Lambda(T)),$$

$$P^{K}(.) = P(. \mid N(T) = K)$$

$$= \pi_{t}(K)P(N(t) = aK \mid \Lambda(T))$$

$$= \exp[\Lambda(t)]$$

$$\times \frac{(\Lambda(T) - \Lambda(t))^{K-aK}}{\Lambda(T)^{K}} \frac{K!}{(K - aK)!}$$

$$\times \exp[-\Lambda(t)] \frac{\Lambda(t)^{aK}}{(aK)!}$$

$$= {K \choose aK} \left[ \frac{\Lambda(t)}{\Lambda(T)} \right]^{aK} \left[ 1 - \frac{\Lambda(t)}{\Lambda(T)} \right]^{K-aK},$$

$$K \in \mathbb{N}, a \in \left\{ \frac{1}{K}, \frac{2}{K}, \dots, \frac{K-1}{K}, 1 \right\}, t \in [0, T].$$

Finally, by Jacod's (1985) celebrated theorem, Hypothèse (H'), a  $\mathcal{F}_t$  semi-martingale is a  $\mathcal{G}_t = \mathcal{F}_t \vee \sigma(\xi)$  semi-martingale (in the above case,  $\sigma(\xi) \equiv \sigma(N(T))$  and  $\mathcal{F}_t$  is the natural filtration of the Cox process) under non-restrictive formal conditions. Since a Cox process with an integrable intensity process is semi-martingale, the doubly stochastic binomial point process is also semi-martingale.

#### 4. The unconditional distribution of R(t; K)

The unconditional distribution of a point process directed by a stochastic measure can be modeled as a mixture distribution of the point process (binomial) with the marginal distribution of the directing measure as the mixing distribution (see p. 235 of Daley and Vere-Jones (1988) for a full discussion). This implies that the unconditional distribution of the doubly stochastic binomial point process R(t; K) is a mixture distribution of the distribution of the self-normalized integrated intensity at time t,  $\Psi_t(s)$  (the mixing distribution), and the binomial distribution of the probability of aK trades given K final trades,  $\mathcal{B}(aK \mid K, s)$ . Thus, if  $\Theta_t(aK \mid K)$  is the probability of aK

trades at time t given K trades at time T, then this distribution is the following mixture distribution:

$$P(R(t;K) = a) = \Theta_t(aK \mid K)$$

$$= \int_0^1 \mathcal{B}(aK \mid K, s) \Psi_t(s) \, \mathrm{d}s,$$

$$\mathcal{B}(aK \mid K, s) = \binom{K}{aK} s^{aK} (1 - s)^{K - aK},$$

$$\Phi_t(s) = P\left(\frac{\Lambda(t)}{\Lambda(T)} \le s\right), \quad \forall s \in [0, 1],$$

$$\Psi_t(s) = \frac{\mathrm{d}\Phi_t(s)}{\mathrm{d}s},$$

$$K \in \mathbb{N}, \ a \in \left\{\frac{1}{K}, \frac{2}{K}, \dots, \frac{K - 1}{K}, 1\right\}, \ t \in [0, T].$$

## 4.1. The moments of the self-normalized integrated intensity

Since the distribution of relative intra-day trade count is a binomial mixture distribution, it is easy to extract the moments of the mixing distribution by using the following relation to relate moments of the relative trade count (observed data) to the moments of the mixing distribution:

$$\sum_{k=0}^{K} \left(\frac{k}{K}\right)^n \Theta_t(k \mid K) = \frac{1}{K^n} \int_0^1 \left[\sum_{k=0}^{K} k^n \mathcal{B}(k \mid K, s)\right] \Psi_t(s) \, \mathrm{d}s.$$
(3)

The first four non-central moments of the mixing distribution can be calculated from the non-central moments of the relative trade count (observed binned data) distribution and lower non-central moments of the mixing distribution. Let  $\delta_i$  be the *i*th non-central moment of the (observed binned data) relative trade count distribution,  $\Theta_t(k \mid K)$ , and  $\lambda_i$  be the *i*th non-central moment for the mixing distribution  $\Psi_t(s)$ , with final trade count K. Then the first four non-central mixing distribution moments are:

(i) 
$$\lambda_1 = \delta_1,$$
 (ii) 
$$\lambda_2 = \frac{K\delta_2 - \lambda_1}{K - 1},$$

(iii) 
$$\lambda_3 = \frac{K^2 \delta_3 - 3K \lambda_2 + 3\lambda_2 - \lambda_1}{K^2 - 3K + 2},$$

(iv) 
$$\lambda_4 = \frac{K^3 \delta_4 - 6K^2 \lambda_3 + 18K \lambda_3 - 12\lambda_3 - 7K \lambda_2 + 7\lambda_2 - \lambda_1}{K^3 - 6K^2 + 11K - 6}.$$

#### 4.2. Moments with different trade counts

This section shows that although self-normalized integrated intensity moments higher than 1 are dependent on final trade count K, a scaling can be introduced so that moments can be extracted from a collection of stocks with different trade counts.

The distribution of trade counts of an observed collection of stocks can be represented by a discrete distribution,  $z_n = Pr(K = n), n \in \{1, ..., B\}$ , where B is the maximum number of trades (B = 5022 in the data):

$$\sum_{i=1}^{B} z_i = 1.$$

Then the non-central moment of the relative trade count of an observed collection of stocks with different final trade counts is the weighted sum of the moments for each different final trade count:

$$\begin{split} &\sum_{K=1}^{B} z_K \sum_{k=0}^{K} \left(\frac{k}{K}\right)^n \Theta_t(k \mid K) \\ &= \sum_{K=1}^{B} z_K \int_0^1 \left[ \sum_{k=0}^{K} \left(\frac{k}{K}\right)^n \mathcal{B}(k \mid K, s) \right] \Psi_t(s) \, \mathrm{d}s. \end{split}$$

As an example, the second non-central moment of an observed collection of stocks with different final trade counts is used to calculate the second non-central moment of the mixing distribution:

$$\sum_{K=1}^{B} z_{K} \sum_{k=0}^{K} \left(\frac{k}{K}\right)^{2} \Theta_{t}(k \mid K)$$

$$= \sum_{K=1}^{B} z_{K} \int_{0}^{1} \left[\sum_{k=0}^{K} \left(\frac{k}{K}\right)^{2} \mathcal{B}(k \mid K, s)\right] \Psi_{t}(s) \, ds$$

$$= \sum_{K=1}^{B} z_{K} \int_{0}^{1} \left(\frac{s}{K} + \frac{(K-1)s^{2}}{K}\right) \Psi_{t}(s) \, ds$$

$$= \sum_{K=1}^{B} \frac{z_{K}}{K} \int_{0}^{1} s \Psi_{t}(s) \, ds + \int_{0}^{1} s^{2} \Psi_{t}(s) \, ds$$

$$- \sum_{K=1}^{B} \frac{z_{K}}{K} \int_{0}^{1} s^{2} \Psi_{t}(s) \, ds. \tag{4}$$

Since the trade count distribution of an observed collection of stocks is known, the following substitution is convenient:

$$V_n = \sum_{K=1}^B \frac{z_K}{K^n}.$$

Substituting the above into equation (4):

$$\begin{split} V_0 \sum_{k=0}^K & \left(\frac{k}{K}\right)^2 \Theta_t(k \mid K) = V_1 \int_0^1 s \Psi_t(s) \, \mathrm{d}s + \int_0^1 s^2 \Psi_t(s) \, \mathrm{d}s \\ & - V_1 \int_0^1 s^2 \Psi_t(s) \, \mathrm{d}s. \end{split}$$

Again, let  $\delta_i$  be the *i*th non-central moment of the (observed binned data) relative trade count distribution,  $\Theta_t(k \mid K)$ , and  $\lambda_i$  be the *i*th non-central moment for the mixing distribution (noting that  $V_0 = 1$ ):

$$\delta_2 = V_1 \lambda_1 + \lambda_2 - V_1 \lambda_2.$$

So the first four non-central moments of the mixing distribution calculated using a collection of stocks with different trade counts can be written using  $V_n$  constants (which can be readily pre-computed):

(i) 
$$\lambda_1 = \delta_1,$$
 (ii) 
$$\lambda_2 = \frac{\delta_2 - V_1 \lambda_1}{1 - V_2},$$

(iii) 
$$\lambda_3 = \frac{\delta_3 - \lambda_1 V_2 - \lambda_2 (3V_1 - 3V_2)}{1 - 3V_1 + 2V_2},$$

(iv) 
$$\lambda_4 = \frac{\delta_4 - \lambda_3 (6V_1 - 18V_2 + 12V_3) - \lambda_2 (7V_2 - 7V_3) - \lambda_1 V_3}{1 - 6V_1 + 11V_2 - 6V_3}.$$

The machinery is now available to extract the moments of the mixing distribution (the moments of the fi-di distributions of the self-normalized integrated intensity) by observing the relative trade counts of a collection of stocks.

#### 5. Data—observing relative trade counts

New York Stock Exchange (NYSE) trade data from the TAQ database were used to collect relative trade count data of all stocks that traded from 1 June 2001 to 31 August 2001 [A total of 62 trading days; 3 July 2001 (half-day trading) and 8 June 2001 (NYSE computer malfunction delayed market opening) were excluded from the analysis.] for a total of 203 158 relative trade count sample paths for all stocks (figure 1). The relative trade count data were collected in a 2-D histogram with time in minutes in the x axis and relative volume in the y axis. Two histogram sizes were used, a  $391 \times 253$  and  $391 \times 1261$  histogram (figure 2).

These histogram sizes were chosen for the following reasons. In the time x axis, the NYSE is open from 9:30 to 16:00, a total of 390 minutes. It is natural to collect sample path data each minute from precisely 9:31 to 16:00, giving 390 data points in the time axis. This 390 points is augmented with the initial state of the market at 9:30 when no trades have executed, giving a total of 391 data points in the time x axis.

In the relative volume y axis (proportion of final trading completed) an examination of the properties of the sample path distributions indicated that approximately 250 bins was appropriate (see p. 22 of Simonoff (1996) for details).

However, if 250 bins were used, then stocks with n final trades and k intra-day trades at some time during the day would have an intra-day executed proportional trade count of k / n. This fraction would fall on the bin boundary of a 250 bin grid if n and 250 have a common divisor. On exact boundaries, a problem arises caused by rounding a floating point y value (proportion executed) to an integer number of bins. The actual integer value generated (which bin) becomes uncertain and is dependent on the hardware/ software implementation of the particular computer floating point algorithm. The solution was to use a prime number of bins (no common divisor), 251, so that the

![](_page_6_Figure_3.jpeg)

Figure 1. The frequency count of stocks with different daily trade counts. The counts were generated for stocks trading on the NYSE between 9:30 and 16:00 from 1 June 2001 to 31 August 2001.

proportional trade count of k / n never falls on exact bin boundaries. The total number of bins in the y axis is then augmented by including the two end conditions of 'no trades executed' and 'no further trades executed' to give a total of 253 y axis bins.

The higher resolution histogram, 391 1261 (prime 1259 þ 2 end conditions), was used to check for any sampling artifacts in the relative trade count data that the 391 253 bin matrix may have introduced. No sampling artifacts were found.

#### 6. The self-normalized integrated intensity moments

#### 6.1. The mean is intra-day seasonality

6.1.1. The self-normalized integrated intensity mean— E[G(t)/G(T )]. The mean trading intensity is examined for stocks split into eight different trade count bands of approximately 30 000 relative volume trajectories in each band (figure 3). The trading intensity of low trade count stocks is boosted by market open because these stocks are much more likely to trade at market open than at other times. Conversely, mean trading intensity is reduced on market open for high trade count stocks (stocks trading 401 trades per day or more) because these stocks only experience one trade during the NYSE market opening period, whereas these stocks would expect to trade several times during the same period (NYSE market open is approximately 4–5 min).

![](_page_6_Figure_10.jpeg)

Figure 2. Binned relative trade count trajectories in a 391 253 histogram. This is an estimation of the unconditional finitedimensional distributions of Rðt; KÞ and is a binomial mixture distribution. The binned trajectories were generated from stocks on the NYSE that traded at least 50 times between 9:30 and 16:00 from 1 June 2001 to 31 August 2001. The histogram is truncated at 3000 and the truncation artifact can be seen near t ¼ 0 and t ¼ 391.

#### 6.2. Variance is scaled by trade count

**6.2.1.** The self-normalized integrated intensity variance— $Var[\Lambda(t)/\Lambda(T)]$ . The variance of the self-normalized integrated intensity is dependent on trade count with higher trade count stocks having lower self-normalized integrated intensity variance. The relationship between trade count and variance is approximately empirically scaled by the inverse square root of trade count (see figure 4 and equation (5)). Research by the author suggests that the scaling of the variances of the self-normalized integrated intensities of stocks with different trade counts may be related to the Hurst exponent (Embrechts and Maejima 2002) of the integrated intensity of the underlying Cox process of trade arrivals:

$$\operatorname{Var}\left[\frac{\Lambda(t)}{\Lambda(T)}\right] \propto \frac{1}{\sqrt{K}}.\tag{5}$$

#### 7. A time scaled model of the integrated intensity

An examination of the mean and variance of the self-normalized integrated intensity,  $\Lambda(t)/\Lambda(T)$ , permits an analysis of simple models of the integrated intensity  $\Lambda(t)$  of the underlying Cox process.

The integrated intensity is the compensator (Brémaud 1981) of the random counting measure N(t) of the Cox process, so  $\Lambda(t) = \mathbb{E}[N(t)]$  and, in particular,  $\Lambda(T) = \mathbb{E}[N(T)]$ . Since stocks have different final trade counts the integrated intensity for stock i,  $\Lambda_i(t)$ , can be simply modeled using a baseline integrated intensity  $\mathbb{E}[\Lambda_0(T)] = 1$  in two ways:

- (i) The integrated intensity for stock i is modeled as a baseline integrated intensity scaled by a stock-specific constant,  $\alpha_i = \mathbb{E}[N_i(T)]$ , the expectation of final trade count.  $\Lambda_i(t) = \alpha_i \Lambda_0(t)$ ; or
- (ii) The integrated intensity for stock i is modeled as a baseline integrated intensity where the time parameter is multiplied by a stock-specific constant,  $\alpha_i = \mathbb{E}[N_i(T)]$ , the expectation of final trade count.  $\Lambda_i(t) = \Lambda_0(\alpha_i t)$ .

In the first case, the self-normalized integrated intensity is the same for all stocks irrespective of final trade count and therefore all higher moments of the self-normalized integrated intensity will be the same irrespective of trade count:

$$\frac{\Lambda_i(t)}{\Lambda_i(T)} = \frac{\alpha_i \Lambda_0(t)}{\alpha_i \Lambda_0(T)} = \frac{\Lambda_0(t)}{\Lambda_0(T)}.$$

However, the variance for the self-normalized integrated intensity is different for different trade counts (approximately proportional to  $1/\sqrt{K}$ ) and therefore the first case is rejected by the evidence in figure 5. This supports case (ii), that stock i may be simply modeled as a baseline integrated intensity where the

time parameter is multiplied by a stock-specific constant,  $\alpha_i = \mathbb{E}[N_i(T)]$ , the expectation of final trade count:

$$\Lambda_i(t) = \Lambda_0(\alpha_i t) = \int_0^{\alpha_i t} \lambda_0(s) \, \mathrm{d}s = \alpha_i \int_0^t \lambda_0(\alpha_i r) \, \mathrm{d}r.$$

Thus the simple baseline intensity model is consistent with the integrated intensity (the random market clock) being scaled by trade count, providing indirect confirmation of the results of Ané and Geman (2000).

#### 8. Summary

If trade arrival is modeled as a Cox point process, then relative volume is modeled as a doubly stochastic binomial point process. The conditional distribution of this point process is shown to be binomial using an enlargement of filtration on the underlying Cox process where the filtration is enlarged by knowledge of the trade

![](_page_7_Figure_19.jpeg)

Figure 3. The mean for the self-normalized integrated intensity for stocks within different trade count bands. To aid visual comparison, the constant time component has been subtracted,  $\mathbb{E}[\Lambda(t)/\Lambda(T)] - (t/T)$  (all of these means are actually monotonically increasing from 0 to 1). Mean market trading intensity is boosted by market opening for low trade count stocks and attenuated by market opening for high trade count stocks.

![](_page_7_Figure_21.jpeg)

Figure 4. The variance of the self-normalized integrated intensity for stocks within different trade count bands. Note that trade count has a significant effect on self-normalized integrated intensity variance.

![](_page_8_Figure_2.jpeg)

Figure 5. Graph showing the variance of the self-normalized integrated intensity for different trade count bands scaled by the inverse of the relationship in equation (52) (scaled by  $\sqrt{K}$ ).

final count, N(T) = K. The doubly stochastic binomial point process is also shown to be a semi-martingale:

$$P(R(t;K) = a \mid \Lambda(T)) = {K \choose aK} \left[ \frac{\Lambda(t)}{\Lambda(T)} \right]^{aK} \left[ 1 - \frac{\Lambda(t)}{\Lambda(T)} \right]^{K-aK},$$

$$K \in \mathbb{N}, \ a \in \left\{ 0, \frac{1}{K}, \dots, \frac{K-1}{K}, 1 \right\}, \ t \in [0, T].$$

The unconditional probability distribution of relative volume is a binomial mixture distribution where the mixing distributions are the finite-dimensional distributions of the self-normalized integrated intensity of the underlying Cox process:

$$\Phi_t(s) = P\left(\frac{\Lambda(t)}{\Lambda(T)} \le s\right), \ s \in [0, 1],$$

$$P(R(t; K) = a) = \int_0^1 \text{Binomial}(aK; K, s)\Phi_t(ds).$$

Relative volume is scaled to the unit interval. Thus the empirical finite-dimensional distributions of Relative Volume R(t; K) for the NYSE are simply and readily collected in a 2-D histogram. By modeling R(t; K) as a binomial mixture distribution, the moments of the self-normalized integrated intensity are calculated from these empirical fi-di distributions. The expectation of the self-normalized integrated intensity is the same as the expectation of relative volume and the deterministic intra-day trading variation—the classic 'U' shape found in equity markets. The variances of the self-normalized integrated intensity  $\Lambda(t)/\Lambda(T)$  are scaled approximately  $1/\sqrt{K}$  for stocks with final trade count K. This implies that the integrated intensity for stock i,  $\Lambda_i(t)$ , can be simply modeled as a baseline integrated intensity  $\mathbb{E}[\Lambda_0(T)] = 1$ , where the time parameter is multiplied by a stock-specific constant,  $\alpha_i = \mathbb{E}[N_i(T)]$ , the expectation of final trade count,  $\Lambda_i(t) = \Lambda_0(\alpha_i t)$ .

#### References

Admati, A. and Pfleiderer, P., A theory of intraday patterns: volume and price variability. *Rev. finan. Stud.*, 1988, 1, 3–40. Amendinger, J., Initial enlargement of filtrations and additional information in financial markets. Berlin Technical University,

Berlin, 1999.

Ané, T. and Geman, H., Order flow, transaction clock, and normality of asset returns. *J. Finan.*, 2000, **55**, 2259–2284.

Barndorff-Nielsen, O.E. and Shephard, N., Aggregation and model construction for volatility models. Unpublished, 1998. Available online at: http://www.nuff.ox.ac.uk.

Baudoin, F., Modelling Anticipations on Financial Markets, Paris-Princeton Lectures on Mathematical Finance 2002, Lecture Notes in Mathematics, vol. 1814, pp. 43–92, 2002 (Springer: New York).

Berkowitz, S., Logue, D. and Noser, E., The total cost of transactions on the NYSE. *J. Finan.*, 1988, **43**, 97–112.

Brémaud, P., Point Processes and Queues: Martingale Dynamics, 1981 (Springer: New York).

Brock, W. and Kleidon, A., Periodic market closure and trading volume: a model of intraday bids and asks. *J. econ. Dynam. Contr.*, 1992, **16**, 451–490.

Coppejans, M., Domowitz, I. and Madhavan, A., Liquidity in an automated auction. Working Paper, 2001.

Cox, D., Some statistical methods connected with series of events (with discussion). J. R. statist. Soc. B, 1955, 17, 129–164.

Daley, D. and Vere-Jones, D., An Introduction to the Theory of Point Processes, 1988 (Springer: New York).

Daley, D. and Vere-Jones, D., An Introduction to the Theory of Point Processes, vol. 1: Elementary Theory and Methods, 2002 (Sringer: New York).

Embrechts, P. and Maejima, M., Selfsimilar Processes, 2002 (Princeton University Press).

Engle, R. and Russell, J., The autoregressive conditional duration model. *Econometrica*, 1998, **66**, 1127–1163.

Föllmer, H. and Imkeller, P., Anticipation cancelled by a Girsanov transformation: a paradox on Weiner space. *Ann. Inst. Henri Poincaré*, 1993, **29**, 569–586.

Gasbarra, D., Valkeila, E. and Vostrikova, L., Enlargement of filtration and additional information in pricing models: a Bayesian approach, 2004. Available online at: http://math.tkk.fi/reports/a476.pdf.

Gouriéroux, C., Jasiak, J. and Le Fol, G., Intra-day market activity. *J. finan. Mkts*, 1999, **2**, 193–216.

Jacod, J., Grossissement initial, hypothése et théoréme de Girsanov, séminaire de calcul stochastique 1982/83. In Lecture Notes in Mathematics, vol. 1118, pp. 15–35, 1985 (Springer: New York).

Jeulin, T., Semi-martingales et grossissement d'une filtration. In Lecture Notes in Mathematics, vol. 920, 1980 (Springer: New York).

Rydberg, T. and Shephard, N., BIN models for trade-by-trade data. Modelling the number of trades in a fixed interval of time. Unpublished. Available online at: http://www.nuff.ox.ac.uk.

Segall, A. and Kailath, T., The modeling of randomly modulated jump processes. *IEEE Trans. inf. Theory*, 1975, **IT-21**, 135–143.

Simonoff, J., Smoothing Methods in Statistics, 1996 (Springer: New York).

Yor, M., Grossissement de filtrations et absolue continuité de noyaux. In *Lecture Notes in Mathematics*, vol. 1118, pp. 6–14, 1985 (Springer: New York).
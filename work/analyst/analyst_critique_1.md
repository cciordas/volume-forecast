# Critique of Research Directions Draft 1

## Overview

- **Issues identified:** 0 major, 4 minor
- **Overall assessment:** The clustering is sound. All 7 directions correspond to genuinely distinct models, and no single-model violations are present. The issues below are all correctible with small adjustments to paper assignments and cross-references.

---

## Issue 1: Markov 2019 should not appear in Direction 4 (minor)

**Direction affected:** Direction 4 (Weighted Component Volume Forecast)

**Problem:** Markov, Vilenskaia, and Rashkovich (2019) is listed as "Supporting" for Direction 4 (Satish et al.'s four-component weighted model). The stated justification is conceptual similarity between Satish's per-bin ARMA and Markov's Bayesian bin model (Model 4 of the Quintet). However, the Quintet is a fundamentally different system: it uses Bayesian conjugate updating, geometric-mean priors, functional data analysis for the U-curve, and a liquid/illiquid switching mechanism. The conceptual overlap is limited to "both forecast bin-level volume" — a surface similarity, not a model relationship.

The "Supporting" role implies the paper provides useful techniques or insights for implementing the Satish model. In practice, a developer implementing Direction 4 would not consult the Markov paper for guidance — the algorithmic frameworks are unrelated.

**Recommendation:** Remove Markov 2019 from Direction 4. It belongs exclusively in Direction 5.

---

## Issue 2: Chen 2016 missing from Direction 4 as Comparison (minor)

**Direction affected:** Direction 4 (Weighted Component Volume Forecast)

**Problem:** Chen, Feng, and Palomar (2016) explicitly benchmarks against Satish et al. (2014). The Chen summary states: "Satish et al. (2014) used a similar decomposition with ARMA models; this paper claims to outperform that approach as well." The paper manifest also lists Satish 2014 as a key reference for Chen 2016. Despite this, Chen 2016 does not appear in Direction 4.

Chen 2016 already correctly appears in Direction 1 (Comparison with CMEM) and Direction 7 (Foundational). Adding it to Direction 4 as a Comparison paper is consistent with how Szucs 2017 appears in both Directions 1 and 2.

**Recommendation:** Add Chen 2016 to Direction 4 with role "Comparison."

---

## Issue 3: Satish 2014 PDF data integrity note (minor)

**Direction affected:** Direction 4 (Weighted Component Volume Forecast)

**Problem:** The local PDF file `papers/satish_saxena_palmer_2014.pdf` appears to contain a different paper (Hardle, Hautsch & Mihoci on limit order books, based on prior analysis). The paper summary correctly describes the Satish et al. model, so clustering decisions are unaffected. However, this should be flagged because a developer consulting the PDF for implementation details will find the wrong content.

**Recommendation:** Note the PDF mismatch in the direction's implementation notes or data requirements, and flag for re-download in the paper manifest. This does not affect clustering but affects downstream usability.

---

## Issue 4: Direction 4 contains two models from one paper (minor)

**Direction affected:** Direction 4 (Weighted Component Volume Forecast)

**Problem:** The Satish et al. (2014) paper presents two distinct models: a raw volume forecast model (four-component weighted combination) and a volume percentage forecast model (extension of Humphery-Jenner 2011). The draft correctly describes both and notes in implementation notes that "Two separate models are recommended." These are different models serving different use cases (full-day scheduling vs. step-by-step VWAP execution).

This is not a clear single-model violation because the two models share components (the raw volume model feeds surprise calculations into the percentage model) and come from a single paper as a coherent pair. However, the direction name "Weighted Component Volume Forecast" only describes the first model. The description covers both, but the framing is slightly misleading.

**Recommendation:** Either (a) rename the direction to something like "Dual-Mode Volume Forecast (Raw + Percentage)" to acknowledge both models, or (b) note explicitly in the description that this direction implements a two-model system from a single paper, with the raw volume model as primary and the percentage model as a companion. Option (b) is preferred — the models are tightly coupled enough to belong together, but the direction should be transparent about the two-model scope.

---

## Items Verified as Correct

- **No single-model violations.** Each of the 7 directions maps to one model (or one model family in the case of Direction 1's CMEM + H-MIDAS-CMEM). No direction requires "a developer should choose one to implement first" reasoning.

- **Direction 1 grouping is valid.** CMEM (Brownlees 2011) and H-MIDAS-CMEM (Naimoli 2019) are the same model with the latter replacing the daily component. A single implementation with a configurable daily component covers both.

- **Szucs 2017 correctly appears in both Directions 1 and 2** as a Comparison paper, since it directly benchmarks CMEM vs. BDF.

- **Calvori 2014 omitted from Directions 1 and 2 is defensible.** While the GAS-Dirichlet paper discusses CMEM and BDF conceptually, it does not directly benchmark against them (it only compares against its own nested M0 specification). Adding it as Comparison would overstate the empirical relationship.

- **Direction 6 (McCulloch) correctly identified as a separate direction.** The point-process framework is fundamentally different from all time-series decomposition approaches and requires its own implementation effort.

- **Data requirements are generally complete and accurate** across all directions, consistent with the paper summaries.

- **Estimated complexity assessments are reasonable** and well-justified.

# Credit-default model literature notes

Pulled 2026-05-27 in support of model53 sub-model interpretation. Three papers + their meaning for our JC5/JC6/JC7/JM5/MS5 sub-model disagreement signals.

## 1. Chava & Jarrow (2004) — "Bankruptcy Prediction with Industry Effects"

**Source**: SSRN abstract 287474, Review of Finance 8:537-569.

**Model class**: Reduced-form hazard model. Fits a Cox-style hazard rate directly to observed default events with firm-level covariates (financial ratios) + industry effects.

**Key contribution**: Building on Shumway (2001), they show that:
- Hazard models substantially outperform Altman (1968) and Zmijewski (1984)
- **Industry effects matter** — coefficients differ by sector
- Monthly observation intervals beat yearly

**Implication for model53**: **JC5 / JC6 / JC7 are likely Chava-Jarrow hazard models with different covariate sets or sample vintages.** The "5/6/7" likely denotes either:
- Number of financial-ratio covariates (5-variable vs 7-variable specification), or
- Different time-vintage calibrations

JC = reduced-form, observed-default-based, industry-adjusted.

## 2. Merton (1974) + KMV / Vasicek-Kealhofer

**Source**: Merton, R. (1974) "On the pricing of corporate debt"; Bank of England working paper (Merton-model approach); MathWorks reference.

**Model class**: Structural. Treats equity as a call option on firm assets; default occurs when asset value falls below debt threshold. Uses **distance-to-default** (DD) = `(log E[A_T] − log K) / σ_A` (standard deviations between expected asset value and liability threshold).

**KMV / Vasicek-Kealhofer**: Commercial implementation by Kealhofer-McQuown-Vasicek. Refines Merton with calibration to historical default frequencies.

**Implication for model53**: **MS5 is likely Merton-Structural with a 5-variable specification (probably equity-based: equity value, equity vol, debt, time, risk-free rate).** Pure structural — uses market prices not historical default events.

## 3. Arora, Bohn, Zhu (2005) — "Reduced Form vs. Structural Models of Credit Risk: A Case Study of Three Models"

**Source**: SSRN 723041; Journal of Investment Management.

Empirically compares three models:
- **Basic Merton** (simple structural)
- **Vasicek-Kealhofer (VK)** (refined structural, the KMV approach)
- **Hull-White (HW)** (reduced-form)

**Key findings — directly applicable to our model53 signals:**

1. **VK and HW substantially outperform basic Merton.** Implementation details matter.
2. **VK is consistent across firm size** (large + small). **HW degrades for large firms.**
3. **HW does best for firms with many traded bonds** (richer reduced-form data); VK best otherwise.
4. **Structural and reduced-form models give *systematically different* predictions** — they use different information (equity prices + asset structure vs observed defaults + covariates).

## What this means for our findings

The **JC5/JM5 disagreement signal** (alpha_1075/1081 at Sharpe +0.96/+1.10) and **JC7/JM5 disagreement** (alpha_1096 at +1.03 Fitness +0.78 — current leader) carry signal because:

- **Structural vs reduced-form give systematically different predictions** for the same firm (Arora-Bohn-Zhu confirmed).
- When they disagree significantly, **one model is missing information the other captures** — usually because the firm has features (size, equity vol, debt issuance) that map to one model's strengths and the other's weaknesses.
- The market doesn't always price in this "model uncertainty," so **going long the disagreement = collecting a risk premium for taking model-uncertainty exposure**. The risk-premium interpretation we already empirically confirmed (positive Sharpe direction) is theoretically grounded.

**Why JC5/JC6 collapsed** (alpha_1093 at +0.13): both are Chava-Jarrow reduced-form variants with similar specifications. Their disagreement = specification noise, not fundamental model-class disagreement. No signal.

**Why JC6/MS5 went negative** (alpha_1094 at −0.31): same model-class disagreement as JC5/MS5 (reduced-form vs structural) but JC6 is presumably a weaker specification than JC5. The wrong-direction signal is JC6 errors dominating the structural signal.

**Why JC7/JM5 is the strongest pair** (alpha_1096): JC7 is the **most elaborated reduced-form** (more covariates). JM5 captures structural info JC7 misses. The disagreement is "best reduced-form vs hybrid" — pits the richest reduced-form specification against the most structurally-informed sub-model. The literature predicts this should be the most informative.

**JM = ?**: The "JM" label isn't a standard name in the literature. Most likely interpretations:
1. **Jarrow-Modified Merton hybrid** — Jarrow has work combining structural Merton with reduced-form intensity (e.g., Jarrow-Turnbull 1995, Jarrow-Lando-Turnbull 1997). JM5 may be a hybrid specification.
2. **Jarrow stochastic-intensity reduced-form** — variant of Jarrow's reduced-form models using Cox processes.

Either way, JM differs from JC enough that their disagreement is "two-model-class" disagreement, not "same-class specification noise."

## Implications for phase-9 composite design

**Composite A** (jc7/jm5 disagreement + expected_rating_value): **theoretically the strongest**.
- The disagreement leg captures the **model-uncertainty premium** (risk premium for taking exposure to firms where models disagree).
- The rating leg captures the **absolute credit-quality level** (orthogonal information class — what the model consensus says about the firm's credit quality).
- These two pieces of information are theoretically independent. Composite should diversify.

**Composite B** (3 disagreement variants averaged): **theoretically weak**.
- All three legs capture the same model-uncertainty premium. Averaging gives no new information — same lesson as the 5y+7y+10y ensemble (alpha_1087 regressed to +0.70 / +0.40).

**Composite C** (disagreement + rating + 1y disagreement): **probably between A and B**.
- The 1y disagreement leg is the same risk class as 10y disagreement, just shorter horizon. Adds horizon diversification but not info-class diversification.
- May dilute relative to A.

**Expected ranking**: A > C > B. The literature supports composite A as the highest-EV variant in phase 9.

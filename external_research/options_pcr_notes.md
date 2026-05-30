# Options put-call ratio literature notes

Pulled 2026-05-28 for Phase-19 pivot into options-PCR alpha class. Three foundational papers + implications for WQB option8/option9 datafields (138 fields total, 12 `pcr_oi_*` term-structure points from 10 to 1080 days, all cov≈0.98 on TOP3000).

## 1. Pan & Poteshman (2006) — "The Information in Option Volume for Future Stock Prices"

**Source**: Review of Financial Studies 19(3), 871-908. SSRN 622869. Author copy: mit.edu/~junpan/volume.pdf

**Method**: daily put-call ratio constructed from **buy-to-open option volume only** (not open interest, not closing trades). Unique CBOE dataset 1990-2001 that distinguishes trade initiation type.

**Key empirical findings**:
- **Direction**: low PCR > high PCR (informed-trader interpretation, NOT contrarian)
- **Magnitudes**:
  - Next-day return spread: >40 bp (decile sort)
  - Next-week return spread: >1.0%
- **Predictability strength** scales with:
  - Concentration of informed traders (small/illiquid/info-asymmetric firms)
  - Option-contract leverage (deep OTM > ATM)
- **Source of predictability**: non-public information possessed by option traders, not market inefficiency. Implies the signal is genuine alpha, not a correction-of-mispricing play.

**Implication for WQB**:
- WQB's `pcr_oi_*` is OPEN INTEREST based, not volume. Same direction expected (low PCR = bullish accumulated positioning) but slower-moving and probably weaker per-day effect.
- Horizon: 1-7 days strongest; expect smoothing wrapper to extend usable holding without too much decay.
- Operator: `group_rank(-1 * pcr_oi_*, subindustry)` — long low-PCR, short high-PCR.

## 2. Cremers & Weinbaum (2010) — "Deviations from Put-Call Parity and Stock Return Predictability"

**Source**: Journal of Financial and Quantitative Analysis 45, 335-367. SSRN 968237.

**Method**: implied volatility spread between matched call-put pairs of the same strike/expiration. The IV spread measures the put-call-parity deviation — a "sentiment" reading from option pricing.

**Key empirical findings**:
- **Direction**: stocks with relatively expensive CALLS (positive IV spread) outperform stocks with relatively expensive PUTS by **50 bp/week**
- Both halves contribute: positive abnormal returns on the call-expensive leg, negative abnormal returns on the put-expensive leg
- Cannot be explained by short-sale constraints
- **Predictability strength** scales with:
  - Information asymmetry of the firm
  - Residual analyst coverage (more dispersed analyst opinions)
  - Option-contract leverage

**Implication for WQB**:
- WQB does NOT expose call-IV vs put-IV separately, so we can't construct the exact Cremers-Weinbaum IV-spread signal.
- Closest analog from option8/option9: `option_breakeven_*` — combined call+put breakeven price. Not directly the IV-spread but related to the implied volatility skew direction.
- May be worth a Phase-19b experimental variant; not the primary baseline.

## 3. An, Ang, Bali & Cakici (2014) — "The Joint Cross Section of Stocks and Options"

**Source**: Journal of Finance 69(5), 2279-2337. NBER 19590. SSRN 1533089.

**Method**: monthly change in option implied volatility per stock, sorted cross-sectionally into deciles. Examines BIDIRECTIONAL prediction (stocks ↔ options).

**Key empirical findings**:
- **Direction**: stocks with large increase in CALL IV → high next-month returns. Stocks with large increase in PUT IV → low next-month returns.
- **Magnitudes**: decile spread of ~1% per MONTH (sorted on past-month IV change)
- **Persistence**: predictability lasts up to **6 months** (much longer than Pan-Poteshman's week)
- **Bidirectional**: high past stock returns → increase in call/put IV next month, but decrease in realized vol
- Consistent with rational informed-trading models

**Implication for WQB**:
- WQB option8/9 doesn't expose per-stock IV time series directly (only `pcr_oi_*` and `option_breakeven_*`). Can't replicate exactly.
- The 22-day delta + 22-day mean architecture (alpha_1146 recipe) maps loosely to this paper's "monthly IV change" → 1-month return cross-section.
- Persistent 6-month effect argues for keeping the smooth-the-rank wrapper (ts_mean(rank, M)) — slow-decaying signal supports long-horizon stable holdings.

---

## Synthesis — Phase-19 Baseline Design

**Field selection**: `pcr_oi_30` as primary (30-day open interest, captures both Pan-Poteshman 1-week horizon and An-Ang monthly horizon). Cov 0.98 on TOP3000 is sufficient.

**Sign convention**: HIGH `pcr_oi_30` = more accumulated put positioning = bearish per literature. Long signal direction: `-1 * pcr_oi_30` (or invert via group_rank).

**Architecture #1 (baseline level)**:
```
group_rank(-1 * pcr_oi_30, subindustry)
```
Simplest cross-sectional bet. PCR-OI is naturally slower than news sentiment (open interest changes slowly), so turnover may be acceptable without smoothing — unlike the 108% turnover we hit on raw news sentiment in Phase-13a.

**Architecture #2 (smooth-the-rank, alpha_1146 recipe)**:
```
ts_mean(group_rank(-1 * pcr_oi_30, subindustry), 22)
```
Safety wrapper from our news-sentiment win. Only worth firing if baseline #1 has turnover > 45% gate.

**Architecture #3 (term-structure slope, mirroring alpha_0972 parkinson play)**:
```
ts_mean(group_rank(-1 * (pcr_oi_30 - pcr_oi_180), subindustry), 22)
```
30-day minus 180-day PCR captures the **PCR term-structure SLOPE**. Short-term bearish positioning relative to long-term = imminent reversal signal. Analog to alpha_0972's `parkinson_60 - parkinson_180` win.

**Architecture #4 (PCR-momentum, An-Ang analog)**:
```
ts_mean(group_rank(ts_delta(pcr_oi_30, 22), subindustry), 22)
```
22-day change in PCR per An-Ang's monthly IV change. Direction: HIGH delta = rising bearish positioning = lagged short signal. So sign-flip required after rank.

**Settings (locked)**: TOP3000 / SUBINDUSTRY / truncation 0.08 / delay 1 — submission archetype.

**Risk profile vs alpha_1146**: option-positioning data is a fundamentally different information source than news sentiment text. Correlation with alpha_1146 should be low (likely <0.3). High probability of producing a submittable diversification add.

**Order of operations**: fire Architecture #1 first (1 sim, cheap). If Sharpe > 0.5, fan out to #2/#3/#4. If Sharpe near zero or wrong direction (positive PCR predicts positive returns instead of negative), flip sign and retry on Architecture #2.

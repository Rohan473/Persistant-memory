# News-sentiment literature notes

Pulled 2026-05-28 in support of Phase-13 pivot away from mdl53 credit-disagreement (see [[reference-mdl53-concentrated-weight-wall]]). Three foundational papers + implications for WQB news12/news18 datafields (996 fields available on TOP3000, 14 sentiment fields at cov=1.0).

## 1. Tetlock (2007) — "Giving Content to Investor Sentiment: The Role of Media in the Stock Market"

**Source**: Journal of Finance 62(3), 1139-1168. SSRN abstract 685145. Author copy: business.columbia.edu/sites/default/files-efs/pubfiles/3097/Tetlock_Media_Sentiment_JF.pdf

**Method**: counts negative words (custom dictionary) in WSJ "Abreast of the Market" column daily, 1984-1999. Builds a principal-component "pessimism factor" that captures broad investor sentiment independent of fundamentals.

**Key empirical findings**:
- **Direction**: high media pessimism → downward price pressure followed by reversion to fundamentals
- **Magnitude**: 1-SD pessimism → ~20-30 bp negative abnormal returns
- **Predictive horizon**: 5-10 trading days
- **Reversal horizon**: 2-3 weeks (gradual return to fundamentals)
- **Cross-section**: small / low-volume stocks respond more strongly (limited arbitrage)
- **Volume effect**: unusually high OR low pessimism predicts high trading volume

**Implication for alpha design**:
- Operator: group_rank within sector/subindustry, possibly ts_mean over 5-day window to capture the slow-decay effect
- Holding period: ~5-10 days (matches Tetlock predictive horizon)
- Direction: depends on the field's sign convention — if `composite_sentiment_score_2` is signed positive (high = bullish), the immediate-response play is LONG high-sentiment / SHORT low-sentiment. Reversal play is the opposite at 2-3 week horizon.
- Tetlock's strongest cross-section is the immediate response, not the reversal — argues for shorter holding

## 2. Heston & Sinha (2017) — "News vs Sentiment: Predicting Stock Returns from News Stories"

**Source**: Financial Analysts Journal 73(3), 67-83. SSRN abstract 2311310. Most relevant to WQB because uses **Thomson Reuters neural-network sentiment scores** — likely the same engine behind WQB's news12 `*_sentiment_score` fields.

**Method**: 900k+ news stories scored via three methods (Harvard psychosocial dictionary, Loughran-McDonald financial dictionary, **proprietary Thomson Reuters neural network**). Tests both daily and weekly aggregations.

**Key empirical findings**:
- **Daily horizon (single news event)**: 1.99% excess return on announcement day, 0.17% next day, 0.04% day after — then disappears. **Daily news predicts 1-2 days only.**
- **Weekly horizon (5-day sum)**: 3.75% excess return in following week, >2% over the next 13 weeks (full quarter). **Weekly aggregation predicts a quarter ahead.**
- **Asymmetry — KEY**:
  - **Positive news**: incorporated quickly (~1 week effect, then dissipates)
  - **Negative news**: long-delayed reaction (negative premium persists ≥ 1 full quarter, likely due to short-sale constraints)
- News volume matters: only high-news-flow names show the effect cleanly.

**Implication for alpha design**:
- **Use both horizons**: 1-2 day delta (`ts_delta(sentiment, 2)`) captures the immediate Tetlock effect; 5-day mean (`ts_mean(sentiment, 5)`) captures the Heston quarter-ahead momentum
- **Holding period**: weekly aggregation → can hold ~5-22 days; turnover risk is real but bounded
- **Sign asymmetry suggests**: separate positive/negative sentiment legs with different decays — but WQB news12 fields appear to give SIGNED scores (not split). Workaround: use `if_else(sentiment > 0, sentiment, 0)` for positive leg and `if_else(sentiment < 0, sentiment, 0)` for negative leg with different ts_mean windows.
- **News-volume weighting**: weight sentiment by `ts_mean(news_volume, 5)` to suppress signal in low-coverage names.

## 3. Garcia (2013) — "Sentiment During Recessions"

**Source**: Journal of Finance 68(3), 1267-1300. SSRN abstract 1571101. Author copy: leeds-faculty.colorado.edu/garcia/media_v33.pdf

**Method**: 100 years (1905-2005) of NYT financial columns. Counts positive/negative word fractions using Harvard psychosocial dictionary. Compares predictability across NBER recession vs expansion subperiods.

**Key empirical findings**:
- **Recession effect**: 1-SD pessimism shock → -26.7 bp/day predicted return on DJIA, concentrated on **Mondays / post-holiday days**
- **Expansion effect**: -2.4 bp/day — basically no predictability
- **Magnitude differential**: ~10x stronger in recessions
- Asymmetry along business cycle is consistent with "sentiment" (mispricing → reversion) rather than "information" (permanent revaluation)

**Implication for alpha design**:
- Direct application is hard — we can't inject NBER recession dummy in real-time
- **Workable proxy**: high-VIX periods, yield-curve inversion, drawdown regime → use these as conditioning variables
- **Simpler implementation**: just accept the alpha will be noisier than 0.26 bp/day suggests in normal market regimes; use longer holding period to average over regimes
- **Don't over-fit the recession sample**: any alpha that works only post-2008 is probably picking up Garcia's recession effect, not a robust signal

---

## Synthesis — Phase-13 Baseline Design

**Field selection**: based on Heston-Sinha matching the WQB news12 source most closely, use `composite_sentiment_score_2` (raw daily) and/or `mean_composite_sentiment_score` (already smoothed). Heston implies the smoothed/mean variant is the higher-Sharpe play.

**Architecture #1 (baseline level alpha)**:
```
group_rank(zscore(mean_composite_sentiment_score), subindustry)
```
- Long high-mean-sentiment, short low-mean-sentiment
- Already smoothed → maps to Heston's weekly aggregation
- subindustry grouping per portfolio submission archetype

**Architecture #2 (sentiment momentum)**:
```
group_rank(ts_delta(composite_sentiment_score_2, 5), subindustry)
```
- Captures the 5-day change in sentiment per Tetlock + Heston daily

**Architecture #3 (asymmetry — separate positive/negative)**:
```
group_rank(
  ts_mean(if_else(composite_sentiment_score_2 > 0, composite_sentiment_score_2, 0), 5)
  - 0.5 * ts_mean(if_else(composite_sentiment_score_2 < 0, composite_sentiment_score_2, 0), 22),
  subindustry
)
```
- Positive sentiment: 5-day mean (per Heston, positive news incorporated fast)
- Negative sentiment: 22-day mean (per Heston, negative effects persist longer)
- Half weight on negative because the longer-horizon effect is naturally amplified

**Settings (locked)**: TOP3000 / SUBINDUSTRY / truncation 0.08 / delay 1 — matches alpha_1018/1019/1020 submission archetype + the news12 fields require TOP3000 for coverage per [[reference-universe-signal-dependence]].

**Risk vs mdl53 trap**: sentiment fields are continuous across the universe (cov=1.0 on the `mean_*` variants), unlike credit PD ratios which were bimodal. CONCENTRATED_WEIGHT should not bind here. Test architecture #1 first as the simplest sanity check.

**Order of operations**: fire #1 baseline first (1 sim). If Sharpe > 0.5, signal exists — proceed to #2 and #3. If Sharpe near 0, the WQB news fields may have a sign convention different from literature, and we need to flip and retest.

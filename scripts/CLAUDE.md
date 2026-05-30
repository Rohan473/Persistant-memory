# scripts/ — Workflow guide

**Read this before** firing any sim, designing a hypothesis, or searching for a new datafield.

---

## 1. Pre-flight checklist (run BEFORE designing an alpha)

Skip these and you'll waste sims on signals that duplicate existing portfolio exposure.

| Check | Command | Why |
|---|---|---|
| **Family saturation** | `python scripts/query.py saturation` | Avoid families >12% portfolio share; prefer EMPTY families |
| **Pattern overlap** | `grep -l "<expr_pattern>" private/nodes/alphas/*.md` | Catches structural duplicates the family classifier misses |
| **Field coverage** | `python scripts/query.py datafield <name>` | <0.75 coverage on TOP3000 → sub_universe_failure |
| **Sim budget** | `cat memory_layer/sim_usage.json` | 30/day WQB hard cap; refills on alpha acceptance |
| **Prior literature** | Read 3 SSRN/arxiv papers on the datafield's domain | See `feedback_literature_first_on_new_datafields.md` |
| **Field usage in active book** | Count appearances in active alphas before assuming "novel" | vwap/high/low under-used; close/returns/volume over-used |

---

## 2. Correlation — THE most important rule

There are **two different correlation metrics** that diverge by >0.40:

| Metric | Source | What it measures |
|---|---|---|
| **Local pre-flight similarity** | `memory_layer/preflight.py` | Field/operator/concept overlap (structural) |
| **WQB self-correlation** | `GET /alphas/{id}/correlations/self` | Actual PnL correlation |

**Critical:** low pre-flight similarity ≠ low PnL correlation. Reference case: alpha_1260 had pre-flight 0.27 but WQB 0.71. Any short-horizon reversion signal will correlate via PnL regardless of which field it uses.

**SELF_CORRELATION fail rule (memory: `reference_self_correlation_threshold.md`):**
- FAIL only if `corr > 0.70 AND new_sharpe < 1.10 × target_sharpe`
- Below 0.70 → always PASS
- WQB computes async after submission; poll `/alphas/{id}/correlations/self` (empty body = pending)

---

## 3. Sim execution

| Tool | When | Notes |
|---|---|---|
| `sweep.py run <template>` | Single template, sequential variants | Default; respects daily budget + quiet hours |
| `run_concurrent.py t1 t2 t3 --tabs` | Multiple templates in parallel | WQB caps at 3 concurrent slots; `--tabs` opens Windows Terminal tabs |
| `--ignore-quiet-hours` | After 21:00 local | Required to fire late-evening sims |
| `--force` | Re-run identical expression | Defaults skip duplicates |

**Sim budget reality:** `sim_usage.json` is a local guard; WQB itself refills the 30-sim counter on each alpha acceptance — see `reference_sim_budget_refill_on_acceptance.md`.

---

## 4. Tool inventory (when to use what)

| Script | Purpose |
|---|---|
| `query.py` | KG queries — subcommands: `concept`, `datafield`, `setting`, `failures`, `lineage`, `best`, `regime`, `year`, `crisis-robust`, `gaps-catalogue`, `quality-gaps`, `search`, `memory`, **`saturation`** |
| `sweep.py` | Fire template — `list`, `show`, `expand`, `run` |
| `run_concurrent.py` | Parallel template execution |
| `overfit_checker.py` | Post-sim diagnosis — risk score, OOS survival, live grade with `--live` |
| `critic.py` | Fresh-context external review of recent decisions |
| `get_alpha.py` | Fetch one alpha's full WQB record (status, OS checks, selfCorr) |
| `writeback_alpha.py` | Recover completed-but-unwritten WQB sim into the KG |
| `analyze_mdl53.py` | Branch-specific (mdl53 sweep analysis) |
| `portfolio_analysis.py` | Aggregate portfolio metrics |
| `dashboard.py` | Visual report |
| `import_submitted.py` | Sync submitted alphas from WQB into KG |
| `build_graph.py` | Rebuild knowledge graph from alpha files |
| `fetch_catalogue.py` | Refresh `brain_catalogue.json` from WQB |

---

## 5. Expression syntax landmines

| Rule | Source memory |
|---|---|
| `hump(x, hump=0.01)` — 2nd arg is keyword | `reference_wqb_hump_operator.md` |
| Decimals only — `1e-9` errors, use `0.000001` | `reference_wqb_no_scientific_notation.md` |
| `log(x)` errors — try `ln(x)` or different formulation | observed alpha_1256 |
| `ts_decay_linear` for low_fitness — DON'T | `feedback_ts_decay_linear_smoothing_trap.md` |
| `ts_step(N)` is monotonic counter, NOT cyclic | observed alpha_1216 |
| Fundamental composite fields (mdl177_emm) rank-wrapped → flat | observed alpha_1267 |

---

## 6. Designing for novelty — `ts_rank` vs `ts_sum`

Discovered 2026-05-30: **`ts_rank(close, N)` momentum gives ~8× the Sharpe of `ts_sum(returns, N)` momentum** (0.83 vs 0.10 for JT 12-1) because ts_rank is bounded per-stock and resilient to 2022-style regime crashes. Apply this rule to any new return-accumulation idea — prefer `ts_rank` scaffolding.

---

## 7. Post-sim workflow

1. Read alpha frontmatter — `sharpe`, `fitness`, `turnover`, `failure_modes`, `pipeline_state`
2. **If `pipeline_state: IS_PASS`** → run `python scripts/overfit_checker.py <alpha_id> --verbose`
3. **If submitted** → run `python scripts/overfit_checker.py <alpha_id> --live --verbose` for grade + selfCorr + train/test comparison
4. **If user reports portfolio score change** → check correlation table; the family overlap (not single-pair) drives score drag

---

## 8. Memory references (load via `query.py memory <keyword>`)

The portfolio-context lessons live in `~/.claude/projects/E--New-folder-WQB-knowledge-graph/memory/`. Always-loaded index: `MEMORY.md`.

Most-cited:
- `reference_portfolio_saturation.md` — when to use `query.py saturation`
- `reference_field_novelty_vs_pnl_novelty.md` — why local similarity ≠ WQB correlation
- `reference_self_correlation_threshold.md` — the 0.70 × 1.10 rule
- `reference_submission_order_strategy.md` — submit weakest-Sharpe sibling first
- `reference_correlation_oc_rev_bridge.md` — `rank(1-close/open)` forces ≥0.70 correlation
- `feedback_predict_correlation_before_iterating.md` — don't burn sims on predictably-correlated variants
- `feedback_match_external_config.md` — match universe/settings/period when validating
- `feedback_d0_validation_first.md` — fire one D0 probe before any batch
- `feedback_min_coverage_threshold.md` — 0.75 minimum

---

## 9. Pure-field probes can be local-optimal but un-submittable

Observed 2026-05-30: `rank(-book_leverage_ratio_3)` produced the best standalone metrics (Sharpe 1.41, Margin 84.74‱, +228 estimate) but **failed IS Testing on Sub-universe Sharpe** because `book_leverage_ratio_3` coverage drops on TOP500/TOP200 even though it's 100% on TOP3000. None of {ts_mean smoothing, winsorize, industry-neutralization} fixed this — it's structural to the field.

**The fix is to compose with a second leg that has good small-cap coverage** (price-based, volume-based, returns-based). The composite passes sub-universe even when the pure field doesn't. Side effect: composites also bring correlation overlap — pick the second leg for minimum PnL correlation with existing book.

Rule: when designing a fundamental probe, check coverage on the sub-universes (TOP500, TOP200) AS WELL AS the main universe. If coverage drops below ~0.70 on the smaller subset, plan for a composite from the start.

## 9a. Yearly-consistency gate (the hidden one)

Observed 2026-05-30 evening: alpha_1280 had aggregate Sharpe 1.58 with +178 score estimate, but submission stalled with WQB flagging weak train-set performance. Cause: 2020 Sharpe was 0.48 and 2023 was 0.36 — the aggregate was carried by 2021/2022 strength.

**WQB's submission process runs a train/test split on the IS years.** A high-aggregate alpha with one or two weak years can still be downgraded or stalled at submission, even if it passes the standard 8 IS Testing gates.

**Hard gate to apply during design:** prefer composites where **every yearly Sharpe ≥ 0.8**. Use `ts_mean(close, 22)` style slow signals over `close - ts_mean(close, 5)` fast signals if the fast signal has uneven yearly profile. Slower windows produce more consistent year-by-year behavior, even at the cost of lower aggregate Sharpe.

Check this from alpha frontmatter by reading the simulation result panel in WQB — the per-year Sharpe row is the data that matters for the hidden gate.

---

## 10. The one workflow rule

> **Saturation → grep → coverage (main AND sub-universes) → literature → design → sim → overfit_check → live correlation.**
>
> Skipping any step burns sims on signals that won't add portfolio value, even if standalone metrics look strong. The portfolio rewards novelty far more than marginal Sharpe.

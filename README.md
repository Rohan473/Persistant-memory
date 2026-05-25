# WQ Brain Alpha Knowledge Graph

A structured knowledge graph of all WorldQuant Brain alpha development sessions, built automatically from Claude.ai chat exports.

---

## What's inside

| Item | Count |
|------|-------|
| Chat sessions processed | 16 |
| Alpha nodes extracted | 91 |
| Distinct concepts | 13 |
| Distinct datafields | 24 |
| Distinct operators | 23 |
| Failure mode types | 7 |
| Setting combos | 7 |
| Total graph edges | 1295 |

---

## Key patterns observed

### Most-explored signals
- **CLV (Close-Location Value)** — `((close-low)-(high-close))/(high-low)` — the single highest-Sharpe direction found (alpha_0803, Sharpe 2.19), but all variants failed due to extreme turnover (70–138%). The core signal is strong; the challenge is reducing turnover via decay settings while preserving Sharpe.
- **EBIT/CapEx ratio** — `-rank(ebit/capex)` — the canonical beginner alpha. Industry neutralization brought Sharpe to 1.31 (alpha_0901) but failed self-correlation (0.9886). All smoothing attempts destroyed the signal because EBIT/CapEx is quarterly data — smoothing the same quarterly value adds noise. Final unresolved path: use `Decay=5` in simulation *settings* (not formula).
- **Win-streak momentum** — `-ts_rank(returns>0?1:0, 250)` — Sharpe 1.62 (alpha_0600) but turnover 131%. A 250-day linear decay partially fixed this (alpha_0604, iterating).
- **EBIT margin + valuation composite** — `group_rank(... fnd2_ebitdm, fnd2_ebitfr ...)` — heavy iteration in the abe10f6c session; best Sharpe ~1.04 but consistently failed fitness.
- **Analyst revision momentum** — `analyst_revision_rank_derivative` — consistently negative Sharpe (-0.70). Decaying and z-scoring did not help. Signal direction may be inverted or the field needs cross-sectional normalization.

### Recurring failure modes
1. **low_fitness (76 alphas)** — Most common failure. Fitness = f(Sharpe, turnover); failing Sharpe always fails fitness.
2. **low_sharpe (61 alphas)** — The 1.25 threshold is strict; most fundamental signals underperform without proper neutralization.
3. **high_turnover (12 alphas)** — Mainly technical/price-volume signals. Fix: add `ts_decay_linear` or increase `Decay` in settings.
4. **sector_bias (11 alphas)** — Using market neutralization on fundamental ratios compares tech to utilities. Fix: use industry or subindustry neutralization.
5. **os_failure (3 alphas)** — IS/OS divergence; signals overfit on 2019-2023 training window.

### Underexplored datafields (gaps)
These appear in 0–2 alphas despite being available in WQ Brain:
- `cap` (market cap) — 0 alphas — pure size signal unexplored
- `vwap` — 2 alphas — intraday price discovery signal underused
- `cashflow_op` — 2 alphas — operating cash flow signals mostly skipped
- `beta_last_30_days_spy` — 1 alpha — market sensitivity signal barely touched

### Universe split
- **TOP3000 delay=1**: 65 alphas (main workbench)
- **TOP1000 delay=1**: 13 alphas (competition-oriented testing)
- **TOP500 delay=1**: 13 alphas (concentrated portfolio testing)

---

## How to use query.py

```bash
# List all alphas implementing a concept, sorted by Sharpe
python scripts/query.py concept mean_reversion

# List all alphas using a specific datafield
python scripts/query.py datafield close

# List alphas tested under a specific setting
python scripts/query.py setting TOP3000 1 industry

# Frequency table of all failure modes
python scripts/query.py failures

# Datafields used in fewer than 3 alphas (exploration gaps)
python scripts/query.py gaps

# Print the full derivation tree for an alpha
python scripts/query.py lineage alpha_0900

# Top N alphas by Sharpe with full details
python scripts/query.py best 10
```

---

## File structure

```
exports/              Raw chat exports (.json)
nodes/
  alphas/             alpha_NNNN.md — one file per distinct alpha expression
  concepts/           Concept entity nodes (mean_reversion, momentum, ...)
  datafields/         Datafield entity nodes (close, ebit, capex, ...)
  operators/          Operator entity nodes (rank, ts_rank, group_neutralize, ...)
  settings/           Setting combo nodes (TOP3000_1_industry.md, ...)
  failure_modes/      Failure mode nodes
  sessions/           Session summary nodes + _manifest.json
    raw/              Full session text files (input for extraction)
graph/
  graph.gpickle       NetworkX DiGraph (Python pickle)
  edges.csv           All edges as CSV (source, target, relation)
  graph.png           Rendered visualization
  graph.gexf          Gephi-compatible export
scripts/
  parse_exports.py    Phase 2: parse exports, filter WQ Brain sessions
  backfill_entities.py  Post-extraction: fill missing entity node files
  build_graph.py      Phase 4: build NetworkX graph from markdown files
  query.py            Phase 5: CLI retrieval helper
  visualize.py        Phase 6: render graph.png + graph.gexf
```

---

## Re-running the pipeline (adding new exports)

When you add new Claude.ai or ChatGPT exports:

### Option A: Full re-run (simplest)
```bash
# 1. Drop new .json file(s) into exports/
# 2. Re-parse (only new sessions are added; existing ones skip via manifest)
python scripts/parse_exports.py
# 3. Backfill entities
python scripts/backfill_entities.py
# 4. Rebuild graph
python scripts/build_graph.py
# 5. Re-visualize
python scripts/visualize.py
```

### Incremental extraction
For new sessions only, check `nodes/sessions/_manifest.json` to see which `session_id`s already exist. Copy the new session `.txt` from `nodes/sessions/raw/` and run the extraction guide manually or spawn a sub-agent with the new session file and a fresh alpha ID range (start after the highest existing ID).

### Alpha ID range tracking
Current highest used IDs per agent batch:
- 0001–0017 (Pearson correlation session)
- 0100–0108 (uncorrelated price-volume session)
- 0200–0212 (EBIT ranking session)
- 0300–0309 (creating alphas + combined report)
- 0400–0407 (grouping sectors)
- 0600–0609 (liabilities session)
- 0700–0706 (paper strategies + WQB overview)
- 0800–0807 (fitness improvement + ruflo)
- 0900–0908 (EBIT/CapEx ratio)

Next available batch: start from **alpha_1000** for new sessions.

---

## Ambiguity notes

1. **"This block is not supported on your current device yet."** — Several sessions had file-write blocks that weren't visible in the export. The d9635e35 "Combined alpha report" session is entirely composed of these invisible outputs; no expressions were extractable from it.
2. **Informal metric reporting** — When users pasted IS test results as plain text (e.g., "Sharpe of 1.31 is above cutoff"), agents parsed the numeric value directly. When Sharpe was mentioned only in passing ("it was around 1.3"), the agent used that value and is marked in status appropriately.
3. **Multi-session expressions** — If the same expression appeared in multiple sessions, the earliest session was used as the canonical source.
4. **Settings in formula vs. settings panel** — WQ Brain expressions sometimes embed neutralization inline (`with industry in Neutralization`) and sometimes it's a separate simulation setting. Both are normalized to the `neutralization` frontmatter field.

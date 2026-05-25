# Extraction Guide for WQ Brain Alpha Knowledge Graph

## Working directory
E:\New folder\WQB\knowledge_graph

## Session files location
E:\New folder\WQB\knowledge_graph\nodes\sessions\raw\{session_id}.txt

---

## Step 1: Read session files completely
Use the Read tool. Files can be 1000–4000+ lines. Use limit/offset to read in chunks of 2000 lines. Keep reading until you've seen the entire file.

---

## Step 2: What constitutes an alpha worth extracting?

CREATE an alpha node if:
- There is a distinct WQ Brain expression (from a ```fastexpr-regular block OR inline code like `-rank(ebit/capex)`)
- The same expression with different settings (neutralization/decay/truncation) = separate alpha
- An expression with IS/OS simulation metrics reported

SKIP:
- Lines that say "This block is not supported on your current device yet." (file content not visible)
- Pure discussion without an actual expression or concrete metrics
- Expressions labeled only as "hypothetical" with zero metrics

---

## Step 3: Alpha markdown format

Write to: E:\New folder\WQB\knowledge_graph\nodes\alphas\alpha_NNNN.md

```
---
id: alpha_NNNN
expression: "<exact WQ Brain expression>"
datafields: [ebit, capex, close, ...]
operators: [rank, ts_rank, ...]
concepts: [value, fundamental, ...]
universe: TOP3000
region: USA
delay: 1
neutralization: industry
decay: null
truncation: null
sharpe: 1.31
turnover: 11.45
fitness: 1.06
returns: null
drawdown: null
margin: null
status: rejected
failure_modes: [correlated]
parent_alpha: null
correlated_with: []
session: <UUID from file header>
hypothesis: "<one-line market intuition>"
---

Body: explain what the alpha does, why tried, what happened.
Use wikilinks: [[concept_name]], [[datafield_name]], [[operator_name]]
```

---

## Step 4: Entity node formats

### Concept: E:\New folder\WQB\knowledge_graph\nodes\concepts\{name}.md
```
---
type: concept
name: value
---
One-paragraph description.
```

### Datafield: E:\New folder\WQB\knowledge_graph\nodes\datafields\{name}.md
```
---
type: datafield
name: ebit
---
One-paragraph description.
```

### Operator: E:\New folder\WQB\knowledge_graph\nodes\operators\{name}.md
```
---
type: operator
name: ts_rank
---
Description with syntax.
```

### Setting: E:\New folder\WQB\knowledge_graph\nodes\settings\{universe}_{delay}_{neutralization}.md
```
---
type: setting
universe: TOP3000
delay: 1
neutralization: industry
---
```

### Failure mode: E:\New folder\WQB\knowledge_graph\nodes\failure_modes\{name}.md
```
---
type: failure_mode
name: correlated
---
Description.
```

---

## Parsing IS test results

Pattern:
```
N PASS / M FAIL
* Sharpe of X is [above/below] cutoff of Y   → sharpe: X
* Fitness of X is [above/below] cutoff of Y  → fitness: X
* Turnover of X% is [above/below] cutoff of Y → turnover: X
```

If a year-by-year table is provided (Year / Sharpe / Turnover / Fitness / Returns / Drawdown / Margin):
- returns:  average of the yearly Returns column (%)
- drawdown: maximum value across the yearly Drawdown column (%)
- margin:   average of the yearly Margin column (‱)
Never write null for these fields when a year-by-year table is present in the session.

---

## Status values
- `submitted`: passed all tests (7+ PASS or explicitly "submitted to competition")
- `rejected`: failed criteria AND abandoned in the session
- `iterating`: still being modified at session end
- `idea_only`: discussed but never actually simulated

## Default settings (when not stated)
- universe: TOP3000
- region: USA
- delay: 1
- neutralization: market

## Concept vocabulary (use ONLY these)
mean_reversion, momentum, volatility, value, quality, liquidity, sentiment, technical, fundamental, cross_sectional, time_series, normalization, neutralization

## Failure mode vocabulary
high_turnover, low_fitness, low_sharpe, os_failure, correlated, low_margin, data_quality, overfitting, sector_bias

## Operator vocabulary (recognize these in expressions)
rank, ts_rank, group_neutralize, ts_mean, ts_std, ts_delta, ts_corr, ts_zscore, zscore, decay_linear, ts_decay_linear, ts_decay_exp, sign, abs, log, sqrt, power, min, max, sum, stddev, regression_slope, regression_intercept, indneutralize, correlation, neutralize, winsorize, quantile, percentage_change

## Common datafields
close, open, high, low, volume, vwap, returns, ebit, capex, total_debt, total_assets, book_value, equity, revenue, net_income, adv, cap, shares_outstanding, liabilities, current_liabilities, short_interest, dividend, eps, sales, cashflow, free_cashflow, inventory, receivables, payables, depreciation, amortization, beta, pe_ratio, pb_ratio, ps_ratio, roa, roe, leverage, interest_expense, tax_rate

---

## Notes for specific sessions
- "This block is not supported on your current device yet." = file tool output not visible; skip
- If a session has 0 extractable alphas, note it in your report — that's valid
- For `parent_alpha`: only link within your assigned batch; use null if parent is in another agent's batch
- The session UUID is in the second line of each .txt file: "ID: {uuid}"

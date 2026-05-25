"""
Import the 11 submitted alphas from E:/New folder/submitted_alphas.txt.
Creates alpha_0020..alpha_0029 with status=submitted.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "nodes" / "alphas"

# ── Aggregate helper ─────────────────────────────────────────────────────────
def agg(yearly_metrics):
    """yearly_metrics = list of dicts {sharpe, turnover, fitness, returns, drawdown, margin}"""
    n = len(yearly_metrics)
    return {
        "sharpe":   round(sum(y["sharpe"]   for y in yearly_metrics) / n, 3),
        "turnover": round(sum(y["turnover"] for y in yearly_metrics) / n, 2),
        "fitness":  round(sum(y["fitness"]  for y in yearly_metrics) / n, 3),
        "returns":  round(sum(y["returns"]  for y in yearly_metrics) / n, 2),
        "drawdown": round(max(y["drawdown"] for y in yearly_metrics), 2),
        "margin":   round(sum(y["margin"]   for y in yearly_metrics) / n, 2),
    }

# ── The 11 submitted alphas ──────────────────────────────────────────────────
ALPHAS = [
    {  # 1
        "id":"alpha_0020",
        "expression":"zscore(group_rank(zscore(ts_decay_linear(-mdl77_2400_pbroeresidual, 10)) + zscore(ts_decay_linear(ebit / revenue, 10)) + zscore(ts_decay_linear((cashflow_op - capex) / assets, 10)), subindustry))",
        "datafields":["mdl77_2400_pbroeresidual","ebit","revenue","cashflow_op","capex","assets"],
        "operators":["zscore","group_rank","ts_decay_linear"],
        "concepts":["fundamental","value","quality","cross_sectional","normalization"],
        "universe":"TOP3000","delay":1,"neutralization":"subindustry","decay":4,"truncation":0.08,
        "rating":"Average","parent_alpha":"alpha_0017",
        "hypothesis":"Three orthogonal fundamental signals (P/B/ROE residual, EBIT margin, FCF/assets) decayed and z-scored before subindustry-rank should diversify fundamental risk.",
        "yearly":[
            {"sharpe":0.21,"turnover":19.87,"fitness":0.04,"returns":0.69,"drawdown":5.56,"margin":0.69},
            {"sharpe":1.87,"turnover":20.54,"fitness":1.53,"returns":13.81,"drawdown":4.78,"margin":13.45},
            {"sharpe":2.41,"turnover":20.46,"fitness":1.97,"returns":13.61,"drawdown":4.32,"margin":13.31},
            {"sharpe":1.84,"turnover":19.91,"fitness":1.42,"returns":11.92,"drawdown":5.94,"margin":11.97},
            {"sharpe":1.49,"turnover":18.82,"fitness":0.83,"returns":5.89,"drawdown":1.93,"margin":6.26},
        ],
        "created":"2026-05-05",
    },
    {  # 2
        "id":"alpha_0021",
        "expression":"hump(ts_decay_linear(-group_neutralize(scl12_buzz, subindustry), 5))",
        "datafields":["scl12_buzz"],
        "operators":["hump","ts_decay_linear","group_neutralize"],
        "concepts":["sentiment","mean_reversion","neutralization"],
        "universe":"TOP3000","delay":1,"neutralization":"sector","decay":0,"truncation":0.08,
        "rating":"Average","parent_alpha":None,
        "hypothesis":"Social/news buzz signals are noisy; subindustry-neutralizing then hump-smoothing the negated signal extracts contrarian mean-reversion from overhyped stocks.",
        "yearly":[
            {"sharpe":1.43,"turnover":24.87,"fitness":0.74,"returns":6.63,"drawdown":3.46,"margin":5.33},
            {"sharpe":2.74,"turnover":20.67,"fitness":2.78,"returns":21.23,"drawdown":2.55,"margin":20.54},
            {"sharpe":2.33,"turnover":13.46,"fitness":2.60,"returns":16.80,"drawdown":4.72,"margin":24.97},
            {"sharpe":1.37,"turnover":16.93,"fitness":0.98,"returns":8.70,"drawdown":3.22,"margin":10.28},
            {"sharpe":0.85,"turnover":16.74,"fitness":0.48,"returns":5.32,"drawdown":3.68,"margin":6.36},
        ],
        "created":"2026-05-02",
    },
    {  # 3 -- GOOD rating
        "id":"alpha_0022",
        "expression":"ts_decay_linear(group_rank(ts_rank(operating_income / cap, 120), industry), 10)",
        "datafields":["operating_income","cap"],
        "operators":["ts_decay_linear","group_rank","ts_rank"],
        "concepts":["fundamental","quality","value","time_series","cross_sectional"],
        "universe":"TOP500","delay":1,"neutralization":"industry","decay":0,"truncation":0.08,
        "rating":"Good","parent_alpha":None,
        "hypothesis":"Earnings yield (OpIncome/Cap) ranked over 120 days against its own history, then ranked industry-wide and decayed, captures persistent earnings-yield momentum on a concentrated universe.",
        "yearly":[
            {"sharpe":2.18,"turnover":9.35,"fitness":1.66,"returns":7.27,"drawdown":2.03,"margin":15.55},
            {"sharpe":1.87,"turnover":9.38,"fitness":1.91,"returns":13.03,"drawdown":3.21,"margin":27.78},
            {"sharpe":1.64,"turnover":8.80,"fitness":1.45,"returns":9.71,"drawdown":4.08,"margin":22.05},
            {"sharpe":2.01,"turnover":9.03,"fitness":2.24,"returns":15.51,"drawdown":3.82,"margin":34.34},
            {"sharpe":1.27,"turnover":8.90,"fitness":0.83,"returns":5.37,"drawdown":2.99,"margin":12.06},
        ],
        "created":"2026-04-30",
    },
    {  # 4
        "id":"alpha_0023",
        "expression":"0.90 * ts_decay_linear(-ts_rank(returns>0? 1:0, 250), 250) + 0.10 * -ts_rank(returns>0? 1:0, 250)",
        "datafields":["returns"],
        "operators":["ts_decay_linear","ts_rank"],
        "concepts":["mean_reversion","momentum","time_series"],
        "universe":"TOP3000","delay":1,"neutralization":"subindustry","decay":4,"truncation":0.08,
        "rating":"Average","parent_alpha":"alpha_0604",
        "hypothesis":"90/10 blend of long-window decayed win-streak rank and raw win-streak rank — anchored by the slow signal with a small reactive component.",
        "yearly":[
            {"sharpe":1.05,"turnover":43.93,"fitness":0.31,"returns":3.73,"drawdown":3.95,"margin":1.70},
            {"sharpe":2.75,"turnover":51.80,"fitness":1.73,"returns":20.47,"drawdown":4.32,"margin":7.90},
            {"sharpe":1.27,"turnover":50.78,"fitness":0.55,"returns":9.69,"drawdown":4.11,"margin":3.81},
            {"sharpe":2.50,"turnover":49.03,"fitness":1.62,"returns":20.68,"drawdown":5.20,"margin":8.44},
            {"sharpe":2.69,"turnover":54.54,"fitness":1.42,"returns":15.28,"drawdown":1.91,"margin":5.60},
        ],
        "created":"2026-04-29",
    },
    {  # 5
        "id":"alpha_0024",
        "expression":"ts_decay_linear(-ts_rank(returns>0? 1:0, 250), 200)",
        "datafields":["returns"],
        "operators":["ts_decay_linear","ts_rank"],
        "concepts":["mean_reversion","time_series"],
        "universe":"TOP3000","delay":1,"neutralization":"subindustry","decay":0,"truncation":0.08,
        "rating":"Average","parent_alpha":"alpha_0604",
        "hypothesis":"200-day decayed win-streak reversal: long-horizon reversal of stocks with highest positive-return frequency.",
        "yearly":[
            {"sharpe":0.83,"turnover":16.55,"fitness":0.36,"returns":3.18,"drawdown":4.48,"margin":3.84},
            {"sharpe":0.94,"turnover":13.77,"fitness":0.64,"returns":6.43,"drawdown":5.62,"margin":9.34},
            {"sharpe":0.97,"turnover":14.17,"fitness":0.65,"returns":6.36,"drawdown":3.91,"margin":8.98},
            {"sharpe":2.22,"turnover":12.03,"fitness":2.60,"returns":17.09,"drawdown":5.34,"margin":28.42},
            {"sharpe":1.62,"turnover":13.91,"fitness":1.30,"returns":8.93,"drawdown":2.36,"margin":12.84},
        ],
        "created":"2026-04-29",
    },
    {  # 6
        "id":"alpha_0025",
        "expression":"ts_rank(operating_income/cap,252)",
        "datafields":["operating_income","cap"],
        "operators":["ts_rank"],
        "concepts":["fundamental","quality","time_series"],
        "universe":"TOP3000","delay":1,"neutralization":"subindustry","decay":0,"truncation":0.08,
        "rating":"Average","parent_alpha":None,
        "hypothesis":"If current operating income/market-cap is at the high end of its 1-year history, the stock is likely under-valued — go long.",
        "yearly":[
            {"sharpe":2.80,"turnover":13.20,"fitness":1.91,"returns":6.13,"drawdown":1.89,"margin":9.29},
            {"sharpe":0.92,"turnover":13.89,"fitness":0.51,"returns":4.23,"drawdown":5.80,"margin":6.09},
            {"sharpe":2.33,"turnover":13.70,"fitness":2.13,"returns":11.42,"drawdown":3.84,"margin":16.67},
            {"sharpe":1.81,"turnover":12.65,"fitness":1.97,"returns":15.01,"drawdown":6.66,"margin":23.73},
            {"sharpe":-0.01,"turnover":12.08,"fitness":-0.00,"returns":-0.02,"drawdown":3.07,"margin":-0.04},
        ],
        "created":"2026-04-28",
    },
    {  # 7
        "id":"alpha_0026",
        "expression":"ts_decay_linear(rank(-returns * rank(volume/adv20)) + rank(1 - close/open) + 0.5 * rank(revenue/ebit), 5)",
        "datafields":["returns","volume","adv20","close","open","revenue","ebit"],
        "operators":["ts_decay_linear","rank"],
        "concepts":["mean_reversion","value","liquidity","technical","cross_sectional"],
        "universe":"TOP1000","delay":1,"neutralization":"subindustry","decay":4,"truncation":0.08,
        "rating":"Average","parent_alpha":None,
        "hypothesis":"Three-component blend: volume-weighted return reversal + intraday range reversal + reverse-value (revenue/ebit penalty) — captures multiple short-horizon mispricings on liquid TOP1000.",
        "yearly":[
            {"sharpe":-0.04,"turnover":39.20,"fitness":-0.00,"returns":-0.18,"drawdown":3.76,"margin":-0.09},
            {"sharpe":2.40,"turnover":39.16,"fitness":1.97,"returns":26.49,"drawdown":5.75,"margin":13.53},
            {"sharpe":1.81,"turnover":39.71,"fitness":1.35,"returns":22.00,"drawdown":8.61,"margin":11.08},
            {"sharpe":2.63,"turnover":39.29,"fitness":2.44,"returns":33.81,"drawdown":5.29,"margin":17.21},
            {"sharpe":1.58,"turnover":39.20,"fitness":0.82,"returns":10.67,"drawdown":5.44,"margin":5.44},
        ],
        "created":"2026-05-11",
    },
    {  # 8
        "id":"alpha_0027",
        "expression":"ts_decay_linear(rank(-returns * rank(volume/adv20)) + 0.5 * rank(revenue/ebit), 5)",
        "datafields":["returns","volume","adv20","revenue","ebit"],
        "operators":["ts_decay_linear","rank"],
        "concepts":["mean_reversion","value","liquidity","cross_sectional"],
        "universe":"TOP1000","delay":1,"neutralization":"subindustry","decay":4,"truncation":0.08,
        "rating":"Average","parent_alpha":"alpha_0026",
        "hypothesis":"Stripped-down version of alpha_0026 — removed intraday range component to isolate volume-weighted reversal + value penalty.",
        "yearly":[
            {"sharpe":-0.15,"turnover":30.78,"fitness":-0.02,"returns":-0.60,"drawdown":3.14,"margin":-0.39},
            {"sharpe":1.55,"turnover":30.45,"fitness":1.08,"returns":14.82,"drawdown":9.28,"margin":9.74},
            {"sharpe":2.21,"turnover":30.74,"fitness":1.85,"returns":21.54,"drawdown":9.02,"margin":14.01},
            {"sharpe":2.81,"turnover":30.56,"fitness":2.72,"returns":28.58,"drawdown":5.73,"margin":18.70},
            {"sharpe":0.75,"turnover":30.35,"fitness":0.30,"returns":4.84,"drawdown":5.51,"margin":3.19},
        ],
        "created":"2026-05-11",
    },
    {  # 9
        "id":"alpha_0028",
        "expression":"ts_decay_linear(rank(1 - close/open), 5)",
        "datafields":["close","open"],
        "operators":["ts_decay_linear","rank"],
        "concepts":["mean_reversion","technical","cross_sectional"],
        "universe":"TOP1000","delay":1,"neutralization":"subindustry","decay":4,"truncation":0.08,
        "rating":"Average","parent_alpha":None,
        "hypothesis":"Stocks that closed below their open (negative intraday return) should mean-revert; a 5-day decay smooths daily flips.",
        "yearly":[
            {"sharpe":0.46,"turnover":45.40,"fitness":0.10,"returns":2.32,"drawdown":4.45,"margin":1.02},
            {"sharpe":3.22,"turnover":46.33,"fitness":2.82,"returns":35.56,"drawdown":5.34,"margin":15.35},
            {"sharpe":1.11,"turnover":46.12,"fitness":0.68,"returns":17.39,"drawdown":10.02,"margin":7.54},
            {"sharpe":1.66,"turnover":45.60,"fitness":1.26,"returns":26.07,"drawdown":9.69,"margin":11.43},
            {"sharpe":1.88,"turnover":46.10,"fitness":1.03,"returns":13.83,"drawdown":3.67,"margin":6.00},
        ],
        "created":"2026-05-11",
    },
    {  # 10
        "id":"alpha_0029",
        "expression":"ts_decay_linear(zscore(rank(-returns - group_mean(returns, 1, sector))) + zscore(-ts_rank((net_income_adjusted - cashflow_op) / assets, 252)), 5)",
        "datafields":["returns","net_income_adjusted","cashflow_op","assets"],
        "operators":["ts_decay_linear","zscore","rank","group_mean","ts_rank"],
        "concepts":["mean_reversion","quality","fundamental","cross_sectional","normalization"],
        "universe":"TOP3000","delay":1,"neutralization":"subindustry","decay":4,"truncation":0.08,
        "rating":"Average","parent_alpha":None,
        "hypothesis":"Combine sector-residual return reversal (technical) with negative-accruals quality signal (fundamental) — orthogonal mechanisms smoothed over 5 days.",
        "yearly":[
            {"sharpe":0.87,"turnover":14.95,"fitness":0.33,"returns":2.09,"drawdown":2.31,"margin":2.79},
            {"sharpe":2.62,"turnover":13.94,"fitness":2.39,"returns":11.62,"drawdown":1.74,"margin":16.68},
            {"sharpe":1.73,"turnover":14.14,"fitness":1.23,"returns":7.16,"drawdown":1.63,"margin":10.12},
            {"sharpe":1.25,"turnover":13.91,"fitness":0.91,"returns":7.40,"drawdown":4.53,"margin":10.64},
            {"sharpe":1.73,"turnover":14.37,"fitness":1.15,"returns":6.32,"drawdown":3.19,"margin":8.80},
        ],
        "created":"2026-05-12",
    },
]

# ── Write all alpha files ────────────────────────────────────────────────────
for a in ALPHAS:
    m = agg(a["yearly"])
    parent = f"{a['parent_alpha']}" if a["parent_alpha"] else "null"
    fields_list = "[" + ", ".join(a["datafields"]) + "]"
    ops_list    = "[" + ", ".join(a["operators"])  + "]"
    cons_list   = "[" + ", ".join(a["concepts"])   + "]"

    year_table_rows = []
    for i, y in enumerate(a["yearly"]):
        year = 2019 + i
        year_table_rows.append(
            f"| {year} | {y['sharpe']:+.2f} | {y['turnover']:.2f}% | {y['fitness']:+.2f} | {y['returns']:+.2f}% | {y['drawdown']:.2f}% | {y['margin']:+.2f}‱ |"
        )
    year_table = "\n".join(year_table_rows)

    content = f"""---
id: {a['id']}
expression: "{a['expression']}"
datafields: {fields_list}
operators: {ops_list}
concepts: {cons_list}
universe: {a['universe']}
region: USA
delay: {a['delay']}
neutralization: {a['neutralization']}
decay: {a['decay']}
truncation: {a['truncation']}
sharpe: {m['sharpe']}
turnover: {m['turnover']}
fitness: {m['fitness']}
returns: {m['returns']}
drawdown: {m['drawdown']}
margin: {m['margin']}
status: submitted
rating: {a['rating']}
failure_modes: []
parent_alpha: {parent}
correlated_with: []
session: submitted_alphas_import_{a['created']}
hypothesis: "{a['hypothesis']}"
---

Submitted to WQ Brain on {a['created']}. **Rating: {a['rating']} (Regular Alpha)**.

**Year-by-year breakdown:**
| Year | Sharpe | Turnover | Fitness | Returns | Drawdown | Margin |
|------|--------|----------|---------|---------|----------|--------|
{year_table}

**Aggregate (5-year average):** Sharpe {m['sharpe']} · Turnover {m['turnover']}% · Fitness {m['fitness']} · Returns {m['returns']}% · MaxDD {m['drawdown']}% · Margin {m['margin']}‱
"""
    (OUT / f"{a['id']}.md").write_text(content, encoding="utf-8")
    print(f"  wrote {a['id']}.md  (avg Sharpe {m['sharpe']:>+5.2f}, Fitness {m['fitness']:>+5.2f}, rating {a['rating']})")

print(f"\nDone — {len(ALPHAS)} submitted alpha files written.")

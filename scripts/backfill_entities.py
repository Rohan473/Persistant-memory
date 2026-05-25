"""
Backfill entity node files (operators, failure_modes, settings) from alpha frontmatter.
Also fills gaps in concepts and datafields.
Run before build_graph.py.
"""

from pathlib import Path
from collections import defaultdict
import frontmatter

BASE = Path(__file__).resolve().parent.parent
ALPHAS_DIR = BASE / "nodes" / "alphas"

ENTITY_DIRS = {
    "concepts":      BASE / "nodes" / "concepts",
    "datafields":    BASE / "nodes" / "datafields",
    "operators":     BASE / "nodes" / "operators",
    "settings":      BASE / "nodes" / "settings",
    "failure_modes": BASE / "nodes" / "failure_modes",
}
for d in ENTITY_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ── Built-in descriptions ────────────────────────────────────────────────────
OPERATOR_DESC = {
    "rank": "Cross-sectional rank of values; maps to [0,1] uniform distribution. Most commonly used normalization in WQ Brain.",
    "ts_rank": "Time-series rank over a lookback window. `ts_rank(x, n)` ranks the current value against the past n values.",
    "group_neutralize": "Neutralize a signal within a group (sector, industry, etc.). Removes group-level mean from each stock.",
    "ts_mean": "Rolling time-series mean over n periods.",
    "ts_std": "Rolling time-series standard deviation over n periods.",
    "ts_delta": "Difference between current value and value n periods ago: `ts_delta(x, n) = x - x[n]`.",
    "ts_corr": "Rolling Pearson correlation between two series over n periods.",
    "ts_zscore": "Rolling z-score: `(x - ts_mean(x,n)) / ts_std(x,n)`.",
    "zscore": "Cross-sectional z-score normalization.",
    "decay_linear": "Linearly weighted moving average (alias for ts_decay_linear in some WQ Brain versions).",
    "ts_decay_linear": "Linearly weighted (decaying) moving average over n periods. Most recent value has highest weight.",
    "ts_decay_exp": "Exponentially weighted moving average over n periods.",
    "sign": "Returns +1, -1, or 0 depending on the sign of the input.",
    "abs": "Absolute value.",
    "log": "Natural logarithm.",
    "sqrt": "Square root.",
    "power": "Raise to a power.",
    "min": "Cross-sectional or element-wise minimum.",
    "max": "Cross-sectional or element-wise maximum.",
    "sum": "Sum over group or time window.",
    "stddev": "Standard deviation (alias for ts_std or cross-sectional std).",
    "regression_slope": "Slope of linear regression over n periods.",
    "regression_intercept": "Intercept of linear regression over n periods.",
    "indneutralize": "Industry-neutralize: subtract industry mean from each stock's value.",
    "correlation": "Pearson or rank correlation between two cross-sectional vectors.",
    "neutralize": "Generic neutralize operator; removes mean within a group.",
    "winsorize": "Clip extreme values to a specified percentile bound.",
    "quantile": "Map values to quantile buckets.",
    "percentage_change": "Percentage change from n periods ago.",
}

FAILURE_DESC = {
    "high_turnover": "Alpha turnover exceeds 70% cutoff. WQ Brain requires turnover below 70% to ensure the strategy is not too transaction-cost sensitive.",
    "low_fitness": "Fitness score below 1.0 cutoff. Fitness combines Sharpe and turnover; usually fails when Sharpe is low.",
    "low_sharpe": "In-sample Sharpe below 1.25 cutoff. The primary performance metric on WQ Brain.",
    "os_failure": "Out-of-sample Sharpe significantly lower than in-sample Sharpe, indicating overfitting or regime sensitivity.",
    "correlated": "Alpha's self-correlation (PnL correlation day-to-day) exceeds 0.7, or it is highly correlated with an existing submitted alpha.",
    "low_margin": "Margin (annualized return per dollar of position) below the required threshold.",
    "data_quality": "Issues with datafield availability, missing data, or unexpected NaN values causing simulation errors.",
    "overfitting": "Signal appears strong in-sample but collapses out-of-sample due to excessive parameter tuning.",
    "sector_bias": "Alpha returns are dominated by sector-level effects rather than stock-level signal, often seen with market neutralization on fundamental ratios.",
}

CONCEPT_DESC = {
    "mean_reversion": "Prices or signals that deviate from their historical mean tend to return to it. Basis for contrarian strategies.",
    "momentum": "Assets that have performed well recently continue to outperform. The basis for trend-following strategies.",
    "volatility": "Using realized or implied volatility as a signal or risk control in alpha construction.",
    "value": "Identifying cheap stocks relative to fundamentals (earnings, book value, cash flow). Classic Fama-French factor.",
    "quality": "Selecting stocks based on business quality metrics — profitability, earnings stability, low leverage.",
    "liquidity": "Using trading volume, ADV (average daily volume), or bid-ask spread as a signal or universe filter.",
    "sentiment": "Signals derived from analyst revisions, short interest, or news to capture market mood shifts.",
    "technical": "Price and volume based signals without fundamental data; chart patterns and statistical properties.",
    "fundamental": "Accounting-based signals from balance sheet, income statement, and cash flow data.",
    "cross_sectional": "Ranking or normalizing signals relative to all stocks in the universe at a given point in time.",
    "time_series": "Signals based on a single stock's own historical behavior over time.",
    "normalization": "Transforming raw signals to a standardized scale (rank, z-score, winsorize) for portfolio construction.",
    "neutralization": "Removing group-level (sector, industry, market) effects from a signal to isolate stock-specific alpha.",
}

DATAFIELD_DESC = {
    "close": "Daily closing price. Most fundamental price datafield in WQ Brain.",
    "open": "Daily opening price.",
    "high": "Daily high price.",
    "low": "Daily low price.",
    "volume": "Daily trading volume in shares.",
    "vwap": "Volume-weighted average price for the day.",
    "returns": "Daily returns: (close - prev_close) / prev_close.",
    "ebit": "Earnings Before Interest and Taxes. Quarterly fundamental datafield.",
    "capex": "Capital Expenditure — money spent on fixed assets. Quarterly fundamental.",
    "total_debt": "Total debt on the balance sheet. Quarterly fundamental.",
    "total_assets": "Total assets. Quarterly fundamental.",
    "book_value": "Book value of equity per share. Quarterly fundamental.",
    "equity": "Total shareholders' equity. Quarterly fundamental.",
    "revenue": "Total revenue / sales. Quarterly fundamental.",
    "net_income": "Net income after all expenses and taxes. Quarterly fundamental.",
    "adv": "Average Daily Volume — typically 20-day moving average of volume.",
    "cap": "Market capitalization.",
    "shares_outstanding": "Total shares outstanding.",
    "liabilities": "Total liabilities. Quarterly fundamental.",
    "current_liabilities": "Current (short-term) liabilities. Quarterly fundamental.",
    "short_interest": "Number of shares sold short. Proxy for bearish sentiment.",
    "dividend": "Dividend per share. Quarterly or annual fundamental.",
    "eps": "Earnings per share. Quarterly fundamental.",
    "sales": "Net sales / revenue. Quarterly fundamental.",
    "cashflow": "Operating cash flow. Quarterly fundamental.",
    "free_cashflow": "Free cash flow: operating cash flow minus capex.",
    "inventory": "Inventory value on balance sheet. Quarterly fundamental.",
    "receivables": "Accounts receivable. Quarterly fundamental.",
    "payables": "Accounts payable. Quarterly fundamental.",
    "depreciation": "Depreciation and amortization. Quarterly fundamental.",
    "beta": "Market beta — sensitivity of stock returns to market returns.",
    "pe_ratio": "Price-to-earnings ratio.",
    "pb_ratio": "Price-to-book ratio.",
    "ps_ratio": "Price-to-sales ratio.",
    "roa": "Return on assets: net_income / total_assets.",
    "roe": "Return on equity: net_income / equity.",
    "leverage": "Financial leverage: total_debt / equity.",
    "interest_expense": "Interest expense on debt. Quarterly fundamental.",
    "analyst_revision_rank_derivative": "Rate of change of analyst estimate revision rankings. Captures momentum in analyst sentiment.",
    "fcf": "Free cash flow (alias for free_cashflow in some WQ Brain contexts).",
}


def safe_list(val):
    if not val:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if v and str(v).strip() != "null"]
    if isinstance(val, str) and val not in ("null", ""):
        return [val.strip()]
    return []


def write_if_missing(path, content):
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    # Collect all entities from alphas
    all_operators = set()
    all_failure_modes = set()
    all_settings = set()   # (universe, delay, neutralization)
    all_concepts = set()
    all_datafields = set()

    alpha_files = sorted(ALPHAS_DIR.glob("alpha_*.md"))
    print(f"Scanning {len(alpha_files)} alpha files...")

    for af in alpha_files:
        try:
            post = frontmatter.load(str(af))
            m = post.metadata
            for op in safe_list(m.get("operators")):
                all_operators.add(op)
            for fm in safe_list(m.get("failure_modes")):
                all_failure_modes.add(fm)
            for c in safe_list(m.get("concepts")):
                all_concepts.add(c)
            for df in safe_list(m.get("datafields")):
                all_datafields.add(df)
            univ = str(m.get("universe") or "TOP3000")
            delay = str(m.get("delay") or "1")
            neut = str(m.get("neutralization") or "market")
            if univ != "null":
                all_settings.add((univ, delay, neut))
        except Exception as e:
            print(f"  WARN: {af.name}: {e}")

    # Backfill operators
    created = {"operators": 0, "failure_modes": 0, "settings": 0, "concepts": 0, "datafields": 0}

    for op in sorted(all_operators):
        p = ENTITY_DIRS["operators"] / f"{op}.md"
        desc = OPERATOR_DESC.get(op, f"WQ Brain operator `{op}`. Used in alpha expressions.")
        content = f"---\ntype: operator\nname: {op}\n---\n\n{desc}\n"
        if write_if_missing(p, content):
            created["operators"] += 1

    for fm in sorted(all_failure_modes):
        p = ENTITY_DIRS["failure_modes"] / f"{fm}.md"
        desc = FAILURE_DESC.get(fm, f"Failure mode: {fm}. Alpha did not meet WQ Brain submission criteria.")
        content = f"---\ntype: failure_mode\nname: {fm}\n---\n\n{desc}\n"
        if write_if_missing(p, content):
            created["failure_modes"] += 1

    for (univ, delay, neut) in sorted(all_settings):
        name = f"{univ}_{delay}_{neut}"
        p = ENTITY_DIRS["settings"] / f"{name}.md"
        content = (
            f"---\ntype: setting\nuniverse: {univ}\ndelay: {delay}\n"
            f"neutralization: {neut}\nname: {name}\n---\n\n"
            f"WQ Brain simulation setting: universe={univ}, delay={delay}, neutralization={neut}.\n"
        )
        if write_if_missing(p, content):
            created["settings"] += 1

    for c in sorted(all_concepts):
        p = ENTITY_DIRS["concepts"] / f"{c}.md"
        desc = CONCEPT_DESC.get(c, f"Trading concept: {c}.")
        content = f"---\ntype: concept\nname: {c}\n---\n\n{desc}\n"
        if write_if_missing(p, content):
            created["concepts"] += 1

    for df in sorted(all_datafields):
        p = ENTITY_DIRS["datafields"] / f"{df}.md"
        desc = DATAFIELD_DESC.get(df, f"WQ Brain datafield `{df}`.")
        content = f"---\ntype: datafield\nname: {df}\n---\n\n{desc}\n"
        if write_if_missing(p, content):
            created["datafields"] += 1

    print("\nEntity nodes created/confirmed:")
    for k, v in created.items():
        print(f"  {k:<20} {v:>3} new")

    # Summary of all entities found
    print(f"\nEntities scanned from alphas:")
    print(f"  operators      : {len(all_operators)} unique")
    print(f"  failure_modes  : {len(all_failure_modes)} unique")
    print(f"  settings       : {len(all_settings)} unique")
    print(f"  concepts       : {len(all_concepts)} unique")
    print(f"  datafields     : {len(all_datafields)} unique")


if __name__ == "__main__":
    main()

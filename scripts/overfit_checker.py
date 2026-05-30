"""
Overfit risk checker for WQB alpha expressions.

Checks an alpha for signs of overfitting using:
  1. Static expression analysis  — complexity, bad patterns, window tuning
  2. IS metric consistency       — Fitness/Sharpe ratio, drawdown efficiency
  3. Year-by-year Sharpe checks  — CV, trend, worst year (if data present)
  4. Coverage validation         — every datafield must be cov >= 0.75
  5. Iteration prior             — high sim count in a signal family = high overfit risk
  6. Live WQB data (--live)      — grade, selfCorrelation, osISSharpeRatio, IS check limits

Usage:
  python scripts/overfit_checker.py alpha_1206
  python scripts/overfit_checker.py alpha_1206 --live      # fetch live WQB grade + OS ratio
  python scripts/overfit_checker.py --all-recent 10 --live
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

ALPHAS_DIR = BASE / "private" / "nodes" / "alphas"
CATALOGUE_FILE = BASE / "memory_layer" / "brain_catalogue.json"

# ── thresholds ────────────────────────────────────────────────────────────────
MIN_COVERAGE          = 0.75   # fields below this → structural risk
FITNESS_SHARPE_WARN   = 0.65   # Fitness/Sharpe below this → inconsistent IS
FITNESS_SHARPE_FAIL   = 0.55   # hard fail
MAX_TURNOVER          = 0.45   # 45%
MAX_COMPLEXITY        = 18     # operator_count + 2*literal_count
MAX_WINDOWS           = 3      # distinct numeric time-windows in expression
SHARPE_CV_WARN        = 0.45   # annual Sharpe coefficient of variation
SHARPE_CV_FAIL        = 0.65
WORST_YEAR_SHARPE     = 0.5    # minimum acceptable annual Sharpe
SHARPE_WITHOUT_FIT_S  = 1.9    # Sharpe above this...
SHARPE_WITHOUT_FIT_F  = 1.05   # ...with Fitness below this = overfit signature

# Patterns that indicate IS-tuning of window parameters
SUSPICIOUS_WINDOWS = {5, 10, 20, 22, 60, 126, 240, 252}  # common IS-tuned values
# Having > MAX_WINDOWS distinct values from this set signals parameter search

BAD_PATTERNS = [
    ("ts_decay_linear", "ts_decay_linear smooths IS Sharpe without adding signal — known trap"),
]


# ── OOS survival scoring ─────────────────────────────────────────────────────

# Signal family → economic coherence score (0-10)
# Grounded in published factor literature
_ECO_COHERENCE_SIGNALS = {
    # Novy-Marx (2013) — gross profitability
    "gross_profit_to_assets":   9.0,
    "gross_profit_margin":      8.5,
    # Belo & Lin (2012) — inventory investment
    "inventory_turnover":       8.0,
    "inventory":                7.5,
    # Earnings quality / accruals (Sloan 1996, Dechow)
    "earningsquality":          8.5,
    "mdl177_2_earningsquality": 8.5,
    "mdl77_2earningsquality":   8.5,
    "accrual":                  8.0,
    # Value (Fama-French)
    "equity_value_score":       8.5,
    "mdl77_fangma":             8.0,
    "book_value":               8.0,
    # Momentum / contrarian (De Bondt & Thaler, Jegadeesh & Titman)
    "returns":                  7.5,
    "ts_sum":                   7.5,   # return reversal / momentum
    # Analyst revision / earnings surprise
    "anl4_epsr":                8.0,
    "analyst_revision":         8.0,
    "earnings_momentum":        8.0,
    "earnings_per_share":       7.5,
    # News / sentiment
    "sentiment_score":          7.5,
    "mean_composite_sentiment": 7.5,
    "scl12_buzz":               7.0,
    # Options / PCR
    "pcr_oi":                   6.5,
    # Volatility (Parkinson)
    "parkinson_volatility":     7.0,
    # Cash flow quality
    "cashflow_efficiency":      8.0,
    "cash_flow":                7.5,
    # Liquidity / short interest
    "short_interest":           7.0,
    "adv":                      6.5,
}

def _economic_coherence(datafields: list, expression: str, hypothesis: str) -> float:
    """Score economic coherence 0-10 based on signal family and hypothesis quality."""
    if not datafields:
        return 5.0
    scores = []
    expr_lower = expression.lower()
    hyp_lower  = (hypothesis or "").lower()

    for field in datafields:
        fl = field.lower()
        best = 5.0  # default — unknown field
        for key, score in _ECO_COHERENCE_SIGNALS.items():
            if key.lower() in fl:
                best = max(best, score)
        scores.append(best)

    base = sum(scores) / len(scores)

    # Bonus: hypothesis is non-trivial (> 30 words = thought through)
    if len(hyp_lower.split()) > 30:
        base = min(10.0, base + 0.5)
    # Penalty: pure price-only signal (no fundamental data)
    if all("close" in f or "returns" in f or "volume" in f or "open" in f
           for f in [d.lower() for d in datafields]):
        base = max(0.0, base - 1.5)
    return round(base, 2)


def _parameter_robustness(parsed: dict, failure_modes: list) -> float:
    """Score parameter robustness 0-10: fewer/less-specific parameters = more robust."""
    windows = parsed["windows"]
    n_windows = len(windows)
    n_ops = parsed["operator_count"]
    lits = [v for v in parsed["numeric_literals"]
            if v not in (0, 1, 2, 0.5, 0.08, 2.5, 0.000001)]

    # Base score from window count
    if n_windows == 0:
        score = 9.5
    elif n_windows == 1:
        score = 8.5
    elif n_windows == 2:
        score = 7.0
    elif n_windows == 3:
        score = 5.5
    else:
        score = 4.0

    # Penalty: ts_decay_linear (documented trap)
    if any("ts_decay_linear" in str(op) for op in parsed["operators"]):
        score -= 2.0
    # Penalty: many specific weight literals (e.g. 0.375, 0.55 suggests tuning)
    suspicious_weights = [v for v in lits if 0.0 < v < 1.0 and v not in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75)]
    score -= min(2.0, len(suspicious_weights) * 0.5)
    # Penalty: many operators
    if n_ops > 6:
        score -= 1.0
    elif n_ops > 4:
        score -= 0.5

    return round(max(0.0, min(10.0, score)), 2)


def _regime_robustness(yearly_rows: list, sharpe: float, fitness: float) -> float:
    """Score regime robustness 0-10 from annual Sharpe consistency."""
    if yearly_rows:
        sharpes = [r["sharpe"] for r in yearly_rows]
        mean_s = sum(sharpes) / len(sharpes)
        worst  = min(sharpes)
        std_s  = (sum((s - mean_s)**2 for s in sharpes) / len(sharpes)) ** 0.5
        cv     = std_s / mean_s if mean_s > 0 else 999

        # Worst-year floor
        if worst < 0:
            floor_score = 1.0
        elif worst < 0.5:
            floor_score = 3.0
        elif worst < 1.0:
            floor_score = 6.0
        elif worst < 1.5:
            floor_score = 8.0
        else:
            floor_score = 10.0

        # Consistency bonus/penalty
        if cv < 0.25:
            consistency = 10.0
        elif cv < 0.45:
            consistency = 8.0
        elif cv < 0.65:
            consistency = 6.0
        else:
            consistency = 4.0

        return round((floor_score + consistency) / 2, 2)
    else:
        # Proxy from Fitness/Sharpe ratio when no yearly data
        if not sharpe:
            return 5.0
        fs_ratio = (fitness or 0) / sharpe
        return round(min(10.0, fs_ratio * 10), 2)


def _coverage_quality(datafields: list) -> float:
    """Score coverage quality 0-10: average field coverage across datafields."""
    catalogue = _load_catalogue()
    SKIP_TOKENS = {"std", "filter", "true", "false", "on", "off"}
    if not catalogue or not datafields:
        return 5.0
    coverages = []
    for f in datafields:
        if f in SKIP_TOKENS:
            continue
        cov = catalogue.get(f)
        if cov is not None:
            coverages.append(cov)
    if not coverages:
        return 5.0
    avg_cov = sum(coverages) / len(coverages)
    return round(avg_cov * 10, 2)


def _universe_stability(failure_modes: list, self_corr: float, sharpe: float) -> float:
    """Score universe stability 0-10 from sub_universe_failure and self-correlation."""
    score = 8.0
    if "sub_universe_failure" in (failure_modes or []):
        score -= 3.0
    if self_corr > 0.70:
        score -= 3.0
    elif self_corr > 0.55:
        score -= 1.5
    elif self_corr > 0.35:
        score -= 0.5
    return round(max(0.0, min(10.0, score)), 2)


def _complexity_penalty(parsed: dict) -> float:
    """Return complexity level 0-10 (higher = more complex = bigger penalty)."""
    n_ops  = parsed["operator_count"]
    n_lits = len([v for v in parsed["numeric_literals"]
                  if v not in (0, 1, 2, 0.5, 0.08, 2.5, 0.000001)])
    n_win  = len(parsed["windows"])
    raw = n_ops + 2 * n_lits + n_win
    return round(min(10.0, raw / 2.5), 2)


OOS_WEIGHTS = {
    "economic_coherence":   0.30,
    "parameter_robustness": 0.20,
    "regime_robustness":    0.25,
    "coverage_quality":     0.15,
    "universe_stability":   0.10,
}

def compute_oos_scores(
    datafields: list,
    expression: str,
    hypothesis: str,
    parsed: dict,
    failure_modes: list,
    yearly_rows: list,
    sharpe: float,
    fitness: float,
    self_corr: float,
) -> dict:
    eco   = _economic_coherence(datafields, expression, hypothesis)
    param = _parameter_robustness(parsed, failure_modes)
    regime = _regime_robustness(yearly_rows, sharpe, fitness)
    cov   = _coverage_quality(datafields)
    univ  = _universe_stability(failure_modes, self_corr, sharpe)
    comp  = _complexity_penalty(parsed)

    oos_score = sum(
        OOS_WEIGHTS[k] * v
        for k, v in [
            ("economic_coherence",   eco),
            ("parameter_robustness", param),
            ("regime_robustness",    regime),
            ("coverage_quality",     cov),
            ("universe_stability",   univ),
        ]
    ) - 0.05 * comp   # subtract complexity penalty

    oos_score = round(max(0.0, min(10.0, oos_score)), 2)

    if oos_score >= 7.5:
        confidence = "HIGH — likely to survive OOS"
    elif oos_score >= 5.5:
        confidence = "MEDIUM — uncertain OOS"
    else:
        confidence = "LOW — probably overfit"

    return {
        "economic_coherence":   eco,
        "parameter_robustness": param,
        "regime_robustness":    regime,
        "coverage_quality":     cov,
        "universe_stability":   univ,
        "complexity_penalty":   comp,
        "oos_survival_score":   oos_score,
        "confidence":           confidence,
    }


# ── live WQB fetch ────────────────────────────────────────────────────────────

def _fetch_live(remote_alpha_id: str) -> dict:
    """Fetch live grade, train/test comparison, selfCorrelation, osISSharpeRatio from WQB API.
    Tries competition endpoint first (which always has train/test), falls back to /alphas/{id}."""
    if not remote_alpha_id:
        return {}
    try:
        from memory_layer.brain_api import BrainAPIClient
        client = BrainAPIClient.from_disk()

        # Try competition endpoint first — it always returns train/test blocks
        d = None
        for comp in ["IQC2026S2", "IQC2026S1"]:
            try:
                r = client._request("GET", f"/competitions/{comp}/alphas?limit=100")
                for alpha in r.json().get("results", []):
                    if alpha.get("id") == remote_alpha_id:
                        d = alpha
                        break
                if d:
                    break
            except Exception:
                continue

        if not d:
            r = client._request("GET", f"/alphas/{remote_alpha_id}")
            d = r.json()
        is_block    = d.get("is") or {}
        os_block    = d.get("os") or {}
        train_block = d.get("train") or {}
        test_block  = d.get("test") or {}

        grade     = d.get("grade", "UNKNOWN")
        self_corr = is_block.get("selfCorrelation") or 0
        os_is_ratio = os_block.get("osISSharpeRatio")

        # Train/test performance comparison — the actual score change signal
        train_sharpe  = train_block.get("sharpe")
        test_sharpe   = test_block.get("sharpe")
        train_fitness = train_block.get("fitness")
        test_fitness  = test_block.get("fitness")

        score_change = None
        test_train_ratio = None
        if train_sharpe and test_sharpe:
            score_change     = round(test_sharpe - train_sharpe, 3)
            test_train_ratio = round(test_sharpe / train_sharpe, 3)

        return {
            "grade":             grade,
            "stage":             d.get("stage"),
            "status":            d.get("status"),
            "self_corr":         round(self_corr, 4),
            "prod_corr":         round(is_block.get("prodCorrelation") or 0, 4),
            "os_is_ratio":       os_is_ratio,
            "train_sharpe":      train_sharpe,
            "train_fitness":     train_fitness,
            "test_sharpe":       test_sharpe,
            "test_fitness":      test_fitness,
            "score_change":      score_change,       # test - train Sharpe
            "test_train_ratio":  test_train_ratio,   # test / train Sharpe
            "os_checks":         {c["name"]: c["result"] for c in (os_block.get("checks") or [])},
            "has_test_period":   bool(test_block),
        }
    except Exception as e:
        return {"error": str(e)}


# ── catalogue loader ──────────────────────────────────────────────────────────
_catalogue_cache: dict | None = None

def _load_catalogue() -> dict[str, float]:
    """Returns {field_id: coverage} for TOP3000."""
    global _catalogue_cache
    if _catalogue_cache is not None:
        return _catalogue_cache
    if not CATALOGUE_FILE.exists():
        return {}
    raw = json.loads(CATALOGUE_FILE.read_text(encoding="utf-8"))
    fields = raw.get("datafields", {}).get("USA|1|TOP3000", [])
    _catalogue_cache = {f["id"]: f.get("coverage", 0.0) for f in fields}
    return _catalogue_cache


# ── expression parser ─────────────────────────────────────────────────────────
OPERATORS = {
    "add", "subtract", "multiply", "divide", "abs", "sign", "signed_power", "power",
    "sqrt", "log", "inverse", "reverse", "normalize", "winsorize", "rank", "zscore",
    "group_rank", "group_zscore", "group_neutralize", "group_mean", "group_scale",
    "group_backfill", "ts_mean", "ts_sum", "ts_delta", "ts_delay", "ts_zscore",
    "ts_rank", "ts_std_dev", "ts_corr", "ts_backfill", "ts_decay_linear",
    "ts_regression", "ts_covariance", "ts_scale", "ts_arg_max", "ts_arg_min",
    "ts_count_nans", "ts_av_diff", "ts_product", "ts_quantile", "ts_step",
    "hump", "bucket", "quantile", "scale", "densify", "days_from_last_change",
    "last_diff_value", "kth_element", "if_else", "trade_when", "max", "min",
    "is_nan", "and", "or", "not", "equal", "not_equal", "greater", "greater_equal",
    "less", "less_equal", "vec_avg", "vec_sum",
}

def _parse_expression(expr: str) -> dict:
    """Extract complexity metrics from expression string."""
    used_ops = [op for op in OPERATORS if re.search(rf'\b{op}\b', expr)]

    # Numeric literals (integers and decimals, not part of identifiers)
    literals = re.findall(r'(?<![_\w])\d+\.?\d*(?![_\w])', expr)
    numeric_vals = [float(v) for v in literals]

    # Time-window arguments: numbers appearing inside ts_* / hump / bucket calls
    window_args = re.findall(
        r'(?:ts_mean|ts_sum|ts_delta|ts_delay|ts_zscore|ts_rank|ts_std_dev|'
        r'ts_backfill|ts_decay_linear|ts_corr|ts_covariance|ts_scale|'
        r'ts_regression|ts_av_diff|ts_product|ts_quantile|ts_arg_max|ts_arg_min|'
        r'ts_count_nans)\s*\([^,)]+,\s*(\d+)',
        expr,
    )
    windows = set(int(w) for w in window_args)

    suspicious = windows & SUSPICIOUS_WINDOWS

    complexity = len(used_ops) + 2 * len([v for v in numeric_vals
                                           if v not in (0, 1, 0.5, 2, 4, 0.08, 2.5)])

    return {
        "operators": used_ops,
        "operator_count": len(used_ops),
        "numeric_literals": numeric_vals,
        "windows": sorted(windows),
        "suspicious_windows": sorted(suspicious),
        "complexity_score": complexity,
    }


# ── yearly parser ─────────────────────────────────────────────────────────────
def _parse_yearly(body: str) -> list[dict]:
    """Extract per-year Sharpe from the markdown table in the alpha body."""
    rows = []
    in_table = False
    for line in body.splitlines():
        if "Year" in line and "Sharpe" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) < 2:
                continue
            try:
                year = int(parts[0])
                sharpe = float(parts[1].lstrip("+"))
                rows.append({"year": year, "sharpe": sharpe})
            except ValueError:
                continue
    return rows


def _yearly_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    sharpes = [r["sharpe"] for r in rows]
    mean_s = sum(sharpes) / len(sharpes)
    variance = sum((s - mean_s) ** 2 for s in sharpes) / len(sharpes)
    std_s = variance ** 0.5
    cv = std_s / mean_s if mean_s > 0 else 999
    # Linear trend: negative slope = declining
    n = len(sharpes)
    if n >= 3:
        xs = list(range(n))
        xm = sum(xs) / n
        ym = mean_s
        slope = sum((xs[i] - xm) * (sharpes[i] - ym) for i in range(n)) / \
                sum((xs[i] - xm) ** 2 for i in range(n))
    else:
        slope = 0.0
    return {
        "mean": round(mean_s, 3),
        "std": round(std_s, 3),
        "cv": round(cv, 3),
        "slope_per_year": round(slope, 3),
        "worst_year": min(rows, key=lambda r: r["sharpe"]),
        "best_year": max(rows, key=lambda r: r["sharpe"]),
        "years": rows,
    }


# ── main checker ──────────────────────────────────────────────────────────────
CHECK_PASS = "PASS"
CHECK_WARN = "WARN"
CHECK_FAIL = "FAIL"
CHECK_SKIP = "SKIP"

RISK_LOW    = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH   = "HIGH"


def check_alpha(alpha_id: str, verbose: bool = False, live: bool = False) -> dict:
    path = ALPHAS_DIR / f"{alpha_id}.md"
    if not path.exists():
        print(f"  [ERROR] {alpha_id}.md not found in {ALPHAS_DIR}", file=sys.stderr)
        return {}

    raw = path.read_text(encoding="utf-8")

    # Parse frontmatter
    import frontmatter as fm
    post = fm.loads(raw)
    meta = post.metadata
    body = post.content

    expr        = meta.get("expression", "")
    sharpe      = meta.get("sharpe") or 0.0
    fitness     = meta.get("fitness") or 0.0
    turnover    = meta.get("turnover") or 0.0
    drawdown    = meta.get("drawdown") or 0.0
    returns_pct = meta.get("returns") or 0.0
    failure_modes = meta.get("failure_modes") or []
    datafields  = meta.get("datafields") or []

    checks = []
    fails = 0
    warns = 0

    def add(name: str, status: str, detail: str):
        nonlocal fails, warns
        checks.append({"name": name, "status": status, "detail": detail})
        if status == CHECK_FAIL:
            fails += 1
        elif status == CHECK_WARN:
            warns += 1

    # ── 1. Static expression analysis ────────────────────────────────────────
    parsed = _parse_expression(expr)

    # Complexity
    c = parsed["complexity_score"]
    if c > MAX_COMPLEXITY:
        add("COMPLEXITY", CHECK_FAIL,
            f"Score {c} > {MAX_COMPLEXITY} — {parsed['operator_count']} operators + "
            f"tunable literals suggests over-engineering")
    elif c > MAX_COMPLEXITY * 0.75:
        add("COMPLEXITY", CHECK_WARN,
            f"Score {c} — borderline complexity ({parsed['operator_count']} operators)")
    else:
        add("COMPLEXITY", CHECK_PASS,
            f"Score {c} — {parsed['operator_count']} operators, acceptable")

    # Bad patterns
    for pattern, reason in BAD_PATTERNS:
        if pattern in expr:
            add(f"BAD_PATTERN:{pattern}", CHECK_FAIL, reason)

    # Window tuning
    suspicious = parsed["suspicious_windows"]
    n_windows = len(parsed["windows"])
    if n_windows > MAX_WINDOWS and len(suspicious) > MAX_WINDOWS:
        add("WINDOW_TUNING", CHECK_FAIL,
            f"{n_windows} distinct time-windows {parsed['windows']} — "
            f"{len(suspicious)} from known IS-tuned set {suspicious}")
    elif n_windows > MAX_WINDOWS:
        add("WINDOW_TUNING", CHECK_WARN,
            f"{n_windows} distinct time-windows {parsed['windows']}")
    else:
        add("WINDOW_TUNING", CHECK_PASS,
            f"{n_windows} time-windows {parsed['windows']}")

    # ── 2. IS metric consistency ──────────────────────────────────────────────
    if sharpe and fitness:
        fs_ratio = fitness / sharpe
        if fs_ratio < FITNESS_SHARPE_FAIL:
            add("FITNESS_SHARPE_RATIO", CHECK_FAIL,
                f"Fitness/Sharpe = {fs_ratio:.2f} < {FITNESS_SHARPE_FAIL} — "
                f"IS returns are not consistent year-over-year")
        elif fs_ratio < FITNESS_SHARPE_WARN:
            add("FITNESS_SHARPE_RATIO", CHECK_WARN,
                f"Fitness/Sharpe = {fs_ratio:.2f} — borderline consistency")
        else:
            add("FITNESS_SHARPE_RATIO", CHECK_PASS,
                f"Fitness/Sharpe = {fs_ratio:.2f} — consistent IS performance")
    else:
        add("FITNESS_SHARPE_RATIO", CHECK_SKIP, "No IS metrics available")

    # Sharpe-without-Fitness overfit signature
    if sharpe >= SHARPE_WITHOUT_FIT_S and fitness <= SHARPE_WITHOUT_FIT_F:
        add("SHARPE_WITHOUT_FITNESS", CHECK_FAIL,
            f"Sharpe {sharpe} >= {SHARPE_WITHOUT_FIT_S} with Fitness {fitness} <= "
            f"{SHARPE_WITHOUT_FIT_F} — classic parameter-fit signature")
    else:
        add("SHARPE_WITHOUT_FITNESS", CHECK_PASS,
            f"Sharpe {sharpe} / Fitness {fitness} — no red flag")

    # High turnover
    if turnover > MAX_TURNOVER * 100:
        add("TURNOVER", CHECK_FAIL,
            f"Turnover {turnover}% > {MAX_TURNOVER*100}% — high-churn IS Sharpe often collapses OS")
    elif turnover > MAX_TURNOVER * 80:
        add("TURNOVER", CHECK_WARN, f"Turnover {turnover}% — approaching limit")
    else:
        add("TURNOVER", CHECK_PASS, f"Turnover {turnover}% — healthy")

    # Drawdown efficiency
    if returns_pct and drawdown:
        dd_eff = returns_pct / drawdown
        if dd_eff < 1.2:
            add("DRAWDOWN_EFFICIENCY", CHECK_FAIL,
                f"Returns/Drawdown = {dd_eff:.2f} — too much drawdown for returns earned")
        elif dd_eff < 1.8:
            add("DRAWDOWN_EFFICIENCY", CHECK_WARN,
                f"Returns/Drawdown = {dd_eff:.2f} — borderline")
        else:
            add("DRAWDOWN_EFFICIENCY", CHECK_PASS,
                f"Returns/Drawdown = {dd_eff:.2f}")

    # Sub-universe failure (structural weakness)
    if "sub_universe_failure" in failure_modes:
        add("SUB_UNIVERSE", CHECK_WARN,
            "sub_universe_failure present — portfolio too thin, Fitness capped structurally")
    else:
        add("SUB_UNIVERSE", CHECK_PASS, "No sub_universe_failure")

    # ── 3. Year-by-year Sharpe consistency ───────────────────────────────────
    yearly = _parse_yearly(body)
    if yearly:
        ystats = _yearly_stats(yearly)
        cv = ystats["cv"]
        slope = ystats["slope_per_year"]
        worst = ystats["worst_year"]

        if cv > SHARPE_CV_FAIL:
            add("YEARLY_SHARPE_CV", CHECK_FAIL,
                f"Annual Sharpe CV = {cv:.2f} — highly inconsistent year-to-year "
                f"(mean {ystats['mean']}, std {ystats['std']})")
        elif cv > SHARPE_CV_WARN:
            add("YEARLY_SHARPE_CV", CHECK_WARN,
                f"Annual Sharpe CV = {cv:.2f} — some inconsistency")
        else:
            add("YEARLY_SHARPE_CV", CHECK_PASS,
                f"Annual Sharpe CV = {cv:.2f} — consistent")

        if slope < -0.15:
            add("SHARPE_TREND", CHECK_FAIL,
                f"Sharpe declining {slope:.2f}/year — signal decaying (potential data-mining)")
        elif slope < -0.05:
            add("SHARPE_TREND", CHECK_WARN,
                f"Sharpe trend {slope:.2f}/year — mild decay")
        else:
            add("SHARPE_TREND", CHECK_PASS,
                f"Sharpe trend {slope:.2f}/year — stable or improving")

        if worst["sharpe"] < WORST_YEAR_SHARPE:
            add("WORST_YEAR", CHECK_FAIL,
                f"Worst year {worst['year']}: Sharpe {worst['sharpe']} < {WORST_YEAR_SHARPE} "
                f"— fails in at least one market regime")
        else:
            add("WORST_YEAR", CHECK_PASS,
                f"Worst year {worst['year']}: Sharpe {worst['sharpe']} — acceptable floor")
    else:
        add("YEARLY_SHARPE_CV", CHECK_SKIP,
            "No year-by-year data in alpha file (not returned by current WQB API)")
        add("SHARPE_TREND",     CHECK_SKIP, "No yearly data")
        add("WORST_YEAR",       CHECK_SKIP, "No yearly data")

    # ── 4. Coverage validation ────────────────────────────────────────────────
    catalogue = _load_catalogue()
    # Strip non-field tokens that leak into the datafields list
    SKIP_TOKENS = {"std", "filter", "true", "false", "on", "off"}
    if catalogue and datafields:
        low_cov = []
        for field in datafields:
            if field in SKIP_TOKENS:
                continue
            cov = catalogue.get(field)
            if cov is None:
                add(f"COVERAGE:{field}", CHECK_WARN,
                    f"{field} not found in catalogue — coverage unknown")
            elif cov < MIN_COVERAGE:
                low_cov.append((field, cov))
        if low_cov:
            detail = "; ".join(f"{f}={c:.2f}" for f, c in low_cov)
            add("COVERAGE", CHECK_FAIL,
                f"Fields below {MIN_COVERAGE} coverage: {detail} — sub_universe_failure risk")
        else:
            add("COVERAGE", CHECK_PASS,
                f"All {len(datafields)} datafields above {MIN_COVERAGE} coverage")
    else:
        add("COVERAGE", CHECK_SKIP, "Catalogue not loaded or no datafields recorded")

    # ── 5. Iteration prior ────────────────────────────────────────────────────
    try:
        alpha_num = int(re.search(r'(\d+)', alpha_id).group(1))
        # Count alphas in the same rough signal family by concept overlap
        concepts = set(meta.get("concepts") or [])
        related = []
        for f in ALPHAS_DIR.glob("alpha_*.md"):
            try:
                other = f.read_text(encoding="utf-8")
                m_start = other.index("---") + 3
                m_end = other.index("---", m_start)
                import yaml
                other_meta = yaml.safe_load(other[m_start:m_end]) or {}
                other_concepts = set(other_meta.get("concepts") or [])
                other_num = int(re.search(r'(\d+)', f.stem).group(1))
                if other_num < alpha_num and len(concepts & other_concepts) >= 3:
                    related.append(f.stem)
            except Exception:
                continue
        n_related = len(related)
        if n_related >= 10:
            add("ITERATION_PRIOR", CHECK_FAIL,
                f"{n_related} related alphas tested before this one — "
                f"high IS-search pressure on this signal family")
        elif n_related >= 5:
            add("ITERATION_PRIOR", CHECK_WARN,
                f"{n_related} related alphas tested before this — moderate search pressure")
        else:
            add("ITERATION_PRIOR", CHECK_PASS,
                f"{n_related} related alphas — low search pressure")
    except Exception as e:
        add("ITERATION_PRIOR", CHECK_SKIP, f"Could not compute: {e}")

    # ── Overall risk score ────────────────────────────────────────────────────
    if fails >= 2:
        risk = RISK_HIGH
    elif fails == 1 or warns >= 3:
        risk = RISK_MEDIUM
    else:
        risk = RISK_LOW

    # ── 6. Live WQB data (optional) ───────────────────────────────────────────
    live_data = {}
    if live:
        remote_id = meta.get("remote_alpha_id") or ""
        if remote_id:
            live_data = _fetch_live(remote_id)
            if "error" not in live_data:
                # Test period Sharpe = the "score" in WQB performance comparison
                # This is the qualification metric for Stage 3
                test_sharpe = live_data.get("test_sharpe")
                train_sharpe = live_data.get("train_sharpe")
                ratio = live_data.get("test_train_ratio")
                change = live_data.get("score_change")
                if not live_data.get("has_test_period"):
                    add("SCORE_CHANGE", CHECK_SKIP,
                        "No test period data — testPeriod not set at sim time (fixed for future sims)")
                elif test_sharpe is not None:
                    sign = "+" if change and change >= 0 else ""
                    detail = (f"Score (test Sharpe) = {test_sharpe}  "
                              f"[train={train_sharpe}, change={sign}{change}, ratio={ratio}]")
                    if test_sharpe < 0:
                        add("SCORE_CHANGE", CHECK_FAIL,
                            f"NEGATIVE test Sharpe = {test_sharpe} — alpha loses money OOS. {detail}")
                    elif test_sharpe < 1.0:
                        add("SCORE_CHANGE", CHECK_FAIL,
                            f"Score {test_sharpe} < 1.0 — below IS gate threshold OOS. {detail}")
                    elif test_sharpe < 1.25:
                        add("SCORE_CHANGE", CHECK_WARN,
                            f"Score {test_sharpe} — marginal OOS Sharpe. {detail}")
                    else:
                        add("SCORE_CHANGE", CHECK_PASS,
                            f"Score {test_sharpe} — solid OOS performance. {detail}")

                os_ratio = live_data.get("os_is_ratio")
                if os_ratio is not None:
                    # OS/IS ratio is the definitive overfit signal
                    if os_ratio < 0.5:
                        add("OS_IS_RATIO", CHECK_FAIL,
                            f"OS/IS Sharpe ratio = {os_ratio:.2f} — severe IS overfit "
                            f"(OS Sharpe < half of IS Sharpe)")
                    elif os_ratio < 0.75:
                        add("OS_IS_RATIO", CHECK_WARN,
                            f"OS/IS Sharpe ratio = {os_ratio:.2f} — moderate degradation OS")
                    else:
                        add("OS_IS_RATIO", CHECK_PASS,
                            f"OS/IS Sharpe ratio = {os_ratio:.2f} — holds up OS")
                else:
                    add("OS_IS_RATIO", CHECK_SKIP,
                        "OS checks still PENDING — check back in weeks")

                self_corr = live_data.get("self_corr", 0)
                if self_corr > 0.70:
                    add("SELF_CORRELATION", CHECK_FAIL,
                        f"selfCorrelation = {self_corr} > 0.70 — will FAIL OS self-corr check")
                elif self_corr > 0.55:
                    add("SELF_CORRELATION", CHECK_WARN,
                        f"selfCorrelation = {self_corr} — approaching 0.70 limit")
                else:
                    add("SELF_CORRELATION", CHECK_PASS,
                        f"selfCorrelation = {self_corr} — safe")
        else:
            live_data = {"error": "No remote_alpha_id in frontmatter"}

    # ── OOS Survival Score ────────────────────────────────────────────────────
    live_self_corr = 0.0
    if live_data and "self_corr" in live_data:
        live_self_corr = live_data["self_corr"]

    oos = compute_oos_scores(
        datafields   = datafields,
        expression   = expr,
        hypothesis   = meta.get("hypothesis", ""),
        parsed       = parsed,
        failure_modes = failure_modes,
        yearly_rows  = yearly,
        sharpe       = sharpe,
        fitness      = fitness,
        self_corr    = live_self_corr,
    )

    return {
        "alpha_id": alpha_id,
        "risk": risk,
        "fails": fails,
        "warns": warns,
        "checks": checks,
        "expression": expr,
        "live": live_data,
        "oos": oos,
        "metrics": {
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "drawdown": drawdown,
            "returns": returns_pct,
        },
    }


# ── printer ───────────────────────────────────────────────────────────────────
STATUS_ICON = {CHECK_PASS: "OK  ", CHECK_WARN: "WARN", CHECK_FAIL: "FAIL", CHECK_SKIP: "SKIP"}
RISK_COLOR  = {RISK_LOW: "LOW", RISK_MEDIUM: "MEDIUM", RISK_HIGH: "HIGH"}

def print_report(result: dict) -> None:
    if not result:
        return
    aid  = result["alpha_id"]
    risk = result["risk"]
    m    = result["metrics"]

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  OVERFIT CHECK — {aid}")
    print(f"  Sharpe={m['sharpe']}  Fitness={m['fitness']}  TO={m['turnover']}%")
    print(f"  RISK: {risk}  ({result['fails']} FAIL / {result['warns']} WARN)")
    print(bar)

    for c in result["checks"]:
        icon = STATUS_ICON.get(c["status"], "????")
        print(f"  [{icon}] {c['name']}")
        print(f"          {c['detail']}")

    # Live WQB section
    live = result.get("live") or {}
    if live and "error" not in live:
        print()
        print(f"  --- LIVE WQB DATA ---")
        grade    = live.get("grade", "?")
        stage    = live.get("stage", "?")
        sc       = live.get("self_corr", "?")
        os_ratio = live.get("os_is_ratio")
        print(f"  Grade: {grade:<10} Stage: {stage}  selfCorr: {sc}")

        # Performance comparison: train vs test
        if live.get("has_test_period") and live.get("train_sharpe"):
            tc = live.get("score_change")
            tr = live.get("test_train_ratio")
            sign = "+" if tc and tc >= 0 else ""
            print(f"  Performance comparison:")
            print(f"    Train Sharpe: {live['train_sharpe']}  Fitness: {live['train_fitness']}")
            print(f"    Test  Sharpe: {live['test_sharpe']}  Fitness: {live['test_fitness']}")
            print(f"    Score change: {sign}{tc}  (test/train ratio: {tr})")
        else:
            print(f"  Performance comparison: no test period data")

        os_ratio_str = f"{os_ratio:.2f}" if os_ratio is not None else "PENDING"
        print(f"  OS/IS Sharpe ratio: {os_ratio_str}")
        os_checks = live.get("os_checks", {})
        if os_checks:
            statuses = "  ".join(f"{k}:{v}" for k, v in os_checks.items())
            print(f"  OS checks: {statuses}")
    elif live.get("error"):
        print(f"\n  [live] {live['error']}")

    # OOS Survival Score
    oos = result.get("oos") or {}
    if oos:
        print()
        print("  --- OOS SURVIVAL SCORE ---")
        print(f"  Economic coherence   : {oos['economic_coherence']:>5.1f} / 10  (signal has academic/econ backing)")
        print(f"  Parameter robustness : {oos['parameter_robustness']:>5.1f} / 10  (fewer tuned params = more robust)")
        print(f"  Regime robustness    : {oos['regime_robustness']:>5.1f} / 10  (works across market regimes)")
        print(f"  Coverage quality     : {oos['coverage_quality']:>5.1f} / 10  (field coverage >= 0.75 = full marks)")
        print(f"  Universe stability   : {oos['universe_stability']:>5.1f} / 10  (no sub_universe_failure, low self-corr)")
        print(f"  Complexity penalty   : {oos['complexity_penalty']:>5.1f} / 10  (lower = simpler = less penalized)")
        score = oos['oos_survival_score']
        conf  = oos['confidence']
        print(f"  {'-'*42}")
        print(f"  OOS Survival Score   : {score:>5.1f} / 10  {conf}")

    print(bar)
    if risk == RISK_HIGH:
        print("  VERDICT: Do NOT submit — likely to underperform OS")
    elif risk == RISK_MEDIUM:
        print("  VERDICT: Caution — review flagged items before submitting")
    else:
        print("  VERDICT: Acceptable — no major overfit signals detected")
    print(bar + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="WQB alpha overfit risk checker")
    ap.add_argument("alpha_ids", nargs="*", help="Alpha IDs to check (e.g. alpha_1206)")
    ap.add_argument("--all-recent", type=int, metavar="N",
                    help="Check the N most recently modified alpha files")
    ap.add_argument("--live", action="store_true",
                    help="Fetch live grade/selfCorr/osISSharpeRatio from WQB API")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ids = list(args.alpha_ids)
    if args.all_recent:
        files = sorted(ALPHAS_DIR.glob("alpha_*.md"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        ids = [f.stem for f in files[: args.all_recent]]

    if not ids:
        ap.print_help()
        sys.exit(1)

    for aid in ids:
        result = check_alpha(aid, verbose=args.verbose, live=args.live)
        print_report(result)


if __name__ == "__main__":
    main()

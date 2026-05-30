"""
Factor Ontology Engine
Canonical factor taxonomy with automatic alpha classification.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


@dataclass
class FactorFamily:
    """Canonical factor family with classification rules."""
    id: str
    name: str
    display_name: str
    category: str
    description: str
    keywords: List[str]
    datafield_indicators: List[str]
    operator_indicators: List[str]
    macro_related: bool = False
    sector_related: bool = False


@dataclass
class AlphaExposure:
    """Detected factor exposure for an alpha."""
    alpha_id: str
    factor_family: str
    confidence: float
    indicator: str
    direction: str  # long, short, neutral


@dataclass
class MacroTheme:
    """Macro-economic theme."""
    id: str
    name: str
    description: str
    related_factors: List[str]
    regime_indicators: List[str]


FACTOR_FAMILIES = {
    "quality": FactorFamily(
        id="quality",
        name="quality",
        display_name="Quality",
        category="fundamental",
        description="High quality companies with strong balance sheets, profitability, and low leverage",
        keywords=["profitability", "ROE", "ROA", "gross_margin", "debt_ratio", "fscore", "quality"],
        datafield_indicators=["ebit", "net_income", "assets", "fnd2_ebitdm", "fnd2_ebitfr"],
        operator_indicators=["rank", "group_rank"],
        macro_related=False,
        sector_related=False
    ),
    "value": FactorFamily(
        id="value",
        name="value",
        display_name="Value",
        category="fundamental",
        description="Cheap securities relative to intrinsic measures",
        keywords=["value", "valuation", "cheap", "book", "ebit_ev", "price_earnings", "bp", "capex"],
        datafield_indicators=["ebit", "capex", "revenue", "assets"],
        operator_indicators=["rank", "group_rank", "zscore"],
        macro_related=True,
        sector_related=False
    ),
    "momentum": FactorFamily(
        id="momentum",
        name="momentum",
        display_name="Momentum",
        category="price_action",
        description="Past returns predict future returns - trend following",
        keywords=["momentum", "trend", "past_performance", "relative_strength", "returns"],
        datafield_indicators=["returns", "close"],
        operator_indicators=["ts_rank", "ts_mean", "ts_sum", "ts_corr"],
        macro_related=False,
        sector_related=False
    ),
    "reversal": FactorFamily(
        id="reversal",
        name="reversal",
        display_name="Reversal",
        category="price_action",
        description="Mean reversion - prices revert to historical averages",
        keywords=["reversal", "mean_reversion", "contrarian", "short_term", "oversold"],
        datafield_indicators=["close", "returns", "volume"],
        operator_indicators=["rank", "zscore", "ts_mean", "ts_std_dev"],
        macro_related=False,
        sector_related=False
    ),
    "liquidity": FactorFamily(
        id="liquidity",
        name="liquidity",
        display_name="Liquidity",
        category="market",
        description="Liquidity provision and trading volume effects",
        keywords=["liquidity", "volume", "turnover", "spread", "bid_ask", "adv"],
        datafield_indicators=["volume", "adv20", "vwap", "close"],
        operator_indicators=["rank", "ts_mean", "group_rank"],
        macro_related=True,
        sector_related=False
    ),
    "volatility": FactorFamily(
        id="volatility",
        name="volatility",
        display_name="Volatility",
        category="risk",
        description="Volatility and risk factors - low volatility anomaly",
        keywords=["volatility", "variance", "risk", "dispersion", "std_dev"],
        datafield_indicators=["returns", "close"],
        operator_indicators=["ts_std_dev", "ts_rank", "rank"],
        macro_related=True,
        sector_related=False
    ),
    "growth": FactorFamily(
        id="growth",
        name="growth",
        display_name="Growth",
        category="fundamental",
        description="Revenue and earnings growth expectations",
        keywords=["growth", "growth_rate", "change", "delta", "revision"],
        datafield_indicators=["revenue", "ebit", "net_income"],
        operator_indicators=["ts_delta", "rank", "ts_rank"],
        macro_related=True,
        sector_related=False
    ),
    "defensive": FactorFamily(
        id="defensive",
        name="defensive",
        display_name="Defensive",
        category="style",
        description="Low beta, stable earnings sectors - utilities, consumer staples",
        keywords=["defensive", "low_beta", "stable", "utility", "staples"],
        datafield_indicators=["beta_last_30_days_spy", "fscore_momentum"],
        operator_indicators=["rank"],
        macro_related=True,
        sector_related=True
    ),
    "carry": FactorFamily(
        id="carry",
        name="carry",
        display_name="Carry",
        category="arbitrage",
        description="Yield differential and carry trades",
        keywords=["carry", "yield", "basis", "futures", "roll_yield"],
        datafield_indicators=["close", "vwap"],
        operator_indicators=["rank", "ts_mean"],
        macro_related=True,
        sector_related=False
    ),
    "recovery": FactorFamily(
        id="recovery",
        name="recovery",
        display_name="Recovery",
        category="event",
        description="Post-distress recovery and turnarounds",
        keywords=["recovery", "turnaround", "distress", "bounce", "rebound"],
        datafield_indicators=["ebit", "revenue", "returns"],
        operator_indicators=["rank", "ts_delta", "ts_rank"],
        macro_related=True,
        sector_related=False
    ),
    "distress": FactorFamily(
        id="distress",
        name="distress",
        display_name="Distress",
        category="event",
        description="Financial distress signals and bankruptcy prediction",
        keywords=["distress", "bankruptcy", "default", "credit", "leverage"],
        datafield_indicators=["cap", "ebit", "debt"],
        operator_indicators=["rank", "group_rank"],
        macro_related=True,
        sector_related=False
    ),
    "positioning": FactorFamily(
        id="positioning",
        name="positioning",
        display_name="Positioning",
        category="behavioral",
        description="Institutional positioning and flow-based signals",
        keywords=["positioning", "institutional", "flow", "smart_money", "crowding"],
        datafield_indicators=["volume", "returns", "scl12_buzz"],
        operator_indicators=["ts_rank", "ts_mean", "rank"],
        macro_related=False,
        sector_related=False
    ),
    "flow_based": FactorFamily(
        id="flow_based",
        name="flow_based",
        display_name="Flow-Based",
        category="behavioral",
        description="Money flow and order flow signals",
        keywords=["flow", "order_flow", "buying_pressure", "selling_pressure", "volume"],
        datafield_indicators=["volume", "close", "vwap", "returns"],
        operator_indicators=["ts_rank", "ts_mean", "group_rank"],
        macro_related=False,
        sector_related=False
    ),
    "stat_arb": FactorFamily(
        id="stat_arb",
        name="stat_arb",
        display_name="Statistical Arbitrage",
        category="quantitative",
        description="Pairs trading, cointegration, statistical relationships",
        keywords=["stat_arb", "pairs", "cointegration", "correlation", "spread"],
        datafield_indicators=["close", "returns"],
        operator_indicators=["ts_corr", "ts_zscore", "rank"],
        macro_related=False,
        sector_related=False
    ),
    "macro_sensitive": FactorFamily(
        id="macro_sensitive",
        name="macro_sensitive",
        display_name="Macro-Sensitive",
        category="macro",
        description="Factors sensitive to economic conditions",
        keywords=["macro", "inflation", "rates", "gdp", "treasury", "yield_curve"],
        datafield_indicators=["beta_last_30_days_spy", "returns"],
        operator_indicators=["ts_corr", "ts_mean"],
        macro_related=True,
        sector_related=True
    ),
    "sector_sensitive": FactorFamily(
        id="sector_sensitive",
        name="sector_sensitive",
        display_name="Sector-Sensitive",
        category="sector",
        description="Sector rotation and industry-specific factors",
        keywords=["sector", "industry", "rotation", "subindustry", "peer"],
        datafield_indicators=["close", "returns", "volume"],
        operator_indicators=["group_rank", "group_mean", "group_neutralize"],
        macro_related=False,
        sector_related=True
    ),
    # ── Portfolio-context families (WQB IQC) ──────────────────────────────────
    "attention": FactorFamily(
        id="attention",
        name="attention",
        display_name="Attention / Volume Anomaly",
        category="behavioral",
        description="Stocks receiving abnormal participation relative to recent history. RVOL, news buzz, sentiment volume.",
        keywords=["attention", "rvol", "volume_anomaly", "buzz", "participation", "relative_volume"],
        datafield_indicators=["volume", "scl12_buzz", "snt_buzz", "mean_composite_sentiment_score"],
        operator_indicators=["ts_sum", "ts_mean"],
        macro_related=False,
        sector_related=False
    ),
    "neglect": FactorFamily(
        id="neglect",
        name="neglect",
        display_name="Neglect / Contrarian",
        category="behavioral",
        description="Recent losers not yet repriced. Negative multi-week return signals, contrarian timing.",
        keywords=["neglect", "contrarian", "loser", "underperform", "mean_reversion"],
        datafield_indicators=["returns"],
        operator_indicators=["ts_sum", "rank"],
        macro_related=False,
        sector_related=False
    ),
    "price_state": FactorFamily(
        id="price_state",
        name="price_state",
        display_name="Price State / Intraday",
        category="price_action",
        description="Price level relative to moving averages, intraday open-close patterns, Parkinson volatility.",
        keywords=["price_state", "intraday", "close_weakness", "parkinson", "mean_reversion", "open_close"],
        datafield_indicators=["close", "open", "parkinson_volatility_60", "parkinson_volatility_180", "vwap"],
        operator_indicators=["ts_mean", "rank"],
        macro_related=False,
        sector_related=False
    ),
    "operational": FactorFamily(
        id="operational",
        name="operational",
        display_name="Operational Efficiency",
        category="fundamental",
        description="Sales/inventory ratios, asset turnover, revenue quality. Measures how efficiently capital is deployed.",
        keywords=["operational", "inventory", "sales", "asset_turnover", "throughput", "efficiency"],
        datafield_indicators=["inventory", "sales", "revenue", "asset_turnover", "cap_turnover"],
        operator_indicators=["rank", "group_rank", "ts_delta"],
        macro_related=False,
        sector_related=False
    ),
    "capital_efficiency": FactorFamily(
        id="capital_efficiency",
        name="capital_efficiency",
        display_name="Capital Efficiency",
        category="fundamental",
        description="Return on assets, gross profitability, cash flow quality. Measures profitability per unit of capital.",
        keywords=["capital_efficiency", "profitability", "roa", "roe", "gross_profit", "ebitda", "cash_flow"],
        datafield_indicators=["gross_profit", "ebitda", "net_income", "assets", "cash_flow"],
        operator_indicators=["rank", "group_rank"],
        macro_related=False,
        sector_related=False
    ),
    "balance_sheet": FactorFamily(
        id="balance_sheet",
        name="balance_sheet",
        display_name="Balance Sheet / Stress",
        category="fundamental",
        description="Leverage, liquidity, interest coverage, working capital. Financial health and solvency signals.",
        keywords=["balance_sheet", "leverage", "debt", "liquidity", "coverage", "working_capital", "solvency"],
        datafield_indicators=["debt", "total_debt", "current_ratio", "interest_expense", "working_capital"],
        operator_indicators=["rank", "group_rank"],
        macro_related=True,
        sector_related=False
    ),
    "implied_vol_options": FactorFamily(
        id="implied_vol_options",
        name="implied_vol_options",
        display_name="Implied Vol / Options",
        category="derivatives",
        description="Options-based signals: IV term structure, put-call ratio, vol skew.",
        keywords=["implied_volatility", "pcr", "put_call", "options", "skew", "term_structure", "vol_surface"],
        datafield_indicators=["implied_volatility_call_270", "pcr_oi_call", "pcr_oi_put"],
        operator_indicators=["ts_rank", "rank", "ts_mean"],
        macro_related=True,
        sector_related=False
    ),
}

MACRO_THEMES = {
    "crisis": MacroTheme(
        id="crisis",
        name="Crisis/Recession",
        description="Market stress and recessionary periods",
        related_factors=["quality", "defensive", "liquidity", "low_vol"],
        regime_indicators=["VIX>30", "credit_spreads_wide", "yield_curve_inverted"]
    ),
    "recovery": MacroTheme(
        id="recovery",
        name="Recovery",
        description="Post-crisis economic recovery phase",
        related_factors=["recovery", "momentum", "value", "growth"],
        regime_indicators=["VIX<20", "credit_spreads_narrow", "yield_curve_steep"]
    ),
    "inflation": MacroTheme(
        id="inflation",
        name="Inflation Regime",
        description="High inflation environment",
        related_factors=["value", "commodity_linked", "real_assets"],
        regime_indicators=["CPI>4%", "yields_rising"]
    ),
    "deflation": MacroTheme(
        id="deflation",
        name="Deflation",
        description="Falling prices and deflationary pressure",
        related_factors=["defensive", "cash_rich", "low_leverage"],
        regime_indicators=["CPI<1%", "yields_falling"]
    ),
    "growth_leadership": MacroTheme(
        id="growth_leadership",
        name="Growth Leadership",
        description="Growth stocks outperform value",
        related_factors=["growth", "momentum", "momentum"],
        regime_indicators=["rates_low", "tech_outperformance"]
    ),
    "value_rotation": MacroTheme(
        id="value_rotation",
        name="Value Rotation",
        description="Value stocks outperform growth",
        related_factors=["value", "quality", "dividend"],
        regime_indicators=["rates_rising", "value_outperformance"]
    ),
    "volatility_spike": MacroTheme(
        id="volatility_spike",
        name="Volatility Spike",
        description="Sudden increase in market volatility",
        related_factors=["volatility", "defensive", "liquidity"],
        regime_indicators=["VIX>25", "correlation_spike"]
    ),
    "liquidity_stress": MacroTheme(
        id="liquidity_stress",
        name="Liquidity Stress",
        description="Reduced market liquidity",
        related_factors=["liquidity", "quality", "defensive"],
        regime_indicators=["bid_ask_wide", "volume_drop", "fund_flows_out"]
    ),
    "ai_speculative": MacroTheme(
        id="ai_speculative",
        name="AI/Speculative Regime",
        description="Speculative bubble in AI/tech stocks",
        related_factors=["momentum", "growth", "speculative"],
        regime_indicators=["tech_rally", "IPO_activity", "valuation_expansion"]
    ),
    "risk_on": MacroTheme(
        id="risk_on",
        name="Risk-On",
        description="Risk appetite high, search for yield",
        related_factors=["momentum", "growth", "high_beta"],
        regime_indicators=[" equities_rally", "credit_spreads_tight", "VIX_low"]
    ),
    "risk_off": MacroTheme(
        id="risk_off",
        name="Risk-Off",
        description="Risk aversion, flight to safety",
        related_factors=["defensive", "low_vol", "quality"],
        regime_indicators=["bonds_rally", "VIX_high", "gold_rally"]
    ),
}


class FactorOntologyEngine:
    """Automatic factor classification and exposure detection."""

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path(__file__).parent / "factor_ontology.json"

        self.registry_path = registry_path
        self.families = FACTOR_FAMILIES
        self.macro_themes = MACRO_THEMES
        self._load_custom()

    def _load_custom(self):
        """Load custom factor classifications if they exist."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    if "custom_families" in data:
                        for k, v in data["custom_families"].items():
                            self.families[k] = FactorFamily(**v)
            except Exception:
                pass

    def classify_alpha(
        self,
        expression: str,
        datafields: List[str],
        operators: List[str],
        concepts: List[str]
    ) -> List[AlphaExposure]:
        """
        Classify an alpha into factor families based on expression, datafields, operators.

        Returns list of detected exposures with confidence scores.
        """
        exposures = []
        expression_lower = expression.lower()

        for family_id, family in self.families.items():
            score = 0.0
            matched_indicators = []

            for kw in family.keywords:
                if kw in expression_lower:
                    score += 0.15
                    matched_indicators.append(f"keyword:{kw}")

            for df in family.datafield_indicators:
                if df in datafields:
                    score += 0.2
                    matched_indicators.append(f"datafield:{df}")

            for op in family.operator_indicators:
                if op in operators:
                    score += 0.1
                    matched_indicators.append(f"operator:{op}")

            for concept in concepts:
                if concept.lower() in [kw.lower() for kw in family.keywords]:
                    score += 0.15
                    matched_indicators.append(f"concept:{concept}")

            if score >= 0.2:
                direction = self._infer_direction(expression_lower, family_id)
                exposures.append(AlphaExposure(
                    alpha_id="",
                    factor_family=family_id,
                    confidence=min(1.0, score),
                    indicator="; ".join(matched_indicators[:3]),
                    direction=direction
                ))

        exposures.sort(key=lambda x: x.confidence, reverse=True)
        return exposures

    def classify_from_file(self, path: Path) -> List[str]:
        """Read an alpha .md file and return its top factor family ids."""
        meta = {}
        try:
            import frontmatter as fm
            post = fm.load(str(path))
            meta = post.metadata
        except Exception:
            pass

        if not meta:
            # Manual YAML list parser for the frontmatter block
            try:
                text = path.read_text(encoding="utf-8")
                import re as _re
                # Extract frontmatter between --- markers
                fm_match = _re.match(r'^---\n(.*?)\n---', text, _re.DOTALL)
                if fm_match:
                    fm_text = fm_match.group(1)
                    current_key = None
                    for line in fm_text.split("\n"):
                        list_item = _re.match(r'^- (.+)', line)
                        kv = _re.match(r'^(\w+):\s*(.*)', line)
                        if list_item and current_key:
                            if current_key not in meta:
                                meta[current_key] = []
                            if isinstance(meta[current_key], list):
                                meta[current_key].append(list_item.group(1).strip())
                        elif kv:
                            current_key = kv.group(1)
                            val = kv.group(2).strip()
                            meta[current_key] = val if val else []
            except Exception:
                return []

        def _as_list(v):
            if isinstance(v, list): return [str(x) for x in v]
            if v: return [str(v)]
            return []

        expr = str(meta.get("expression") or "")
        datafields = _as_list(meta.get("datafields"))
        operators = _as_list(meta.get("operators"))
        concepts = _as_list(meta.get("concepts"))

        # Portfolio-context explicit pattern matching (precision over recall)
        families = set()
        expr_l = expr.lower()
        df_str = " ".join(str(f).lower() for f in datafields)
        expr_plus_df = expr_l + " " + df_str

        # Attention: RVOL — volume divided by its own rolling average
        if re.search(r'volume\s*/\s*\(?ts_su?m?\s*\(?\s*volume', expr_l) or \
           re.search(r'volume\s*/\s*\(?\s*ts_mean\s*\(\s*volume', expr_l) or \
           any(f in datafields for f in ["scl12_buzz", "snt_buzz",
                                          "mean_composite_sentiment_score",
                                          "scl12_buzz_fast_d1", "snt_buzz_fast_d1"]):
            families.add("attention")

        # Neglect: explicit negated multi-week return accumulation
        if re.search(r'-\s*ts_sum\s*\(\s*returns', expr_l) or \
           re.search(r'rank\s*\(-\s*ts_sum\s*\(\s*returns', expr_l) or \
           re.search(r'-\s*ts_rank\s*\(\s*returns', expr_l) or \
           re.search(r'returns\s*>\s*0\?.*250', expr_l):
            families.add("neglect")

        # Price state: intraday OC midpoint, Parkinson vol, close vs own MA
        if re.search(r'\(\s*open\s*\+\s*close\s*\)', expr_l) or \
           re.search(r'1\s*-\s*close\s*/\s*open', expr_l) or \
           re.search(r'close\s*/\s*open', expr_l) or \
           re.search(r'close\s*-\s*vwap', expr_l) or \
           re.search(r'parkinson', expr_plus_df):
            families.add("price_state")

        # Operational: sales/inventory/revenue datafields
        if re.search(r'(?:^|\b)(inventor|sales_|revenue|asset_turn|cap_turn)', df_str):
            families.add("operational")

        # Capital efficiency: profitability datafields
        if re.search(r'(?:gross_profit|ebitd|net_income|return_on|roa_|roe_)', df_str):
            families.add("capital_efficiency")

        # Balance sheet: leverage/debt/liquidity datafields
        if re.search(r'(?:total_debt|leverage|interest_exp|working_cap|current_ratio)', df_str):
            families.add("balance_sheet")

        # Implied vol / options datafields
        if re.search(r'(?:implied_volatility|pcr_oi|pcr_vol|put_call)', df_str):
            families.add("implied_vol_options")

        # Model composite: mdl77 and named composite model fields
        if re.search(r'mdl77_|equity_value_score|mdl177_', expr_l) and not families:
            # Use the field name to infer the sub-family where possible
            if re.search(r'growth|gpam|eps|revision|earn', expr_l + df_str):
                families.add("growth")
            elif re.search(r'liquidity|liqrisk', expr_l + df_str):
                families.add("liquidity")
            elif re.search(r'value|pegy|bp_|pe_|ev_', expr_l + df_str):
                families.add("value")
            elif re.search(r'profit|margin|roa|roe|quality', expr_l + df_str):
                families.add("capital_efficiency")
            else:
                families.add("model_composite")

        # For remaining unclassified alphas, fall back to ontology engine
        # with high confidence threshold to avoid generic noise
        if not families:
            exposures = self.classify_alpha(expr, datafields, operators, concepts)
            legacy_families = {"quality", "value", "momentum", "reversal",
                               "liquidity", "volatility", "growth", "distress",
                               "positioning", "flow_based", "sector_sensitive"}
            families = {e.factor_family for e in exposures[:2]
                        if e.confidence >= 0.45 and e.factor_family in legacy_families}

        return sorted(families) if families else ["unclassified"]

    def portfolio_saturation(
        self,
        alphas_dir: Path,
        active_states: set = None
    ) -> Dict:
        """
        Compute factor family saturation across the active portfolio.

        Returns dict: {family_id: {"count": N, "alphas": [id, ...]}}
        """
        if active_states is None:
            active_states = {"IS_PASS", "ACTIVE_OS", "SUBMITTED"}

        saturation: Dict[str, List[str]] = defaultdict(list)
        total = 0

        for path in sorted(alphas_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                state_m = re.search(r'^pipeline_state:\s*(\S+)', text, re.MULTILINE)
                state = state_m.group(1) if state_m else ""
                if state not in active_states:
                    continue
                alpha_id = path.stem
                families = self.classify_from_file(path)
                if not families:
                    families = ["unclassified"]
                for f in families:
                    saturation[f].append(alpha_id)
                total += 1
            except Exception:
                continue

        return {"total_active": total, "by_family": dict(saturation)}

    def _infer_direction(self, expression: str, factor_family: str) -> str:
        """Infer whether the alpha is long/short/neutral on the factor."""
        has_negative = "-" in expression or "*-" in expression or "-1*" in expression

        negative_factors = {"momentum", "reversal", "value", "growth"}
        positive_factors = {"quality", "liquidity", "defensive", "carry"}

        if has_negative:
            if factor_family in positive_factors:
                return "short"
            elif factor_family in negative_factors:
                return "long"
        else:
            if factor_family in positive_factors:
                return "long"
            elif factor_family in negative_factors:
                return "short"

        return "neutral"

    def get_related_factors(self, factor_id: str) -> List[str]:
        """Get factors that typically complement or conflict with given factor."""
        complement_map = {
            "momentum": ["liquidity", "positioning"],
            "reversal": ["volatility", "stat_arb"],
            "value": ["quality", "growth"],
            "quality": ["defensive", "value"],
            "momentum": ["momentum"],
            "reversal": ["reversal"],
            "growth": ["momentum", "sentiment"],
        }
        conflict_map = {
            "momentum": ["reversal"],
            "reversal": ["momentum"],
            "value": ["growth"],
        }

        return {
            "complements": complement_map.get(factor_id, []),
            "conflicts": conflict_map.get(factor_id, [])
        }

    def get_factor_cluster(self, factors: List[str]) -> Dict:
        """Get cluster information for a set of factors."""
        macro_sensitive = []
        sector_sensitive = []
        pure_quant = []

        for f in factors:
            if f in self.families:
                family = self.families[f]
                if family.macro_related:
                    macro_sensitive.append(f)
                elif family.sector_related:
                    sector_sensitive.append(f)
                else:
                    pure_quant.append(f)

        return {
            "macro_sensitive": macro_sensitive,
            "sector_sensitive": sector_sensitive,
            "pure_quant": pure_quant,
            "cluster_type": self._determine_cluster_type(factors)
        }

    def _determine_cluster_type(self, factors: List[str]) -> str:
        """Determine the overall cluster type."""
        has_macro = any(f in MACRO_THEMES.get(f, MacroTheme("", "", "", [], [])).related_factors
                        for f in factors if f in MACRO_THEMES)
        has_sector = any(self.families.get(f, FactorFamily("", "", "", "", "", [], [], [])).sector_related
                         for f in factors if f in self.families)

        if has_macro and has_sector:
            return "macro_sector"
        elif has_macro:
            return "macro_oriented"
        elif has_sector:
            return "sector_oriented"
        else:
            return "pure_quant"

    def get_regime_factors(self, regime: str) -> List[str]:
        """Get factors that typically perform well in a given regime."""
        theme = self.macro_themes.get(regime)
        if theme:
            return theme.related_factors
        return []

    def save_classifications(self, alpha_id: str, exposures: List[AlphaExposure]) -> None:
        """Save factor classifications to registry."""
        data = {}
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass

        if "classifications" not in data:
            data["classifications"] = {}

        data["classifications"][alpha_id] = [asdict(e) for e in exposures]

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)


ontology_engine = FactorOntologyEngine()


def classify_alpha(
    expression: str,
    datafields: List[str],
    operators: List[str],
    concepts: List[str]
) -> List[Dict]:
    """Convenience function for alpha classification."""
    exposures = ontology_engine.classify_alpha(
        expression, datafields, operators, concepts
    )
    return [asdict(e) for e in exposures]


def get_factor_info(factor_id: str) -> Optional[Dict]:
    """Get detailed factor information."""
    if factor_id in FACTOR_FAMILIES:
        return asdict(FACTOR_FAMILIES[factor_id])
    return None


def get_related_factors(factor_id: str) -> Dict:
    """Get related factors (complements and conflicts)."""
    return ontology_engine.get_related_factors(factor_id)


def get_regime_factors(regime: str) -> List[str]:
    """Get factors for a specific regime."""
    return ontology_engine.get_regime_factors(regime)
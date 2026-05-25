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
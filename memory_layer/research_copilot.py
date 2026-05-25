"""
AI Research Copilot
Institutional-style research assistant that reasons like a senior quant PM.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict


@dataclass
class CopilotResponse:
    """Structured response from the copilot."""
    response_type: str
    content: str
    reasoning: List[str]
    suggestions: List[str]
    related_alphas: List[str]
    confidence: float


class ResearchCopilot:
    """
    Institutional-grade research copilot.
    Reasons like a senior quant PM with macro awareness.
    """

    def __init__(self):
        self._load_context()

    def _load_context(self):
        """Load necessary context."""
        self.ontology_engine = None
        self.regime_analyzer = None
        self.failure_learning = None
        self.correlation_engine = None

        try:
            from memory_layer import factor_ontology
            self.ontology_engine = factor_ontology.ontology_engine
        except:
            pass

        try:
            from memory_layer import regime_analysis
            self.regime_analyzer = regime_analysis.analyzer
        except:
            pass

        try:
            from memory_layer import failure_learning
            self.failure_learning = failure_learning.learning_engine
        except:
            pass

        try:
            from memory_layer import correlation_engine
            self.correlation_engine = correlation_engine.correlation_engine
        except:
            pass

    def explain_failure(
        self,
        alpha_id: str,
        failure_modes: List[str],
        sharpe: Optional[float],
        turnover: Optional[float],
        fitness: Optional[float],
        expression: str,
        operators: List[str],
        datafields: List[str]
    ) -> CopilotResponse:
        """Explain why an alpha failed."""
        reasoning = []
        suggestions = []
        related = []

        if "high_turnover" in failure_modes:
            reasoning.append("High turnover typically comes from noisy signals or insufficient smoothing")
            if "rank" in operators or "ts_delay" in operators:
                reasoning.append("Rank without decay generates new signals each day")
            if "ts_mean" not in operators and "ts_decay_linear" not in operators:
                suggestions.append("Add ts_decay_linear(..., 5-10) to smooth the signal")
                suggestions.append("Alternatively, increase simulation Decay setting to 10-20")
            related.append("Look at alpha_0803 which had Sharpe 2.19 but 116% turnover - same CLV signal")

        if "low_sharpe" in failure_modes:
            reasoning.append("Sharpe below 1.25 indicates signal quality issues")
            if "group_neutralize" not in operators and "neutralization" not in expression:
                suggestions.append("Add neutralization - market neutralization is baseline")
            elif "market" in expression and any(df in datafields for df in ["ebit", "revenue"]):
                reasoning.append("Market neutralization on fundamentals compares different sectors")
                suggestions.append("Switch to industry neutralization to compare within sectors")
                related.append("alpha_0901 with industry neutralization got Sharpe 1.31 vs 0.74 with market")

        if "correlated" in failure_modes:
            reasoning.append("Self-correlation above 0.7 means positions barely change day-to-day")
            if any(df in datafields for df in ["ebit", "capex", "revenue", "assets"]):
                reasoning.append("Quarterly fundamentals barely change - smoothing them adds noise")
                suggestions.append("Don't apply ts_mean/ts_decay on quarterly fundamentals")
                suggestions.append("Try using simulation Decay setting instead of formula decay")

        if "sector_bias" in failure_modes:
            reasoning.append("Alpha passes overall tests but fails within sectors")
            suggestions.append("Add industry or subindustry neutralization")
            suggestions.append("Use group_neutralize(..., industry) or group_neutralize(..., subindustry)")

        if sharpe and sharpe >= 1.5 and fitness and fitness < 1.0:
            reasoning.append(f"Ceiling alpha: Sharpe {sharpe} is excellent but blocked by other factors")
            if turnover and turnover > 70:
                suggestions.append("Focus on reducing turnover while preserving Sharpe")

        return CopilotResponse(
            response_type="failure_explanation",
            content=self._format_failure_explanation(reasoning),
            reasoning=reasoning,
            suggestions=suggestions,
            related_alphas=related[:3],
            confidence=0.85
        )

    def _format_failure_explanation(self, reasoning: List[str]) -> str:
        """Format reasoning into coherent explanation."""
        return " | ".join(reasoning[:3])

    def detect_hidden_exposures(
        self,
        expression: str,
        datafields: List[str],
        operators: List[str],
        concepts: List[str],
        neutralization: str
    ) -> CopilotResponse:
        """Detect hidden factor exposures."""
        exposures = []
        reasoning = []
        suggestions = []

        if "market" in neutralization:
            if any(df in datafields for df in ["ebit", "revenue", "assets", "capex"]):
                exposures.append("Sector exposure - fundamentals vary by sector")
                reasoning.append("Market neutral compares tech to utilities which is unfair")
                suggestions.append("Use industry or subindustry neutralization")

        if not neutralization or neutralization == "None":
            exposures.append("Market beta exposure - no neutralization applied")
            reasoning.append("Without neutralization, alpha may be just market beta")
            suggestions.append("Add market neutralization")

        if "close" in datafields and "volume" in datafields:
            exposures.append("CLV signal may have price impact on small caps")
            reasoning.append("CLV signals can be weaker on low-liquidity stocks")

        if "returns" in datafields:
            exposures.append("Momentum exposure - past returns predict future returns")
            reasoning.append("Pure momentum can be unstable in regime changes")

        if "rank" in operators and neutralization == "market":
            exposures.append("Fundamental rank sector bias")
            reasoning.append("Ranking fundamentals without sector adjustment compares across sectors")

        if not exposures:
            return CopilotResponse(
                response_type="hidden_exposure",
                content="No significant hidden exposures detected",
                reasoning=["Alpha appears well-structured"],
                suggestions=[],
                related_alphas=[],
                confidence=0.9
            )

        return CopilotResponse(
            response_type="hidden_exposure",
            content=f"Detected: {', '.join(exposures)}",
            reasoning=reasoning,
            suggestions=suggestions,
            related_alphas=[],
            confidence=0.8
        )

    def detect_factor_conflicts(
        self,
        concepts: List[str],
        operators: List[str],
        expression: str
    ) -> CopilotResponse:
        """Detect conflicting factor exposures."""
        conflicts = []
        reasoning = []
        suggestions = []

        if "momentum" in concepts and "mean_reversion" in concepts:
            conflicts.append("Momentum + Mean Reversion - opposite directional signals")
            reasoning.append("ts_rank on returns is momentum, ts_mean is mean reversion")
            suggestions.append("Pick one: pure momentum OR pure reversal, not both")
            suggestions.append("If using ts_rank on returns, don't also apply ts_mean smoothing")

        if "value" in concepts and "growth" in concepts:
            conflicts.append("Value + Growth - typically inverse factor tilts")
            reasoning.append("Value factors buy cheap, growth factors buy expensive")

        neg_count = expression.count("-")
        if neg_count > 2:
            conflicts.append("Multiple negative signs - complex directional mixing")
            reasoning.append("Multiple inversions can create unintended factor exposures")

        if not conflicts:
            return CopilotResponse(
                response_type="factor_conflict",
                content="No factor conflicts detected",
                reasoning=["Factor combination appears coherent"],
                suggestions=[],
                related_alphas=[],
                confidence=0.9
            )

        return CopilotResponse(
            response_type="factor_conflict",
            content=f"Conflicts: {', '.join(conflicts)}",
            reasoning=reasoning,
            suggestions=suggestions,
            related_alphas=[],
            confidence=0.85
        )

    def suggest_orthogonal_sleeves(
        self,
        alpha_id: str,
        factor_families: List[str],
        available_alphas: List[str] = None
    ) -> CopilotResponse:
        """Suggest orthogonal alpha sleeves."""
        from memory_layer.correlation_engine import correlation_engine

        suggestions = []
        reasoning = []
        related = []

        complementary = {
            "momentum": ["reversal", "quality", "liquidity"],
            "reversal": ["volatility", "stat_arb"],
            "value": ["quality", "growth"],
            "quality": ["defensive", "momentum"],
            "liquidity": ["momentum", "volatility"],
            "growth": ["momentum", "sentiment"]
        }

        for ff in factor_families:
            if ff in complementary:
                for suggested in complementary[ff]:
                    suggestions.append(f"Add {suggested} factor sleeve for diversification")

        orthogonal = correlation_engine.find_orthogonal_alphas(alpha_id, max_correlation=0.2)

        if orthogonal:
            suggestions.append(f"Consider combining with {orthogonal[0]['alpha_id']} (corr: {orthogonal[0]['correlation']:.2f})")
            related.append(orthogonal[0]['alpha_id'])
            if len(orthogonal) > 1:
                suggestions.append(f"And {orthogonal[1]['alpha_id']} (corr: {orthogonal[1]['correlation']:.2f})")
                related.append(orthogonal[1]['alpha_id'])

        reasoning.append("Portfolio benefits from orthogonal alpha combinations")
        reasoning.append(f"Current alpha has factors: {', '.join(factor_families)}")

        return CopilotResponse(
            response_type="orthogonal_sleeves",
            content=f"Suggested: {', '.join(suggestions[:2])}",
            reasoning=reasoning,
            suggestions=suggestions[:3],
            related_alphas=related,
            confidence=0.75
        )

    def explain_macro_sensitivity(
        self,
        factor_families: List[str],
        neutralization: str,
        universe: str
    ) -> CopilotResponse:
        """Explain macro sensitivity of alpha."""
        reasoning = []
        suggestions = []
        exposures = []

        macro_related = {"value", "quality", "growth", "defensive", "liquidity", "volatility",
                         "carry", "recovery", "distress"}

        macro_factors = [f for f in factor_families if f in macro_related]

        if macro_factors:
            exposures.append(f"Macro-sensitive: {', '.join(macro_factors)}")
            reasoning.append(f"These factors correlate with economic conditions")

            if "value" in macro_factors:
                reasoning.append("Value performs differently in inflation vs growth regimes")
                suggestions.append("Test value alphas in different rate environments")

            if "defensive" in macro_factors:
                reasoning.append("Defensive factors perform in risk-off periods")

        if neutralization == "None" and universe in ["TOP3000", "TOP5000"]:
            exposures.append("High market beta")
            reasoning.append("No neutralization means full market exposure")

        if not exposures:
            return CopilotResponse(
                response_type="macro_sensitivity",
                content="Alpha appears macro-agnostic",
                reasoning=["Pure quant factors without macro dependencies"],
                suggestions=[],
                related_alphas=[],
                confidence=0.8
            )

        return CopilotResponse(
            response_type="macro_sensitivity",
            content=f"Macro exposures: {', '.join(exposures)}",
            reasoning=reasoning,
            suggestions=suggestions,
            related_alphas=[],
            confidence=0.8
        )

    def infer_economic_meaning(
        self,
        expression: str,
        datafields: List[str],
        operators: List[str],
        concepts: List[str]
    ) -> CopilotResponse:
        """Infer economic meaning from alpha expression."""
        meanings = []
        reasoning = []

        if "close" in datafields and "volume" in datafields:
            if "ts_mean" in operators or "rank" in operators:
                meanings.append("CLV (Close Location Value) - measures where price closed in daily range")
                reasoning.append("High CLV = close near high (buying pressure), low = close near low (selling)")

        if "ebit" in datafields and "capex" in datafields:
            meanings.append("EBIT/CapEx - measures capital efficiency / quality of investment")
            reasoning.append("High EBIT/CapEx = company generates good returns on capital invested")

        if "analyst_revision" in str(datafields):
            meanings.append("Analyst revision momentum - tracks earnings estimate changes")
            reasoning.append("Positive revisions indicate improving fundamentals")

        if "returns" in datafields:
            if "ts_rank" in operators:
                meanings.append("Time-series rank momentum - relative strength over lookback")
                reasoning.append("High ts_rank = strong recent performance relative to history")

        if "volume" in datafields:
            if "rank" in operators:
                meanings.append("Volume rank - relative trading activity")
                reasoning.append("High volume = high interest/liquidity")

        if not meanings:
            meanings.append("Quant factor - specific economic interpretation depends on datafields")

        return CopilotResponse(
            response_type="economic_interpretation",
            content=" | ".join(meanings),
            reasoning=reasoning,
            suggestions=[],
            related_alphas=[],
            confidence=0.7
        )

    def detect_overfitting(
        self,
        expression: str,
        sharpe: float,
        turnover: float,
        universe: str,
        parent_alpha: Optional[str]
    ) -> CopilotResponse:
        """Detect potential overfitting indicators."""
        indicators = []
        reasoning = []

        if sharpe > 2.0:
            indicators.append("Very high Sharpe (>2.0) often indicates overfitting")
            reasoning.append("Such high Sharpe rarely survives out-of-sample")

        if "rank" in expression and "ts_mean" not in expression:
            if turnover and turnover > 80:
                indicators.append("Rank without smoothing + high turnover = likely noise fitting")
                reasoning.append("May be fitting to historical noise patterns")

        if parent_alpha:
            if "ts_delta" in expression and sharpe > parent_alpha + 0.3:
                indicators.append("Large improvement from small modification suggests cherry-picking")

        if universe == "TOP500":
            indicators.append("Small universe (TOP500) more prone to overfitting")
            reasoning.append("Fewer stocks = easier to fit noise")

        if not indicators:
            return CopilotResponse(
                response_type="overfitting_detection",
                content="No strong overfitting indicators detected",
                reasoning=["Alpha characteristics appear reasonable"],
                suggestions=[],
                related_alphas=[],
                confidence=0.8
            )

        return CopilotResponse(
            response_type="overfitting_detection",
            content=f"Potential overfitting: {', '.join(indicators)}",
            reasoning=reasoning,
            suggestions=["Test on longer history", "Test on different universe", "Check out-of-sample performance"],
            related_alphas=[],
            confidence=0.75
        )

    def suggest_portfolio_combination(
        self,
        alpha_ids: List[str],
        alpha_metrics: Dict[str, Dict]
    ) -> CopilotResponse:
        """Suggest how to combine multiple alphas."""
        from memory_layer.correlation_engine import correlation_engine, portfolio_engine

        reasoning = []
        suggestions = []

        if len(alpha_ids) < 2:
            return CopilotResponse(
                response_type="portfolio_combination",
                content="Need at least 2 alphas for portfolio combination",
                reasoning=["Single alpha cannot be diversified"],
                suggestions=["Add another alpha"],
                related_alphas=[],
                confidence=0.9
            )

        weights = {a: 1.0 / len(alpha_ids) for a in alpha_ids}
        metrics = portfolio_engine.compute_portfolio_metrics(weights, alpha_metrics)

        if metrics.get("diversification_score", 0) > 0.7:
            reasoning.append("Alphas have good diversification")
            suggestions.append("Equal weighting is appropriate")
        elif metrics.get("avg_correlation", 0) > 0.5:
            reasoning.append("Alphas are correlated - consider removing one")
            suggestions.append("Look for orthogonal alternatives")

        if metrics.get("expected_turnover", 0) > 60:
            suggestions.append("Combined portfolio may have high turnover - consider decay settings")

        return CopilotResponse(
            response_type="portfolio_combination",
            content=f"Expected portfolio Sharpe: {metrics.get('expected_sharpe', 0):.2f}",
            reasoning=reasoning,
            suggestions=suggestions,
            related_alphas=alpha_ids[:2],
            confidence=0.7
        )


copilot = ResearchCopilot()


def explain_alpha_failure(
    alpha_id: str,
    failure_modes: List[str],
    sharpe: Optional[float],
    turnover: Optional[float],
    fitness: Optional[float],
    expression: str,
    operators: List[str],
    datafields: List[str]
) -> Dict:
    """Explain alpha failure."""
    return asdict(copilot.explain_failure(
        alpha_id, failure_modes, sharpe, turnover, fitness, expression, operators, datafields
    ))


def detect_hidden_exposures(
    expression: str,
    datafields: List[str],
    operators: List[str],
    concepts: List[str],
    neutralization: str
) -> Dict:
    """Detect hidden factor exposures."""
    return asdict(copilot.detect_hidden_exposures(
        expression, datafields, operators, concepts, neutralization
    ))


def detect_factor_conflicts(
    concepts: List[str],
    operators: List[str],
    expression: str
) -> Dict:
    """Detect factor conflicts."""
    return asdict(copilot.detect_factor_conflicts(concepts, operators, expression))


def suggest_orthogonal(alpha_id: str, factor_families: List[str]) -> Dict:
    """Suggest orthogonal sleeves."""
    return asdict(copilot.suggest_orthogonal_sleeves(alpha_id, factor_families))


def explain_macro_sensitivity(factor_families: List[str], neutralization: str, universe: str) -> Dict:
    """Explain macro sensitivity."""
    return asdict(copilot.explain_macro_sensitivity(factor_families, neutralization, universe))


def infer_economic_meaning(
    expression: str,
    datafields: List[str],
    operators: List[str],
    concepts: List[str]
) -> Dict:
    """Infer economic meaning."""
    return asdict(copilot.infer_economic_meaning(expression, datafields, operators, concepts))


def detect_overfitting(
    expression: str,
    sharpe: float,
    turnover: float,
    universe: str,
    parent_alpha: Optional[str] = None
) -> Dict:
    """Detect overfitting."""
    return asdict(copilot.detect_overfitting(expression, sharpe, turnover, universe, parent_alpha))


def suggest_combination(alpha_ids: List[str], alpha_metrics: Dict[str, Dict]) -> Dict:
    """Suggest portfolio combination."""
    return asdict(copilot.suggest_portfolio_combination(alpha_ids, alpha_metrics))
"""
Failure-Mode Learning System
Track recurring research failures and detect structural patterns.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re


@dataclass
class FailureSignature:
    """A pattern of characteristics that leads to failure."""
    id: str
    name: str
    description: str
    indicators: List[str]
    warning_signs: List[str]
    suggested_fixes: List[str]
    severity: str  # critical, high, medium, low


@dataclass
class RecurringPattern:
    """A pattern of mistakes that repeat across alphas."""
    pattern_id: str
    pattern_type: str
    frequency: int
    affected_alphas: List[str]
    common_operators: List[str]
    common_datafields: List[str]
    common_concepts: List[str]
    recommendation: str


@dataclass
class AlphaFailureAnalysis:
    """Complete failure analysis for an alpha."""
    alpha_id: str
    primary_failure: str
    contributing_factors: List[str]
    operator_patterns: List[str]
    structural_issues: List[str]
    hidden_exposures: List[str]
    similar_failures: List[str]
    suggested_improvements: List[str]


FAILURE_SIGNATURES = {
    "high_turnover": FailureSignature(
        id="high_turnover",
        name="High Turnover",
        description="Alpha generates excessive trading due to noisy signals or insufficient smoothing",
        indicators=["turnover>70%", "daily_rebalance", "no_decay"],
        warning_signs=["ts_delay in formula", "rank on raw prices", "high_frequency_data"],
        suggested_fixes=[
            "Add ts_decay_linear(..., 10) to formula",
            "Increase simulation Decay setting to 10-20",
            "Use ts_mean on raw signal before ranking",
            "Switch to longer lookback operators"
        ],
        severity="high"
    ),
    "low_fitness": FailureSignature(
        id="low_fitness",
        name="Low Fitness",
        description="Alpha fails the combined Sharpe+Turnover fitness function",
        indicators=["fitness<1.0", "sharpe<1.25 OR turnover>70%"],
        warning_signs=["good_sharpe_bad_turnover", "ok_turnover_low_sharpe"],
        suggested_fixes=[
            "For Sharpe issue: add neutralization, improve signal quality",
            "For Turnover issue: add decay, use longer lookback",
            "Try universe change (TOP500, TOP1000)"
        ],
        severity="critical"
    ),
    "over_neutralization": FailureSignature(
        id="over_neutralization",
        name="Over-Neutralization",
        description="Excessive neutralization removes all signal",
        indicators=["fitness<0.5", "sharpe<0.5 after neutralization"],
        warning_signs=["multiple neutralization layers", "group_neutralize + market neutral"],
        suggested_fixes=[
            "Try industry instead of market neutralization",
            "Remove subindustry neutralization if using industry",
            "Consider sector-relative instead of absolute neutralization"
        ],
        severity="high"
    ),
    "momentum_reversal_conflict": FailureSignature(
        id="momentum_reversal_conflict",
        name="Momentum-Reversal Conflict",
        description="Alpha mixes momentum and reversal signals, causing cancellation",
        indicators=["momentum concept + reversal concept", "ts_rank + ts_mean on returns"],
        warning_signs=["mixed concepts", "dual directional signals"],
        suggested_fixes=[
            "Pick one direction: pure momentum OR pure reversal",
            "If using ts_rank on returns, don't also use ts_mean",
            "Check operator combinations for directional conflicts"
        ],
        severity="high"
    ),
    "excessive_smoothing": FailureSignature(
        id="excessive_smoothing",
        name="Excessive Smoothing",
        description="Too much smoothing destroys signal while trying to reduce noise",
        indicators=["decay>20", "ts_mean>20", "signal_flat"],
        warning_signs=["large decay values", "multiple smoothing layers", "sharpe_decreased_with_decay"],
        suggested_fixes=[
            "Reduce decay to 5-10 range",
            "Try ts_rank instead of ts_mean for smoothing",
            "Check if datafield has low frequency (quarterly) - don't smooth quarterly data"
        ],
        severity="medium"
    ),
    "hidden_beta": FailureSignature(
        id="hidden_beta",
        name="Hidden Beta Exposure",
        description="Alpha has unintended market/sector beta that causes regime failures",
        indicators=["market_correlation>0.7", "sector_correlation>0.5"],
        warning_signs=["no neutralization", "single_sector_datafield", "cap_correlation"],
        suggested_fixes=[
            "Add market neutralization",
            "Try industry or subindustry neutralization",
            "Check beta before/after neutralization"
        ],
        severity="critical"
    ),
    "low_uniqueness": FailureSignature(
        id="low_uniqueness",
        name="Low Uniqueness",
        description="Alpha correlates highly with existing alphas, adding no new signal",
        indicators=["correlation>0.7 with existing alphas"],
        warning_signs=["similar_datafields", "similar_operators", "derived_from_correlated_parent"],
        suggested_fixes=[
            "Use different datafields than correlated alphas",
            "Try different operator combinations",
            "Change universe or neutralization",
            "Focus on orthogonal factor exposures"
        ],
        severity="high"
    ),
    "concentration_risk": FailureSignature(
        id="concentration_risk",
        name="Concentration Risk",
        description="Alpha concentrates in few stocks/sectors, increasing idiosyncratic risk",
        indicators=["top_holdings>30%", "single_sector>50%"],
        warning_signs=["no_universe_filter", "rank without truncation", "narrow_universe"],
        suggested_fixes=[
            "Add truncation to simulation settings",
            "Use smaller universe (TOP500)",
            "Add sector cap in portfolio construction"
        ],
        severity="medium"
    ),
    "regime_collapse": FailureSignature(
        id="regime_collapse",
        name="Regime Collapse",
        description="Alpha performs well in training but fails in specific market regimes",
        indicators=["crisis_sharpe<0", "recovery_sharpe>1.5"],
        warning_signs=["momentum_only", "high_beta", "no_defensive_factor"],
        suggested_fixes=[
            "Add defensive factor (quality, low_vol)",
            "Test in multiple regimes explicitly",
            "Consider regime filtering in simulation"
        ],
        severity="critical"
    ),
    "sector_overexposure": FailureSignature(
        id="sector_overexposure",
        name="Sector Overexposure",
        description="Alpha cannot pass sub-universe tests due to sector concentration",
        indicators=["sub_universe_failure", "sector_bias"],
        warning_signs=["fundamental_ratios", "industry_neutralization_only", "no_subindustry"],
        suggested_fixes=[
            "Switch from market to industry neutralization",
            "Add subindustry neutralization",
            "Test with industry-relative returns"
        ],
        severity="high"
    ),
    "datafield_mismatch": FailureSignature(
        id="datafield_mismatch",
        name="Datafield Mismatch",
        description="Using incompatible datafield frequencies in same formula",
        indicators=["daily_data + quarterly_data", "mismatched_granularity"],
        warning_signs=["ebit + close", "revenue + volume", "quarterly_fundamentals_in_ts_ops"],
        suggested_fixes=[
            "Don't apply ts_mean on quarterly fundamentals",
            "Use ranks instead of raw values",
            "Apply neutralization after combining different frequencies"
        ],
        severity="high"
    ),
    "concept_drift": FailureSignature(
        id="concept_drift",
        name="Concept Drift",
        description="Alpha's concept becomes less predictive over time",
        indicators=["newer_years_worse_sharpe", "training_window_sensitive"],
        warning_signs=["single_time_period", "recent_concepts_only"],
        suggested_fixes=[
            "Test on longer history",
            "Add rolling window testing",
            "Check if concept is regime-dependent"
        ],
        severity="medium"
    ),
}


class FailureLearningEngine:
    """Detect and learn from recurring failure patterns."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "failure_learning.json"

        self.data_path = data_path
        self.signatures = FAILURE_SIGNATURES
        self._load_data()

    def _load_data(self):
        """Load existing failure learning data."""
        self.patterns: Dict[str, RecurringPattern] = {}
        self.alpha_analyses: Dict[str, AlphaFailureAnalysis] = {}
        self.known_alphas: Set[str] = set()

        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)

                    for p_id, p_data in data.get("patterns", {}).items():
                        self.patterns[p_id] = RecurringPattern(**p_data)

                    for a_id, a_data in data.get("analyses", {}).items():
                        self.alpha_analyses[a_id] = AlphaFailureAnalysis(**a_data)

                    self.known_alphas = set(data.get("known_alphas", []))
            except Exception:
                pass

    def _save_data(self):
        """Save failure learning data."""
        data = {
            "patterns": {k: asdict(v) for k, v in self.patterns.items()},
            "analyses": {k: asdict(v) for k, v in self.alpha_analyses.items()},
            "known_alphas": list(self.known_alphas)
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def analyze_alpha_failure(
        self,
        alpha_id: str,
        expression: str,
        datafields: List[str],
        operators: List[str],
        concepts: List[str],
        failure_modes: List[str],
        sharpe: Optional[float],
        turnover: Optional[float],
        fitness: Optional[float],
        correlated_with: List[str],
        neutralization: str
    ) -> AlphaFailureAnalysis:
        """Analyze why an alpha failed and identify patterns."""
        self.known_alphas.add(alpha_id)

        primary_failure = failure_modes[0] if failure_modes else "unknown"
        contributing_factors = []
        operator_patterns = []
        structural_issues = []
        hidden_exposures = []
        similar_failures = []

        if sharpe is not None and sharpe < 1.25:
            contributing_factors.append(f"Sharpe {sharpe:.2f} below 1.25 cutoff")

        if turnover is not None and turnover > 70:
            contributing_factors.append(f"Turnover {turnover:.1f}% exceeds 70% limit")
            if "ts_delay" in operators or "rank" in operators:
                operator_patterns.append("rank/ts_delay without decay")

        if fitness is not None and fitness < 1.0:
            contributing_factors.append(f"Fitness {fitness:.2f} below 1.0 cutoff")

        if "correlated" in failure_modes:
            structural_issues.append("Self-correlation > 0.7 - quarterly data being smoothed")
            if "ts_mean" in operators or "ts_decay_linear" in operators:
                structural_issues.append("Smoothing quarterly fundamentals destroys signal")

        if "sector_bias" in failure_modes:
            structural_issues.append("Market neutralization on fundamentals compares different sectors")
            hidden_exposures.append("Sector beta not neutralized - need industry/subindustry")

        if "sub_universe_failure" in failure_modes:
            structural_issues.append("Alpha fails sub-universe tests - likely sector concentration")

        if "momentum" in concepts and "mean_reversion" in concepts:
            structural_issues.append("Momentum + mean reversion concepts conflict")
            operator_patterns.append("Mixing directional signals")

        if len(set(operators).intersection({"ts_mean", "ts_sum", "ts_std_dev"})) > 2:
            structural_issues.append("Multiple time-series operators may over-smooth")

        if "rank" in operators and neutralization == "market":
            if any(df in datafields for df in ["ebit", "revenue", "assets"]):
                hidden_exposures.append("Fundamental ranks with market neutral may have sector bias")

        similar_failures = self._find_similar_failures(
            operators, datafields, concepts, failure_modes
        )

        suggested_improvements = self._generate_suggestions(
            failure_modes, operator_patterns, structural_issues, hidden_exposures
        )

        analysis = AlphaFailureAnalysis(
            alpha_id=alpha_id,
            primary_failure=primary_failure,
            contributing_factors=contributing_factors,
            operator_patterns=operator_patterns,
            structural_issues=structural_issues,
            hidden_exposures=hidden_exposures,
            similar_failures=similar_failures,
            suggested_improvements=suggested_improvements
        )

        self.alpha_analyses[alpha_id] = analysis
        self._update_patterns(analysis)
        self._save_data()

        return analysis

    def _find_similar_failures(
        self,
        operators: List[str],
        datafields: List[str],
        concepts: List[str],
        failure_modes: List[str]
    ) -> List[str]:
        """Find alphas with similar failure patterns."""
        similar = []

        for alpha_id, analysis in self.alpha_analyses.items():
            overlap = 0

            if set(analysis.operator_patterns).intersection(set(operators)):
                overlap += 1
            if set(analysis.structural_issues).intersection(set(failure_modes)):
                overlap += 1

            if overlap >= 1:
                similar.append(alpha_id)

        return similar[:5]

    def _generate_suggestions(
        self,
        failure_modes: List[str],
        operator_patterns: List[str],
        structural_issues: List[str],
        hidden_exposures: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        for fm in failure_modes:
            if fm in self.signatures:
                suggestions.extend(self.signatures[fm].suggested_fixes[:2])

        for op in operator_patterns:
            if "without decay" in op:
                suggestions.append("Add ts_decay_linear(..., 5-10) after rank operations")

        for hs in hidden_exposures:
            if "sector" in hs.lower():
                suggestions.append("Try industry or subindustry neutralization")

        return list(set(suggestions))[:4]

    def _update_patterns(self, analysis: AlphaFailureAnalysis):
        """Update recurring pattern tracking."""
        key = f"{analysis.primary_failure}_{'_'.join(analysis.operator_patterns[:2])}"

        if key in self.patterns:
            pattern = self.patterns[key]
            pattern.frequency += 1
            if analysis.alpha_id not in pattern.affected_alphas:
                pattern.affected_alphas.append(analysis.alpha_id)
        else:
            self.patterns[key] = RecurringPattern(
                pattern_id=key,
                pattern_type=analysis.primary_failure,
                frequency=1,
                affected_alphas=[analysis.alpha_id],
                common_operators=analysis.operator_patterns,
                common_datafields=[],
                common_concepts=[],
                recommendation=self._get_pattern_recommendation(analysis)
            )

    def _get_pattern_recommendation(self, analysis: AlphaFailureAnalysis) -> str:
        """Get recommendation for a pattern type."""
        sig = self.signatures.get(analysis.primary_failure)
        if sig:
            return sig.suggested_fixes[0] if sig.suggested_fixes else ""
        return "Review alpha structure and consider alternative approach"

    def get_warnings(self, alpha_characteristics: Dict) -> List[Dict]:
        """Get warnings for potential failure patterns before simulation."""
        warnings = []

        if alpha_characteristics.get("turnover", 0) > 50:
            warnings.append({
                "type": "high_turnover",
                "warning": "Signal appears high-turnover - consider adding decay",
                "severity": "high"
            })

        if alpha_characteristics.get("operators") and "rank" in alpha_characteristics["operators"]:
            if not alpha_characteristics.get("has_decay"):
                warnings.append({
                    "type": "likely_high_turnover",
                    "warning": "Rank operator without decay typically causes high turnover",
                    "severity": "high"
                })

        if alpha_characteristics.get("neutralization") == "market":
            if alpha_characteristics.get("concepts") and "fundamental" in alpha_characteristics["concepts"]:
                warnings.append({
                    "type": "sector_bias_risk",
                    "warning": "Market neutralization on fundamentals may introduce sector bias",
                    "severity": "medium"
                })

        return warnings

    def get_common_mistakes(self) -> List[Dict]:
        """Get most common failure patterns across all analyzed alphas."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda x: x.frequency,
            reverse=True
        )

        return [
            {
                "pattern_id": p.pattern_id,
                "type": p.pattern_type,
                "frequency": p.frequency,
                "affected_count": len(p.affected_alphas),
                "recommendation": p.recommendation
            }
            for p in sorted_patterns[:10]
        ]

    def find_historical_analogs(
        self,
        operators: List[str],
        concepts: List[str],
        datafields: List[str]
    ) -> List[Dict]:
        """Find historically failed alphas with similar characteristics."""
        matches = []

        for alpha_id, analysis in self.alpha_analyses.items():
            overlap_score = 0

            if set(analysis.operator_patterns).intersection(set(operators)):
                overlap_score += 2
            if any(c in concepts for c in analysis.similar_failures):
                overlap_score += 1

            if overlap_score > 0:
                matches.append({
                    "alpha_id": alpha_id,
                    "primary_failure": analysis.primary_failure,
                    "overlap_score": overlap_score,
                    "suggested_improvements": analysis.suggested_improvements
                })

        return sorted(matches, key=lambda x: x["overlap_score"], reverse=True)[:5]


learning_engine = FailureLearningEngine()


def analyze_failure(
    alpha_id: str,
    expression: str,
    datafields: List[str],
    operators: List[str],
    concepts: List[str],
    failure_modes: List[str],
    sharpe: Optional[float],
    turnover: Optional[float],
    fitness: Optional[float],
    correlated_with: List[str],
    neutralization: str
) -> Dict:
    """Analyze an alpha failure."""
    analysis = learning_engine.analyze_alpha_failure(
        alpha_id, expression, datafields, operators, concepts,
        failure_modes, sharpe, turnover, fitness, correlated_with, neutralization
    )
    return asdict(analysis)


def get_failure_warnings(alpha_characteristics: Dict) -> List[Dict]:
    """Get warnings before alpha simulation."""
    return learning_engine.get_warnings(alpha_characteristics)


def get_common_failures() -> List[Dict]:
    """Get most common failure patterns."""
    return learning_engine.get_common_mistakes()


def find_analogs(operators: List[str], concepts: List[str], datafields: List[str]) -> List[Dict]:
    """Find historical analog failures."""
    return learning_engine.find_historical_analogs(operators, concepts, datafields)
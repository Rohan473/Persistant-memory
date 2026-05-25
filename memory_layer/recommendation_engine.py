"""
Research Recommendation Engine
Suggests unexplored factor combinations and orthogonal research directions.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random


@dataclass
class ResearchRecommendation:
    """A research recommendation."""
    id: str
    recommendation_type: str
    title: str
    description: str
    suggested_factors: List[str]
    suggested_datafields: List[str]
    expected_benefit: str
    risk_level: str
    related_failed_paths: List[str]


@dataclass
class ExplorationCoverage:
    """Coverage of the research exploration space."""
    explored_combinations: Set[Tuple[str, str]]
    unexplored_factors: List[str]
    unexplored_datafields: List[str]
    unexplored_combinations: List[Tuple[str, str]]
    coverage_percentage: float


class RecommendationEngine:
    """Generate research recommendations."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "recommendations.json"

        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        """Load recommendation data."""
        self.recommendations: List[ResearchRecommendation] = []
        self.explored_combos: Set[Tuple[str, str]] = set()
        self.failed_paths: Set[Tuple[str, str]] = set()

        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                    for r in data.get("recommendations", []):
                        self.recommendations.append(ResearchRecommendation(**r))
                    self.explored_combos = {
                        tuple(c) for c in data.get("explored_combos", [])
                    }
                    self.failed_paths = {
                        tuple(f) for f in data.get("failed_paths", [])
                    }
            except Exception:
                pass

    def _save_data(self):
        """Save recommendation data."""
        data = {
            "recommendations": [asdict(r) for r in self.recommendations],
            "explored_combos": [list(c) for c in self.explored_combos],
            "failed_paths": [list(f) for f in self.failed_paths]
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_exploration(
        self,
        factors: List[str],
        datafields: List[str],
        operators: List[str],
        success: bool
    ):
        """Register an exploration attempt."""
        for f in factors:
            for df in datafields:
                combo = (f, df)
                self.explored_combos.add(combo)
                if not success:
                    self.failed_paths.add(combo)

        self._save_data()

    def get_exploration_coverage(
        self,
        all_factors: List[str],
        all_datafields: List[str]
    ) -> ExplorationCoverage:
        """Calculate exploration coverage."""
        total_combos = len(all_factors) * len(all_datafields)
        explored = len(self.explored_combos)

        unexplored_factors = [
            f for f in all_factors
            if not any(f in c[0] for c in self.explored_combos)
        ]

        unexplored_datafields = [
            df for df in all_datafields
            if not any(df in c[1] for c in self.explored_combos)
        ]

        unexplored_combos = []
        for f in all_factors:
            for df in all_datafields:
                if (f, df) not in self.explored_combos:
                    unexplored_combos.append((f, df))

        coverage = explored / total_combos if total_combos > 0 else 0

        return ExplorationCoverage(
            explored_combinations=self.explored_combos,
            unexplored_factors=unexplored_factors,
            unexplored_datafields=unexplored_datafields,
            unexplored_combinations=unexplored_combos[:20],
            coverage_percentage=round(coverage * 100, 1)
        )

    def suggest_unexplored_factors(
        self,
        used_factors: List[str],
        used_datafields: List[str]
    ) -> List[ResearchRecommendation]:
        """Suggest unexplored factor combinations."""
        from memory_layer.factor_ontology import FACTOR_FAMILIES

        recommendations = []

        complement_map = {
            "momentum": ["liquidity", "quality", "volatility"],
            "reversal": ["volatility", "stat_arb"],
            "value": ["quality", "growth", "defensive"],
            "quality": ["value", "defensive", "momentum"],
            "liquidity": ["momentum", "positioning"],
            "growth": ["momentum", "quality"],
            "defensive": ["quality", "liquidity"],
            "volatility": ["liquidity", "defensive"]
        }

        for factor in used_factors:
            if factor in complement_map:
                for suggested in complement_map[factor]:
                    if suggested not in used_factors:
                        rec = ResearchRecommendation(
                            id=f"rec_{factor}_{suggested}",
                            recommendation_type="factor_combine",
                            title=f"Combine {factor} with {suggested}",
                            description=f"Add {suggested} factor for diversification from {factor}",
                            suggested_factors=[factor, suggested],
                            suggested_datafields=[],
                            expected_benefit="Better diversification, potentially higher Sharpe",
                            risk_level="medium",
                            related_failed_paths=[]
                        )
                        recommendations.append(rec)

        return recommendations[:5]

    def suggest_unexplored_datafields(
        self,
        used_datafields: List[str],
        factor: str
    ) -> List[str]:
        """Suggest unexplored datafields for a factor."""
        from memory_layer.factor_ontology import FACTOR_FAMILIES

        if factor not in FACTOR_FAMILIES:
            return []

        family = FACTOR_FAMILIES[factor]
        suggested = []

        for df in family.datafield_indicators:
            if df not in used_datafields:
                suggested.append(df)

        return suggested[:5]

    def avoid_failed_paths(
        self,
        proposed_factors: List[str],
        proposed_datafields: List[str]
    ) -> List[str]:
        """Warn about previously failed paths."""
        warnings = []

        for f in proposed_factors:
            for df in proposed_datafields:
                if (f, df) in self.failed_paths:
                    warnings.append(
                        f"Factor {f} with datafield {df} previously failed - consider alternative"
                    )

        return warnings[:3]

    def recommend_regime_diversification(
        self,
        current_regimes: List[str]
    ) -> List[str]:
        """Recommend exploring different market regimes."""
        all_regimes = [
            "crisis", "recovery", "inflation", "growth_leadership",
            "value_rotation", "volatility_spike", "risk_on", "risk_off"
        ]

        unexplored = [r for r in all_regimes if r not in current_regimes]

        return unexplored[:3]

    def generate_diversity_recommendations(
        self,
        current_alphas: List[Dict]
    ) -> List[ResearchRecommendation]:
        """Generate recommendations for maintaining research diversity."""
        recommendations = []

        current_factors = set()
        current_concepts = set()
        current_datafields = set()

        for alpha in current_alphas:
            current_factors.update(alpha.get("factors", []))
            current_concepts.update(alpha.get("concepts", []))
            current_datafields.update(alpha.get("datafields", []))

        if len(current_factors) < 3:
            recommendations.append(ResearchRecommendation(
                id="rec_factor_diversity",
                recommendation_type="factor_diversity",
                title="Explore more factor families",
                description="Current research is concentrated in few factors",
                suggested_factors=["quality", "momentum", "liquidity"],
                suggested_datafields=["volume", "returns", "close"],
                expected_benefit="More diverse alpha portfolio",
                risk_level="medium",
                related_failed_paths=[]
            ))

        if len(current_datafields) < 5:
            recommendations.append(ResearchRecommendation(
                id="rec_datafield_diversity",
                recommendation_type="datafield_diversity",
                title="Explore new datafields",
                description="Consider underexplored datafields like cap, vwap, beta",
                suggested_factors=[],
                suggested_datafields=["cap", "vwap", "beta_last_30_days_spy", "cashflow_op"],
                expected_benefit="Novel signal sources",
                risk_level="high",
                related_failed_paths=[]
            ))

        if "momentum" in current_concepts and "reversal" not in current_concepts:
            recommendations.append(ResearchRecommendation(
                id="rec_concept_balance",
                recommendation_type="concept_balance",
                title="Explore mean reversion",
                description="Researched momentum but not reversal - consider exploring",
                suggested_factors=["reversal"],
                suggested_datafields=["returns", "close"],
                expected_benefit="Regime diversification",
                risk_level="medium",
                related_failed_paths=[]
            ))

        return recommendations[:3]

    def get_research_roadmap(
        self,
        current_progress: Dict
    ) -> List[Dict]:
        """Generate a research roadmap based on current progress."""
        roadmap = []

        roadmap.append({
            "phase": 1,
            "title": "Factor Foundation",
            "tasks": [
                "Classify all existing alphas by factor family",
                "Map factor-regime performance relationships"
            ],
            "priority": "high"
        })

        roadmap.append({
            "phase": 2,
            "title": "Regime Coverage",
            "tasks": [
                "Add regime annotations to all alphas",
                "Build regime-specific performance matrices"
            ],
            "priority": "medium"
        })

        roadmap.append({
            "phase": 3,
            "title": "Portfolio Construction",
            "tasks": [
                "Compute full correlation matrix",
                "Build orthogonal sleeve recommendations"
            ],
            "priority": "medium"
        })

        roadmap.append({
            "phase": 4,
            "title": "Autonomous Exploration",
            "tasks": [
                "Implement research agent loops",
                "Set up autonomous hypothesis generation"
            ],
            "priority": "low"
        })

        return roadmap


recommendation_engine = RecommendationEngine()


def get_exploration_coverage() -> Dict:
    """Get exploration coverage statistics."""
    from memory_layer.factor_ontology import FACTOR_FAMILIES
    from memory_layer.structure import load_metadata

    metadata = load_metadata()
    all_factors = list(FACTOR_FAMILIES.keys())
    all_datafields = list(set(
        m.get("datafields", []) for m in metadata if m.get("node_type") == "Alpha"
    ))

    coverage = recommendation_engine.get_exploration_coverage(all_factors, all_datafields)
    return asdict(coverage)


def suggest_factors(used_factors: List[str], used_datafields: List[str]) -> List[Dict]:
    """Suggest unexplored factor combinations."""
    recs = recommendation_engine.suggest_unexplored_factors(used_factors, used_datafields)
    return [asdict(r) for r in recs]


def warn_about_failures(proposed_factors: List[str], proposed_datafields: List[str]) -> List[str]:
    """Warn about previously failed paths."""
    return recommendation_engine.avoid_failed_paths(proposed_factors, proposed_datafields)


def recommend_diversity(current_alphas: List[Dict]) -> List[Dict]:
    """Recommend maintaining research diversity."""
    recs = recommendation_engine.generate_diversity_recommendations(current_alphas)
    return [asdict(r) for r in recs]


def get_roadmap(current_progress: Dict) -> List[Dict]:
    """Get research roadmap."""
    return recommendation_engine.get_research_roadmap(current_progress)
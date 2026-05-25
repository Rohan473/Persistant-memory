"""
Correlation and Portfolio Engine
Institutional sleeve analysis and portfolio optimization.
"""

import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from itertools import combinations


@dataclass
class AlphaPairCorrelation:
    """Pairwise correlation between two alphas."""
    alpha_1: str
    alpha_2: str
    correlation: float
    regime_correlations: Dict[str, float]
    factor_overlap: float
    orthogonality_score: float


@dataclass
class PortfolioComposition:
    """Portfolio allocation across alphas."""
    name: str
    alphas: Dict[str, float]
    total_weight: float
    expected_sharpe: float
    expected_turnover: float
    diversification_score: float
    factor_exposures: Dict[str, float]


@dataclass
class SleeveRecommendation:
    """Recommended alpha sleeve for portfolio construction."""
    sleeve_id: str
    description: str
    suggested_alphas: List[str]
    expected_sharpe: float
    correlation_profile: str
    risk_contribution: str
    factor_exposures: Dict[str, float]


class CorrelationEngine:
    """Compute and analyze alpha correlations."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "correlation_data.json"

        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        """Load correlation data."""
        self.pair_correlations: Dict[Tuple[str, str], AlphaPairCorrelation] = {}
        self.alpha_factors: Dict[str, Set[str]] = {}
        self.alpha_concepts: Dict[str, Set[str]] = {}

        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)

                    for key, corr_data in data.get("pair_correlations", {}).items():
                        a1, a2 = key.split("::")
                        self.pair_correlations[(a1, a2)] = AlphaPairCorrelation(**corr_data)

                    self.alpha_factors = {
                        k: set(v) for k, v in data.get("alpha_factors", {}).items()
                    }
                    self.alpha_concepts = {
                        k: set(v) for k, v in data.get("alpha_concepts", {}).items()
                    }
            except Exception:
                pass

    def _save_data(self):
        """Save correlation data."""
        data = {
            "pair_correlations": {
                f"{k[0]}::{k[1]}": asdict(v)
                for k, v in self.pair_correlations.items()
            },
            "alpha_factors": {k: list(v) for k, v in self.alpha_factors.items()},
            "alpha_concepts": {k: list(v) for k, v in self.alpha_concepts.items()}
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_alpha(
        self,
        alpha_id: str,
        factor_families: List[str],
        concepts: List[str],
        datafields: List[str]
    ):
        """Register alpha characteristics for correlation analysis."""
        self.alpha_factors[alpha_id] = set(factor_families)
        self.alpha_concepts[alpha_id] = set(concepts)
        self._save_data()

    def compute_correlation(
        self,
        alpha_1: str,
        alpha_2: str,
        empirical_correlation: Optional[float] = None
    ) -> AlphaPairCorrelation:
        """Compute correlation between two alphas."""
        key = (alpha_1, alpha_2)
        if key in self.pair_correlations:
            return self.pair_correlations[key]

        factors_1 = self.alpha_factors.get(alpha_1, set())
        factors_2 = self.alpha_factors.get(alpha_2, set())

        if factors_1 and factors_2:
            overlap = len(factors_1.intersection(factors_2))
            total = len(factors_1.union(factors_2))
            factor_overlap = overlap / total if total > 0 else 0
        else:
            factor_overlap = 0

        concepts_1 = self.alpha_concepts.get(alpha_1, set())
        concepts_2 = self.alpha_concepts.get(alpha_2, set())

        if concepts_1 and concepts_2:
            concept_overlap = len(concepts_1.intersection(concepts_2)) / max(1, len(concepts_1.union(concepts_2)))
        else:
            concept_overlap = 0

        if empirical_correlation is not None:
            correlation = empirical_correlation
        else:
            correlation = (factor_overlap + concept_overlap) / 2

        orthogonality_score = 1 - correlation

        corr = AlphaPairCorrelation(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            correlation=correlation,
            regime_correlations={},
            factor_overlap=factor_overlap,
            orthogonality_score=orthogonality_score
        )

        self.pair_correlations[key] = corr
        self._save_data()

        return corr

    def get_pair_correlation(self, alpha_1: str, alpha_2: str) -> Optional[float]:
        """Get correlation between two alphas."""
        key = (alpha_1, alpha_2)
        if key in self.pair_correlations:
            return self.pair_correlations[key].correlation
        return None

    def find_orthogonal_alphas(
        self,
        target_alpha: str,
        max_correlation: float = 0.3,
        candidate_alphas: List[str] = None
    ) -> List[Dict]:
        """Find alphas orthogonal to target alpha."""
        orthogonal = []

        for alpha in (candidate_alphas or list(self.alpha_factors.keys())):
            if alpha == target_alpha:
                continue

            corr = self.compute_correlation(target_alpha, alpha)

            if corr.correlation <= max_correlation:
                orthogonal.append({
                    "alpha_id": alpha,
                    "correlation": corr.correlation,
                    "factor_overlap": corr.factor_overlap,
                    "orthogonality_score": corr.orthogonality_score
                })

        return sorted(orthogonal, key=lambda x: x["orthogonality_score"], reverse=True)

    def get_factor_overlap_matrix(self, alphas: List[str]) -> Dict:
        """Get factor overlap matrix for a set of alphas."""
        matrix = {}

        for a1, a2 in combinations(alphas, 2):
            corr = self.compute_correlation(a1, a2)
            matrix[f"{a1}-{a2}"] = {
                "factor_overlap": corr.factor_overlap,
                "orthogonality": corr.orthogonality_score
            }

        return matrix


class PortfolioEngine:
    """Portfolio construction and optimization."""

    def __init__(self, correlation_engine: CorrelationEngine):
        self.correlation = correlation_engine

    def compute_portfolio_metrics(
        self,
        alphas: Dict[str, float],
        alpha_metrics: Dict[str, Dict]
    ) -> Dict:
        """Compute portfolio-level metrics."""
        if not alphas:
            return {}

        weights = np.array(list(alphas.values()))
        weights = weights / weights.sum()

        sharpes = np.array([
            alpha_metrics.get(a, {}).get("sharpe", 0) for a in alphas.keys()
        ])
        turnovers = np.array([
            alpha_metrics.get(a, {}).get("turnover", 0) for a in alphas.keys()
        ])

        expected_sharpe = np.dot(weights, sharpes)
        expected_turnover = np.dot(weights, turnovers)

        alphas_list = list(alphas.keys())
        avg_correlation = 0
        pairs = 0

        for a1, a2 in combinations(alphas_list, 2):
            corr = self.correlation.get_pair_correlation(a1, a2)
            if corr is not None:
                avg_correlation += corr
                pairs += 1

        if pairs > 0:
            avg_correlation /= pairs

        diversification_score = 1 - avg_correlation

        factor_exposures = self._compute_factor_exposures(alphas_list)

        return {
            "expected_sharpe": round(expected_sharpe, 3),
            "expected_turnover": round(expected_turnover, 2),
            "diversification_score": round(diversification_score, 3),
            "factor_exposures": factor_exposures,
            "avg_correlation": round(avg_correlation, 3)
        }

    def _compute_factor_exposures(self, alphas: List[str]) -> Dict[str, float]:
        """Compute portfolio factor exposures."""
        factor_counts = defaultdict(int)

        for alpha in alphas:
            factors = self.correlation.alpha_factors.get(alpha, set())
            for f in factors:
                factor_counts[f] += 1

        total = len(alphas)
        if total == 0:
            return {}

        return {
            f: round(cnt / total, 3)
            for f, cnt in factor_counts.items()
        }

    def optimize_weights(
        self,
        candidate_alphas: List[str],
        alpha_metrics: Dict[str, Dict],
        target_sharpe: float = 1.25,
        max_turnover: float = 50,
        max_factor_exposure: float = 0.4,
        min_alphas: int = 3,
        max_alphas: int = 10
    ) -> Dict:
        """Find optimal portfolio weights."""
        best_portfolio = None
        best_score = -999

        for num_alphas in range(min_alphas, min(max_alphas + 1, len(candidate_alphas) + 1)):
            for combo in combinations(candidate_alphas, num_alphas):
                equal_weights = {a: 1.0 / num_alphas for a in combo}
                metrics = self.compute_portfolio_metrics(equal_weights, alpha_metrics)

                if metrics.get("expected_turnover", 999) > max_turnover:
                    continue

                sharpe_score = min(metrics.get("expected_sharpe", 0), target_sharpe)
                div_score = metrics.get("diversification_score", 0)

                score = sharpe_score * 0.6 + div_score * 0.4

                if score > best_score:
                    best_score = score
                    best_portfolio = {
                        "alphas": equal_weights,
                        "metrics": metrics,
                        "score": score
                    }

        return best_portfolio or {"alphas": {}, "metrics": {}, "score": 0}

    def suggest_sleeves(
        self,
        alpha_pool: List[str],
        alpha_metrics: Dict[str, Dict],
        target_sleeves: int = 4
    ) -> List[SleeveRecommendation]:
        """Suggest orthogonal alpha sleeves for diversified portfolio."""
        from memory_layer.factor_ontology import ontology_engine

        recommendations = []

        factor_groups = defaultdict(list)
        for alpha in alpha_pool:
            factors = self.correlation.alpha_factors.get(alpha, set())
            if factors:
                primary = list(factors)[0]
            else:
                primary = "other"
            factor_groups[primary].append(alpha)

        for factor, alphas in factor_groups.items():
            if len(alphas) < 2:
                continue

            orthogonal = self.correlation.find_orthogonal_alphas(
                alphas[0], max_correlation=0.3, candidate_alphas=alphas[1:]
            )

            sleeve_alphas = [alphas[0]]
            for o in orthogonal[:2]:
                sleeve_alphas.append(o["alpha_id"])

            if len(sleeve_alphas) >= 2:
                weights = {a: 1.0 / len(sleeve_alphas) for a in sleeve_alphas}
                metrics = self.compute_portfolio_metrics(weights, alpha_metrics)

                rec = SleeveRecommendation(
                    sleeve_id=f"factor_{factor}",
                    description=f"{factor.title()} factor sleeve",
                    suggested_alphas=sleeve_alphas,
                    expected_sharpe=metrics.get("expected_sharpe", 0),
                    correlation_profile="low_correlation",
                    risk_contribution="primary_factor_exposure",
                    factor_exposures=metrics.get("factor_exposures", {})
                )
                recommendations.append(rec)

        return recommendations[:target_sleeves]


correlation_engine = CorrelationEngine()
portfolio_engine = PortfolioEngine(correlation_engine)


def register_alpha_factors(
    alpha_id: str,
    factor_families: List[str],
    concepts: List[str],
    datafields: List[str]
) -> None:
    """Register alpha for correlation analysis."""
    correlation_engine.register_alpha(alpha_id, factor_families, concepts, datafields)


def get_pair_correlation(alpha_1: str, alpha_2: str) -> Optional[float]:
    """Get correlation between two alphas."""
    return correlation_engine.get_pair_correlation(alpha_1, alpha_2)


def find_orthogonal_sleeves(
    target_alpha: str,
    candidate_alphas: List[str] = None
) -> List[Dict]:
    """Find orthogonal alpha sleeves."""
    return correlation_engine.find_orthogonal_alphas(target_alpha, candidate_alphas=candidate_alphas)


def compute_portfolio_metrics(
    alphas: Dict[str, float],
    alpha_metrics: Dict[str, Dict]
) -> Dict:
    """Compute portfolio-level metrics."""
    return portfolio_engine.compute_portfolio_metrics(alphas, alpha_metrics)


def optimize_portfolio(
    candidate_alphas: List[str],
    alpha_metrics: Dict[str, Dict]
) -> Dict:
    """Find optimal portfolio weights."""
    return portfolio_engine.optimize_weights(candidate_alphas, alpha_metrics)


def suggest_sleeves(
    alpha_pool: List[str],
    alpha_metrics: Dict[str, Dict]
) -> List[Dict]:
    """Suggest alpha sleeves for portfolio."""
    recs = portfolio_engine.suggest_sleeves(alpha_pool, alpha_metrics)
    return [asdict(r) for r in recs]
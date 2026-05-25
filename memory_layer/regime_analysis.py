"""
Regime Analysis Engine
Regime-aware research analysis with yearly performance tracking.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime/year."""
    year: int
    regime: str
    sharpe: Optional[float]
    returns: Optional[float]
    turnover: Optional[float]
    drawdown: Optional[float]
    fitness: Optional[float]
    sample_size: int = 0


@dataclass
class RegimeSensitivity:
    """Computed regime sensitivity metrics for an alpha."""
    alpha_id: str
    best_regimes: List[Tuple[str, float]]
    worst_regimes: List[Tuple[str, float]]
    regime_correlation: float
    crisis_convexity: float
    macro_dependence: float


@dataclass
class AlphaRegimeProfile:
    """Complete regime profile for an alpha."""
    alpha_id: str
    yearly_performance: List[RegimePerformance]
    sensitivity: Optional[RegimeSensitivity]
    macro_related: bool
    sector_related: bool
    regime_strengths: List[str]
    regime_weaknesses: List[str]


REGIMES = {
    "crisis": {
        "name": "Crisis/Recession",
        "description": "Market stress periods (2008, 2020 COVID, 2022 rate shock)",
        "indicators": ["VIX>25", "SPX<-20%", "credit_spreads_wide"],
        "years": [2008, 2009, 2020, 2022]
    },
    "recovery": {
        "name": "Recovery",
        "description": "Post-crisis rebounds (2009-2010, 2021)",
        "indicators": ["VIX<20", "SPX>15%", "growth_outperform"],
        "years": [2009, 2010, 2021]
    },
    "inflation": {
        "name": "Inflation Regime",
        "description": "High inflation periods (2022, 1970s)",
        "indicators": ["CPI>4%", "rates_rising", "value_outperform"],
        "years": [2022]
    },
    "deflation": {
        "name": "Deflation",
        "description": "Falling prices (2009-2010)",
        "indicators": ["CPI<1%", "bonds_rally"],
        "years": [2009, 2010]
    },
    "growth_leadership": {
        "name": "Growth Leadership",
        "description": "Growth beats value (2013-2019, 2020-2021)",
        "indicators": ["tech_rally", "rates_low", "NDX>SPX"],
        "years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    },
    "value_rotation": {
        "name": "Value Rotation",
        "description": "Value beats growth (2008-2009, 2022)",
        "indicators": ["value_outperform", "rates_rising"],
        "years": [2008, 2009, 2022]
    },
    "volatility_spike": {
        "name": "Volatility Spike",
        "description": "VIX spikes (2018, 2020, 2022)",
        "indicators": ["VIX>30", "correlation_spike"],
        "years": [2018, 2020, 2022]
    },
    "liquidity_stress": {
        "name": "Liquidity Stress",
        "description": "Reduced market liquidity (2008, 2020)",
        "indicators": ["bid_ask_wide", "volume_drop"],
        "years": [2008, 2020]
    },
    "ai_speculative": {
        "name": "AI/Speculative",
        "description": "Speculative AI/tech bubble (2023-2024)",
        "indicators": ["tech_rally", "IPO_activity", "NVDA_rally"],
        "years": [2023, 2024]
    },
    "risk_on": {
        "name": "Risk-On",
        "description": "High risk appetite (2017, 2019, 2021)",
        "indicators": ["VIX<15", "high_beta_outperform"],
        "years": [2017, 2019, 2021]
    },
    "risk_off": {
        "name": "Risk-Off",
        "description": "Flight to safety (2018, 2022)",
        "indicators": ["VIX>20", "bonds_rally", "gold_rally"],
        "years": [2018, 2022]
    },
}


class RegimeAnalyzer:
    """Regime-aware analysis for alphas."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "regime_data.json"

        self.data_path = data_path
        self.regimes = REGIMES
        self._load_data()

    def _load_data(self):
        """Load existing regime data."""
        self.alpha_profiles: Dict[str, AlphaRegimeProfile] = {}
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                    for alpha_id, profile_data in data.get("profiles", {}).items():
                        yearly = [RegimePerformance(**y) for y in profile_data.get("yearly_performance", [])]
                        sensitivity_data = profile_data.get("sensitivity")
                        sensitivity = None
                        if sensitivity_data:
                            sensitivity = RegimeSensitivity(
                                alpha_id=alpha_id,
                                best_regime=[(r, s) for r, s in sensitivity_data.get("best_regimes", [])],
                                worst_regimes=[(r, s) for r, s in sensitivity_data.get("worst_regimes", [])],
                                regime_correlation=sensitivity_data.get("regime_correlation", 0),
                                crisis_convexity=sensitivity_data.get("crisis_convexity", 0),
                                macro_dependence=sensitivity_data.get("macro_dependence", 0)
                            )
                        self.alpha_profiles[alpha_id] = AlphaRegimeProfile(
                            alpha_id=alpha_id,
                            yearly_performance=yearly,
                            sensitivity=sensitivity,
                            macro_related=profile_data.get("macro_related", False),
                            sector_related=profile_data.get("sector_related", False),
                            regime_strengths=profile_data.get("regime_strengths", []),
                            regime_weaknesses=profile_data.get("regime_weaknesses", [])
                        )
            except Exception:
                pass

    def _save_data(self):
        """Save regime data to disk."""
        data = {"profiles": {}}
        for alpha_id, profile in self.alpha_profiles.items():
            data["profiles"][alpha_id] = {
                "yearly_performance": [asdict(y) for y in profile.yearly_performance],
                "sensitivity": None if not profile.sensitivity else {
                    "best_regimes": profile.sensitivity.best_regimes,
                    "worst_regimes": profile.sensitivity.worst_regimes,
                    "regime_correlation": profile.sensitivity.regime_correlation,
                    "crisis_convexity": profile.sensitivity.crisis_convexity,
                    "macro_dependence": profile.sensitivity.macro_dependence
                },
                "macro_related": profile.macro_related,
                "sector_related": profile.sector_related,
                "regime_strengths": profile.regime_strengths,
                "regime_weaknesses": profile.regime_weaknesses
            }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_yearly_performance(
        self,
        alpha_id: str,
        year: int,
        sharpe: Optional[float] = None,
        returns: Optional[float] = None,
        turnover: Optional[float] = None,
        drawdown: Optional[float] = None,
        fitness: Optional[float] = None
    ) -> None:
        """Add yearly performance data for an alpha."""
        if alpha_id not in self.alpha_profiles:
            self.alpha_profiles[alpha_id] = AlphaRegimeProfile(
                alpha_id=alpha_id,
                yearly_performance=[],
                sensitivity=None,
                macro_related=False,
                sector_related=False,
                regime_strengths=[],
                regime_weaknesses=[]
            )

        perf = RegimePerformance(
            year=year,
            regime=self._infer_regime(year),
            sharpe=sharpe,
            returns=returns,
            turnover=turnover,
            drawdown=drawdown,
            fitness=fitness
        )

        for i, existing in enumerate(self.alpha_profiles[alpha_id].yearly_performance):
            if existing.year == year:
                self.alpha_profiles[alpha_id].yearly_performance[i] = perf
                break
        else:
            self.alpha_profiles[alpha_id].yearly_performance.append(perf)

        self._save_data()

    def _infer_regime(self, year: int) -> str:
        """Infer the market regime for a given year."""
        for regime, info in self.regimes.items():
            if year in info.get("years", []):
                return regime
        return "normal"

    def compute_regime_sensitivity(self, alpha_id: str) -> Optional[RegimeSensitivity]:
        """Compute regime sensitivity metrics for an alpha."""
        if alpha_id not in self.alpha_profiles:
            return None

        profile = self.alpha_profiles[alpha_id]
        if not profile.yearly_performance:
            return None

        regime_sharpes = defaultdict(list)
        for perf in profile.yearly_performance:
            if perf.sharpe is not None:
                regime_sharpes[perf.regime].append(perf.sharpe)

        avg_by_regime = {r: np.mean(sharpes) for r, sharpes in regime_sharpes.items()}

        sorted_regimes = sorted(avg_by_regime.items(), key=lambda x: x[1], reverse=True)
        best = [(r, s) for r, s in sorted_regimes[:3] if s is not None]
        worst = [(r, s) for r, s in sorted_regimes[-3:] if s is not None]

        if best and worst:
            crisis_sharpe = avg_by_regime.get("crisis", 0)
            recovery_sharpe = avg_by_regime.get("recovery", 0)
            crisis_convexity = crisis_sharpe - recovery_sharpe
        else:
            crisis_convexity = 0

        macro_related_factors = {"value", "quality", "growth", "defensive", "liquidity", "volatility"}
        macro_dependence = 0.5

        return RegimeSensitivity(
            alpha_id=alpha_id,
            best_regimes=best,
            worst_regimes=worst,
            regime_correlation=0.3,
            crisis_convexity=crisis_convexity,
            macro_dependence=macro_dependence
        )

    def get_regime_performance(self, alpha_id: str) -> Dict:
        """Get performance breakdown by regime for an alpha."""
        if alpha_id not in self.alpha_profiles:
            return {}

        profile = self.alpha_profiles[alpha_id]
        regime_metrics = defaultdict(lambda: {"sharpes": [], "returns": [], "turnovers": []})

        for perf in profile.yearly_performance:
            if perf.sharpe is not None:
                regime_metrics[perf.regime]["sharpes"].append(perf.sharpe)
            if perf.returns is not None:
                regime_metrics[perf.regime]["returns"].append(perf.returns)
            if perf.turnover is not None:
                regime_metrics[perf.regime]["turnovers"].append(perf.turnover)

        result = {}
        for regime, metrics in regime_metrics.items():
            result[regime] = {
                "avg_sharpe": round(np.mean(metrics["sharpes"]), 2) if metrics["sharpes"] else None,
                "avg_return": round(np.mean(metrics["returns"]), 2) if metrics["returns"] else None,
                "avg_turnover": round(np.mean(metrics["turnovers"]), 2) if metrics["turnovers"] else None,
                "sample_years": len(metrics["sharpes"])
            }

        return result

    def find_alphas_by_regime(
        self,
        target_regime: str,
        min_sharpe: float = 0.5,
        min_years: int = 1
    ) -> List[Dict]:
        """Find alphas that perform well in a specific regime."""
        results = []

        for alpha_id, profile in self.alpha_profiles.items():
            regime_perf = self.get_regime_performance(alpha_id)
            regime_data = regime_perf.get(target_regime)

            if regime_data and regime_data.get("avg_sharpe", 0) >= min_sharpe:
                if regime_data.get("sample_years", 0) >= min_years:
                    results.append({
                        "alpha_id": alpha_id,
                        "avg_sharpe": regime_data["avg_sharpe"],
                        "sample_years": regime_data["sample_years"]
                    })

        results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
        return results

    def detect_crisis_alphas(self, crisis_sharpe_threshold: float = 0.0) -> List[Dict]:
        """Find alphas that maintain performance during crises."""
        crisis_alphas = []

        for alpha_id, profile in self.alpha_profiles.items():
            regime_perf = self.get_regime_performance(alpha_id)
            crisis_perf = regime_perf.get("crisis")

            if crisis_perf and crisis_perf.get("avg_sharpe", -999) >= crisis_sharpe_threshold:
                crisis_alphas.append({
                    "alpha_id": alpha_id,
                    "crisis_sharpe": crisis_perf["avg_sharpe"],
                    "recovery_sharpe": regime_perf.get("recovery", {}).get("avg_sharpe")
                })

        return sorted(crisis_alphas, key=lambda x: x["crisis_sharpe"], reverse=True)

    def infer_regime_dependencies(
        self,
        factor_families: List[str],
        neutralization: str,
        universe: str
    ) -> Dict:
        """Infer regime dependencies based on alpha characteristics."""
        macro_related = {"value", "quality", "growth", "defensive", "liquidity", "volatility",
                         "carry", "recovery", "distress"}
        sector_related = {"sector_sensitive", "defensive"}

        is_macro = any(f in macro_related for f in factor_families)
        is_sector = any(f in sector_related for f in factor_families)

        if neutralization == "None" and universe in ["TOP3000", "TOP5000"]:
            risk_on_strength = "high"
        else:
            risk_on_strength = "medium"

        if "value" in factor_families or "quality" in factor_families:
            crisis_rating = "strong"
            inflation_rating = "strong"
        else:
            crisis_rating = "neutral"
            inflation_rating = "neutral"

        return {
            "macro_related": is_macro,
            "sector_related": is_sector,
            "risk_on_strength": risk_on_strength,
            "crisis_resilience": crisis_rating,
            "inflation_sensitivity": inflation_rating,
            "recommended_regimes": self._recommend_regimes(is_macro, is_sector, neutralization)
        }

    def _recommend_regimes(self, is_macro: bool, is_sector: bool, neutralization: str) -> List[str]:
        """Recommend appropriate regimes based on alpha characteristics."""
        regimes = []

        if is_macro:
            regimes.extend(["risk_on", "risk_off", "inflation"])
        else:
            regimes.extend(["growth_leadership", "risk_on"])

        if is_sector:
            regimes.append("sector_rotation")

        if neutralization == "None":
            regimes.append("momentum")

        return list(set(regimes))

    def get_regime_info(self, regime: str) -> Optional[Dict]:
        """Get detailed information about a regime."""
        if regime in self.regimes:
            info = self.regimes[regime].copy()
            info["id"] = regime
            return info
        return None

    def list_regimes(self) -> List[Dict]:
        """List all available regimes."""
        return [
            {"id": r, "name": info["name"], "description": info["description"]}
            for r, info in self.regimes.items()
        ]


analyzer = RegimeAnalyzer()


def add_regime_performance(
    alpha_id: str,
    year: int,
    sharpe: Optional[float] = None,
    returns: Optional[float] = None,
    turnover: Optional[float] = None,
    drawdown: Optional[float] = None,
    fitness: Optional[float] = None
) -> None:
    """Add yearly performance data."""
    analyzer.add_yearly_performance(alpha_id, year, sharpe, returns, turnover, drawdown, fitness)


def get_regime_performance(alpha_id: str) -> Dict:
    """Get regime breakdown for an alpha."""
    return analyzer.get_regime_performance(alpha_id)


def find_regime_alphas(regime: str, min_sharpe: float = 0.5) -> List[Dict]:
    """Find alphas that perform well in a specific regime."""
    return analyzer.find_alphas_by_regime(regime, min_sharpe)


def get_alpha_regime_profile(alpha_id: str) -> Optional[Dict]:
    """Get complete regime profile for an alpha."""
    if alpha_id in analyzer.alpha_profiles:
        return asdict(analyzer.alpha_profiles[alpha_id])
    return None


def infer_regime_dependencies(
    factor_families: List[str],
    neutralization: str,
    universe: str
) -> Dict:
    """Infer regime dependencies from alpha characteristics."""
    return analyzer.infer_regime_dependencies(factor_families, neutralization, universe)
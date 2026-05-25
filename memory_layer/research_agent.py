"""
Autonomous Research Agent
Autonomous research agents that generate hypotheses, analyze failed alphas, and propose new sleeves.
"""

import json
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict


@dataclass
class ResearchHypothesis:
    """A research hypothesis to explore."""
    hypothesis_id: str
    created_at: str
    description: str
    factors: List[str]
    datafields: List[str]
    operators: List[str]
    expected_outcome: str
    status: str  # pending, testing, validated, rejected
    related_alpha_id: Optional[str]


@dataclass
class AgentTask:
    """A task for the research agent."""
    task_id: str
    task_type: str  # analyze_failure, explore_hypothesis, find_orthogonal, detect_gaps
    description: str
    parameters: Dict
    status: str  # pending, running, completed, failed
    result: Optional[Dict]
    created_at: str
    completed_at: Optional[str]


@dataclass
class ExplorationRecord:
    """Record of an exploration attempt."""
    exploration_id: str
    timestamp: str
    exploration_type: str
    hypothesis_id: Optional[str]
    result: str  # success, failure, partial
    findings: List[str]
    new_alphas_proposed: List[str]


class ResearchAgent:
    """
    Autonomous research agent that:
    - Generates hypotheses
    - Analyzes failed alphas
    - Proposes new sleeves
    - Detects unexplored areas
    - Maintains research diversity
    """

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "research_agent_data.json"

        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        """Load research agent data."""
        self.hypotheses: Dict[str, ResearchHypothesis] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.explorations: List[ExplorationRecord] = []
        self.explored_paths: Set[Tuple[str, str]] = set()
        self.failed_paths: Set[Tuple[str, str]] = set()

        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)

                    for h_id, h_data in data.get("hypotheses", {}).items():
                        self.hypotheses[h_id] = ResearchHypothesis(**h_data)

                    for t_id, t_data in data.get("tasks", {}).items():
                        self.tasks[t_id] = AgentTask(**t_data)

                    for e_data in data.get("explorations", []):
                        self.explorations.append(ExplorationRecord(**e_data))

                    self.explored_paths = {tuple(p) for p in data.get("explored_paths", [])}
                    self.failed_paths = {tuple(p) for p in data.get("failed_paths", [])}
            except Exception:
                pass

    def _save_data(self):
        """Save research agent data."""
        data = {
            "hypotheses": {k: asdict(v) for k, v in self.hypotheses.items()},
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "explorations": [asdict(e) for e in self.explorations],
            "explored_paths": [list(p) for p in self.explored_paths],
            "failed_paths": [list(p) for p in self.failed_paths]
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_hypothesis(
        self,
        focus_area: Optional[str] = None,
        based_on_alpha: Optional[str] = None
    ) -> ResearchHypothesis:
        """Generate a new research hypothesis."""
        from .factor_ontology import FACTOR_FAMILIES
        from .failure_learning import learning_engine
        from .correlation_engine import correlation_engine

        hypothesis_id = f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if based_on_alpha:
            from .structure import load_metadata
            metadata = load_metadata()
            alpha_meta = None
            for m in metadata:
                if m.get("name") == based_on_alpha and m.get("node_type") == "Alpha":
                    alpha_meta = m
                    break

            if alpha_meta:
                existing_factors = set(alpha_meta.get("concepts", []))
                existing_ops = set(alpha_meta.get("operators", []))
                existing_dfs = set(alpha_meta.get("datafields", []))

                complementary = {
                    "momentum": ["liquidity", "quality", "volatility"],
                    "reversal": ["volatility", "stat_arb"],
                    "value": ["quality", "growth", "defensive"],
                    "quality": ["defensive", "value"],
                }

                suggested_factors = []
                for ef in existing_factors:
                    if ef in complementary:
                        suggested_factors.extend(complementary[ef])

                suggested_factors = list(set(suggested_factors) - existing_factors)[:3]

                new_ops = list(set(["ts_rank", "ts_mean", "rank", "group_neutralize"]) - existing_ops)[:2]

                hypothesis = ResearchHypothesis(
                    hypothesis_id=hypothesis_id,
                    created_at=datetime.now().isoformat(),
                    description=f"Explore complementary factors for {based_on_alpha}",
                    factors=suggested_factors,
                    datafields=[],
                    operators=new_ops,
                    expected_outcome="Find orthogonal alpha with better Sharpe",
                    status="pending",
                    related_alpha_id=based_on_alpha
                )
            else:
                hypothesis = self._generate_random_hypothesis(hypothesis_id, focus_area)
        else:
            hypothesis = self._generate_random_hypothesis(hypothesis_id, focus_area)

        self.hypotheses[hypothesis_id] = hypothesis
        self._save_data()
        return hypothesis

    def _generate_random_hypothesis(
        self,
        hypothesis_id: str,
        focus_area: Optional[str]
    ) -> ResearchHypothesis:
        """Generate a random research hypothesis."""
        from .factor_ontology import FACTOR_FAMILIES

        all_factors = list(FACTOR_FAMILIES.keys())
        factor_combos = [
            ("quality", "value"),
            ("momentum", "liquidity"),
            ("reversal", "volatility"),
            ("growth", "momentum"),
            ("value", "defensive"),
        ]

        if focus_area and focus_area in all_factors:
            selected_factors = [focus_area, random.choice(all_factors)]
        else:
            selected_factors = random.choice(factor_combos)

        operators = random.sample(
            ["ts_rank", "ts_mean", "rank", "group_neutralize", "ts_corr", "ts_decay_linear"],
            k=random.randint(1, 2)
        )

        datafields = random.sample(
            ["close", "volume", "returns", "ebit", "capex", "revenue"],
            k=random.randint(1, 2)
        )

        descriptions = [
            f"Explore {selected_factors[0]} factor with different operators",
            f"Combine {selected_factors[0]} and {selected_factors[1]} for diversification",
            f"Test {operators[0]} on different datafields",
            f"Find orthogonal alternative to current research",
        ]

        return ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            created_at=datetime.now().isoformat(),
            description=random.choice(descriptions),
            factors=selected_factors,
            datafields=datafields,
            operators=operators,
            expected_outcome="Discover new alpha with positive Sharpe",
            status="pending",
            related_alpha_id=None
        )

    def analyze_failed_alpha(self, alpha_id: str) -> Dict:
        """Analyze a failed alpha and propose improvements."""
        from .failure_learning import learning_engine, find_analogs

        metadata = []
        try:
            from .structure import load_metadata
            metadata = load_metadata()
        except:
            pass

        alpha_meta = None
        for m in metadata:
            if m.get("name") == alpha_id and m.get("node_type") == "Alpha":
                alpha_meta = m
                break

        if not alpha_meta:
            return {"error": "Alpha not found"}

        expression = alpha_meta.get("expression", "")
        operators = alpha_meta.get("operators", [])
        concepts = alpha_meta.get("concepts", [])
        datafields = alpha_meta.get("datafields", [])
        failure_modes = alpha_meta.get("failure_modes", [])

        analysis = learning_engine.analyze_alpha_failure(
            alpha_id=alpha_id,
            expression=expression,
            datafields=datafields,
            operators=operators,
            concepts=concepts,
            failure_modes=failure_modes,
            sharpe=alpha_meta.get("sharpe"),
            turnover=alpha_meta.get("turnover"),
            fitness=alpha_meta.get("fitness"),
            correlated_with=alpha_meta.get("correlated_with", []),
            neutralization=alpha_meta.get("settings", ["market"])[0] if alpha_meta.get("settings") else "market"
        )

        analogs = find_analogs(operators, concepts, datafields)

        return {
            "alpha_id": alpha_id,
            "analysis": asdict(analysis),
            "historical_analogs": analogs,
            "suggested_fixes": analysis.get("suggested_improvements", [])
        }

    def find_orthogonal_sleeve(
        self,
        alpha_id: str,
        max_candidates: int = 5
    ) -> List[Dict]:
        """Find orthogonal sleeve for an alpha."""
        from .correlation_engine import correlation_engine

        orthogonal = correlation_engine.find_orthogonal_alphas(
            alpha_id, max_correlation=0.2
        )

        return orthogonal[:max_candidates]

    def detect_research_gaps(self) -> Dict:
        """Detect unexplored areas in the research space."""
        from .factor_ontology import FACTOR_FAMILIES
        from .regime_analysis import analyzer

        used_factors = set()
        used_concepts = set()

        try:
            from .structure import load_metadata
            metadata = load_metadata()
            for m in metadata:
                if m.get("node_type") == "Alpha":
                    used_factors.update(m.get("concepts", []))
                    used_concepts.update(m.get("operators", []))
        except:
            pass

        all_factors = set(FACTOR_FAMILIES.keys())
        unexplored_factors = all_factors - used_factors

        regimes_with_alphas = set()
        for alpha_id in analyzer.alpha_profiles:
            regime_perf = analyzer.get_regime_performance(alpha_id)
            regimes_with_alphas.update(regime_perf.keys())

        all_regimes = set(analyzer.regimes.keys())
        unexplored_regimes = all_regimes - regimes_with_alphas

        return {
            "unexplored_factors": list(unexplored_factors),
            "unexplored_regimes": list(unexplored_regimes),
            "recommended_exploration": [
                f"Explore {f} factor" for f in unexplored_factors[:3]
            ] + [
                f"Test in {r} regime" for r in unexplored_regimes[:2]
            ]
        }

    def maintain_diversity(self, current_alphas: List[str]) -> Dict:
        """Analyze and recommend maintaining research diversity."""
        factor_counts = defaultdict(int)
        operator_counts = defaultdict(int)
        concept_counts = defaultdict(int)

        try:
            from .structure import load_metadata
            metadata = load_metadata()
            for m in metadata:
                if m.get("node_type") == "Alpha" and m.get("name") in current_alphas:
                    for f in m.get("concepts", []):
                        factor_counts[f] += 1
                    for op in m.get("operators", []):
                        operator_counts[op] += 1
                    for c in m.get("concepts", []):
                        concept_counts[c] += 1
        except:
            pass

        recommendations = []

        if len(factor_counts) < 3:
            recommendations.append({
                "type": "factor_diversity",
                "message": "Research concentrates in few factors - consider exploring new factors",
                "suggested_factors": list(set(list(factor_counts.keys()) + ["quality", "momentum", "liquidity"]))[:5]
            })

        dominant_factors = [f for f, c in factor_counts.items() if c > 3]
        if dominant_factors:
            recommendations.append({
                "type": "factor_concentration",
                "message": f"Heavy concentration in {dominant_factors}",
                "suggested_alternatives": ["reversal", "volatility", "stat_arb"]
            })

        return {
            "current_diversity": {
                "factors": dict(factor_counts),
                "operators": dict(operator_counts),
                "concepts": dict(concept_counts)
            },
            "recommendations": recommendations
        }

    def create_task(
        self,
        task_type: str,
        description: str,
        parameters: Dict
    ) -> AgentTask:
        """Create a new agent task."""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            parameters=parameters,
            status="pending",
            result=None,
            created_at=datetime.now().isoformat(),
            completed_at=None
        )

        self.tasks[task_id] = task
        self._save_data()
        return task

    def complete_task(self, task_id: str, result: Dict) -> None:
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result = result
            self.tasks[task_id].completed_at = datetime.now().isoformat()
            self._save_data()

    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks."""
        return [
            asdict(t) for t in self.tasks.values()
            if t.status == "pending"
        ]

    def record_exploration(
        self,
        exploration_type: str,
        hypothesis_id: Optional[str],
        result: str,
        findings: List[str],
        new_alphas: List[str]
    ) -> None:
        """Record an exploration attempt."""
        exploration = ExplorationRecord(
            exploration_id=f"exp_{len(self.explorations) + 1}",
            timestamp=datetime.now().isoformat(),
            exploration_type=exploration_type,
            hypothesis_id=hypothesis_id,
            result=result,
            findings=findings,
            new_alphas_proposed=new_alphas
        )

        self.explorations.append(exploration)

        if result == "success":
            for finding in findings:
                if "factor" in finding.lower() and "datafield" in finding.lower():
                    self.explored_paths.add((finding.split()[1], finding.split()[3]))
        elif result == "failure":
            for finding in findings:
                if "factor" in finding.lower() and "datafield" in finding.lower():
                    self.failed_paths.add((finding.split()[1], finding.split()[3]))

        self._save_data()

    def get_exploration_stats(self) -> Dict:
        """Get exploration statistics."""
        total = len(self.explorations)
        success = len([e for e in self.explorations if e.result == "success"])
        failure = len([e for e in self.explorations if e.result == "failure"])
        partial = len([e for e in self.explorations if e.result == "partial"])

        return {
            "total_explorations": total,
            "successful": success,
            "failed": failure,
            "partial": partial,
            "success_rate": round(success / max(1, total) * 100, 1),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "active_hypotheses": len([h for h in self.hypotheses.values() if h.status == "pending"])
        }


research_agent = ResearchAgent()


def generate_research_hypothesis(focus_area: Optional[str] = None, based_on_alpha: Optional[str] = None) -> Dict:
    """Generate a new research hypothesis."""
    return asdict(research_agent.generate_hypothesis(focus_area, based_on_alpha))


def analyze_failed_alpha_research(alpha_id: str) -> Dict:
    """Analyze a failed alpha and propose improvements."""
    return research_agent.analyze_failed_alpha(alpha_id)


def find_orthogonal_sleeve_research(alpha_id: str) -> List[Dict]:
    """Find orthogonal sleeve for an alpha."""
    return research_agent.find_orthogonal_sleeve(alpha_id)


def detect_research_gaps() -> Dict:
    """Detect unexplored areas."""
    return research_agent.detect_research_gaps()


def maintain_research_diversity(current_alphas: List[str]) -> Dict:
    """Maintain research diversity."""
    return research_agent.maintain_diversity(current_alphas)


def get_agent_stats() -> Dict:
    """Get research agent statistics."""
    return research_agent.get_exploration_stats()


def create_research_task(task_type: str, description: str, parameters: Dict) -> Dict:
    """Create a research task."""
    return asdict(research_agent.create_task(task_type, description, parameters))
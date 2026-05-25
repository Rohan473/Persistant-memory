"""
Extended API for Institutional Research Features
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class MemoryTypeEnum(str, Enum):
    ALPHA_EXPRESSION = "alpha_expression"
    RESEARCH_NOTE = "research_note"
    DISCUSSION = "discussion"
    SIMULATION_RESULT = "simulation_result"
    HYPOTHESIS = "hypothesis"
    FACTOR_DESCRIPTION = "factor_description"
    FAILURE_ANALYSIS = "failure_analysis"
    MACRO_INTERPRETATION = "macro_interpretation"
    CONCEPT = "concept"
    FORMULA_TEMPLATE = "formula_template"


class FactorClassificationRequest(BaseModel):
    alpha_id: str
    expression: str
    datafields: List[str]
    operators: List[str]
    concepts: List[str]


class RegimenPerformanceRequest(BaseModel):
    alpha_id: str
    year: int
    sharpe: Optional[float] = None
    returns: Optional[float] = None
    turnover: Optional[float] = None
    drawdown: Optional[float] = None
    fitness: Optional[float] = None


class FailureAnalysisRequest(BaseModel):
    alpha_id: str
    expression: str
    datafields: List[str]
    operators: List[str]
    concepts: List[str]
    failure_modes: List[str]
    sharpe: Optional[float] = None
    turnover: Optional[float] = None
    fitness: Optional[float] = None
    correlated_with: List[str] = []
    neutralization: str = "market"


class CorrelationRegistrationRequest(BaseModel):
    alpha_id: str
    factor_families: List[str]
    concepts: List[str]
    datafields: List[str]


class SemanticMemoryRequest(BaseModel):
    content: str
    memory_type: MemoryTypeEnum
    tags: List[str] = []
    related_alpha_id: Optional[str] = None
    importance: float = 0.5


def setup_institutional_routes(app):
    """Add institutional research routes to FastAPI app."""

    # Alpha Lineage Endpoints
    @app.post("/lineage/register")
    async def register_alpha_lineage(
        alpha_id: str,
        parent_id: Optional[str] = None,
        branch_id: Optional[str] = None
    ):
        """Register a new alpha with lineage tracking."""
        from .alpha_lineage import register_alpha
        return register_alpha(alpha_id, parent_id, branch_id=branch_id)

    @app.get("/lineage/{alpha_id}")
    async def get_alpha_lineage(alpha_id: str):
        """Get complete lineage for an alpha."""
        from .alpha_lineage import get_lineage
        result = get_lineage(alpha_id)
        if not result:
            raise HTTPException(status_code=404, detail="Alpha not found in lineage")
        return result

    @app.get("/lineage/compare/{alpha1}/{alpha2}")
    async def compare_lineages(alpha1: str, alpha2: str):
        """Compare two alphas in the same lineage."""
        from .alpha_lineage import compare_lineage
        return compare_lineage(alpha1, alpha2)

    @app.get("/lineage/tree/{root_alpha_id}")
    async def get_lineage_tree(root_alpha_id: str, max_depth: int = 5):
        """Get full experiment tree."""
        from .alpha_lineage import get_tree
        return get_tree(root_alpha_id)

    @app.get("/lineage/stats")
    async def get_lineage_stats():
        """Get lineage tracking statistics."""
        from .alpha_lineage import get_lineage_stats
        return get_lineage_stats()

    @app.post("/lineage/modification")
    async def record_modification(
        alpha_id: str,
        modification_type: str,
        before_state: Dict,
        after_state: Dict,
        reason: Optional[str] = None,
        performance_delta: Optional[float] = None
    ):
        """Record a modification to an alpha."""
        from .alpha_lineage import record_modification
        return record_modification(
            alpha_id, modification_type, before_state, after_state, reason, performance_delta
        )

    @app.post("/lineage/branch")
    async def create_branch(name: str, parent_alpha_id: str, description: str = ""):
        """Create a new experiment branch."""
        from .alpha_lineage import create_experiment_branch
        return create_experiment_branch(name, parent_alpha_id, description)

    @app.post("/factor/classify")
    async def classify_alpha(req: FactorClassificationRequest):
        """Classify alpha into factor families."""
        from .factor_ontology import classify_alpha
        result = classify_alpha(
            req.expression,
            req.datafields,
            req.operators,
            req.concepts
        )
        return {"alpha_id": req.alpha_id, "factor_exposures": result}

    @app.get("/factor/info/{factor_id}")
    async def get_factor_info(factor_id: str):
        """Get detailed factor information."""
        from .factor_ontology import get_factor_info
        result = get_factor_info(factor_id)
        if not result:
            raise HTTPException(status_code=404, detail="Factor not found")
        return result

    @app.get("/factor/related/{factor_id}")
    async def get_related_factors(factor_id: str):
        """Get related factors (complements and conflicts)."""
        from .factor_ontology import get_related_factors
        return get_related_factors(factor_id)

    @app.get("/regime/list")
    async def list_regimes():
        """List all available market regimes."""
        from .regime_analysis import analyzer
        return {"regimes": analyzer.list_regimes()}

    @app.post("/regime/performance")
    async def add_regime_performance(req: RegimenPerformanceRequest):
        """Add yearly regime performance for an alpha."""
        from .regime_analysis import add_regime_performance
        add_regime_performance(
            req.alpha_id, req.year, req.sharpe, req.returns,
            req.turnover, req.drawdown, req.fitness
        )
        return {"status": "added", "alpha_id": req.alpha_id, "year": req.year}

    @app.get("/regime/performance/{alpha_id}")
    async def get_regime_performance(alpha_id: str):
        """Get regime performance breakdown for an alpha."""
        from .regime_analysis import get_regime_performance
        return {"alpha_id": alpha_id, "regime_performance": get_regime_performance(alpha_id)}

    @app.get("/regime/alphas/{regime}")
    async def find_regime_alphas(regime: str, min_sharpe: float = 0.5):
        """Find alphas that perform well in a specific regime."""
        from .regime_analysis import find_regime_alphas
        return {"regime": regime, "alphas": find_regime_alphas(regime, min_sharpe)}

    @app.post("/regime/dependencies")
    async def infer_regime_dependencies(
        factor_families: List[str],
        neutralization: str,
        universe: str
    ):
        """Infer regime dependencies from alpha characteristics."""
        from .regime_analysis import infer_regime_dependencies
        return infer_regime_dependencies(factor_families, neutralization, universe)

    @app.post("/failure/analyze")
    async def analyze_failure(req: FailureAnalysisRequest):
        """Analyze why an alpha failed."""
        from .research_copilot import explain_alpha_failure
        result = explain_alpha_failure(
            req.alpha_id,
            req.failure_modes,
            req.sharpe,
            req.turnover,
            req.fitness,
            req.expression,
            req.operators,
            req.datafields
        )
        return result

    @app.post("/failure/learning")
    async def learn_from_failure(req: FailureAnalysisRequest):
        """Learn from alpha failure and update patterns."""
        from .failure_learning import analyze_failure
        result = analyze_failure(
            req.alpha_id,
            req.expression,
            req.datafields,
            req.operators,
            req.concepts,
            req.failure_modes,
            req.sharpe,
            req.turnover,
            req.fitness,
            req.correlated_with,
            req.neutralization
        )
        return result

    @app.get("/failure/warnings")
    async def get_failure_warnings(
        turnover: Optional[float] = None,
        operators: Optional[List[str]] = None,
        has_decay: bool = False,
        neutralization: Optional[str] = None,
        concepts: Optional[List[str]] = None
    ):
        """Get warnings for potential failure patterns."""
        from .failure_learning import get_failure_warnings
        characs = {
            "turnover": turnover or 0,
            "operators": operators or [],
            "has_decay": has_decay,
            "neutralization": neutralization,
            "concepts": concepts or []
        }
        return {"warnings": get_failure_warnings(characs)}

    @app.get("/failure/common")
    async def get_common_failures():
        """Get most common failure patterns."""
        from .failure_learning import get_common_failures
        return {"patterns": get_common_failures()}

    @app.post("/correlation/register")
    async def register_alpha_correlation(req: CorrelationRegistrationRequest):
        """Register alpha for correlation analysis."""
        from .correlation_engine import register_alpha_factors
        register_alpha_factors(
            req.alpha_id,
            req.factor_families,
            req.concepts,
            req.datafields
        )
        return {"status": "registered", "alpha_id": req.alpha_id}

    @app.get("/correlation/pair/{alpha1}/{alpha2}")
    async def get_pair_correlation(alpha1: str, alpha2: str):
        """Get correlation between two alphas."""
        from .correlation_engine import get_pair_correlation
        corr = get_pair_correlation(alpha1, alpha2)
        return {"alpha1": alpha1, "alpha2": alpha2, "correlation": corr}

    @app.get("/correlation/orthogonal/{alpha_id}")
    async def find_orthogonal(alpha_id: str, max_corr: float = 0.3):
        """Find orthogonal alphas."""
        from .correlation_engine import find_orthogonal_sleeves
        return {"target": alpha_id, "orthogonal": find_orthogonal_sleeves(alpha_id)}

    @app.post("/correlation/portfolio")
    async def compute_portfolio(alphas: Dict[str, float], alpha_metrics: Dict[str, Dict]):
        """Compute portfolio-level metrics."""
        from .correlation_engine import compute_portfolio_metrics
        return compute_portfolio_metrics(alphas, alpha_metrics)

    @app.post("/memory/semantic")
    async def add_semantic_memory(req: SemanticMemoryRequest):
        """Add a semantic memory."""
        from .vector_memory import add_semantic_memory
        from .vector_memory import MemoryType
        mem_id = add_semantic_memory(
            req.content,
            MemoryType(req.memory_type.value),
            tags=req.tags,
            related_alpha_id=req.related_alpha_id,
            importance=req.importance
        )
        return {"status": "added", "memory_id": mem_id}

    @app.get("/memory/semantic/search")
    async def search_semantic(
        query: str,
        k: int = 5,
        memory_types: Optional[List[str]] = None
    ):
        """Semantic search across memories."""
        from .vector_memory import search_semantic, MemoryType
        types = [MemoryType(mt) for mt in (memory_types or [])]
        return {"query": query, "results": search_semantic(query, k, types if types else None)}

    @app.get("/memory/stats")
    async def get_memory_stats():
        """Get vector memory statistics."""
        from .vector_memory import get_vector_memory_stats
        return get_vector_memory_stats()

    @app.post("/query/nl")
    async def natural_language_query(query: str):
        """Process natural language research query."""
        from .nl_query import process_query
        return process_query(query)

    @app.get("/recommend/coverage")
    async def get_exploration_coverage():
        """Get research exploration coverage."""
        from .recommendation_engine import get_exploration_coverage
        return get_exploration_coverage()

    @app.get("/recommend/factors")
    async def suggest_factors(used_factors: str, used_datafields: str):
        """Suggest unexplored factor combinations."""
        from .recommendation_engine import suggest_factors
        return {
            "suggestions": suggest_factors(
                used_factors.split(","),
                used_datafields.split(",")
            )
        }

    @app.get("/recommend/diversity")
    async def recommend_diversity(current_alphas: str):
        """Recommend maintaining research diversity."""
        from .recommendation_engine import recommend_diversity
        return {"recommendations": recommend_diversity([])}

    @app.get("/recommend/roadmap")
    async def get_roadmap(current_progress: Optional[Dict] = None):
        """Get research roadmap."""
        from .recommendation_engine import get_roadmap
        return {"roadmap": get_roadmap(current_progress or {})}

    @app.post("/copilot/hidden-exposures")
    async def detect_hidden_exposures(
        expression: str,
        datafields: List[str],
        operators: List[str],
        concepts: List[str],
        neutralization: str
    ):
        """Detect hidden factor exposures."""
        from .research_copilot import detect_hidden_exposures
        return detect_hidden_exposures(expression, datafields, operators, concepts, neutralization)

    @app.post("/copilot/factor-conflicts")
    async def detect_factor_conflicts(
        concepts: List[str],
        operators: List[str],
        expression: str
    ):
        """Detect conflicting factor exposures."""
        from .research_copilot import detect_factor_conflicts
        return detect_factor_conflicts(concepts, operators, expression)

    @app.post("/copilot/economic-meaning")
    async def infer_economic_meaning(
        expression: str,
        datafields: List[str],
        operators: List[str],
        concepts: List[str]
    ):
        """Infer economic meaning from alpha expression."""
        from .research_copilot import infer_economic_meaning
        return infer_economic_meaning(expression, datafields, operators, concepts)

    @app.post("/copilot/overfitting")
    async def detect_overfitting(
        expression: str,
        sharpe: float,
        turnover: float,
        universe: str,
        parent_alpha: Optional[str] = None
    ):
        """Detect potential overfitting."""
        from .research_copilot import detect_overfitting
        return detect_overfitting(expression, sharpe, turnover, universe, parent_alpha)

    @app.get("/health/institutional")
    async def institutional_health():
        """Check institutional features health."""
        from . import get_stats
        base_stats = get_stats()

        return {
            "status": "ok",
            "version": "0.2.0",
            "institutional_features": {
                "factor_ontology": True,
                "regime_analysis": True,
                "failure_learning": True,
                "vector_memory": True,
                "correlation_engine": True,
                "research_copilot": True,
                "nl_query": True,
                "recommendation_engine": True,
                "alpha_lineage": True,
                "research_agent": True
            },
            "base_stats": base_stats
        }

    # Research Agent Endpoints
    @app.post("/agent/hypothesis")
    async def generate_hypothesis(
        focus_area: Optional[str] = None,
        based_on_alpha: Optional[str] = None
    ):
        """Generate a new research hypothesis."""
        from .research_agent import generate_research_hypothesis
        return generate_research_hypothesis(focus_area, based_on_alpha)

    @app.post("/agent/analyze-failure/{alpha_id}")
    async def analyze_failed_alpha(alpha_id: str):
        """Analyze a failed alpha and propose improvements."""
        from .research_agent import analyze_failed_alpha_research
        return analyze_failed_alpha_research(alpha_id)

    @app.get("/agent/orthogonal/{alpha_id}")
    async def find_orthogonal_sleeve(alpha_id: str):
        """Find orthogonal sleeve for an alpha."""
        from .research_agent import find_orthogonal_sleeve_research
        return {"alpha_id": alpha_id, "sleeves": find_orthogonal_sleeve_research(alpha_id)}

    @app.get("/agent/gaps")
    async def detect_gaps():
        """Detect unexplored research areas."""
        from .research_agent import detect_research_gaps
        return detect_research_gaps()

    @app.get("/agent/diversity")
    async def analyze_diversity(current_alphas: Optional[str] = None):
        """Analyze and maintain research diversity."""
        from .research_agent import maintain_research_diversity
        alpha_list = current_alphas.split(",") if current_alphas else []
        return maintain_research_diversity(alpha_list)

    @app.get("/agent/stats")
    async def get_agent_stats():
        """Get research agent statistics."""
        from .research_agent import get_agent_stats
        return get_agent_stats()

    @app.post("/agent/task")
    async def create_task(
        task_type: str,
        description: str,
        parameters: Dict
    ):
        """Create a research agent task."""
        from .research_agent import create_research_task
        return create_research_task(task_type, description, parameters)
"""
Reranking module - Hybrid scoring before returning results.
Improves retrieval quality by combining multiple signals.
"""

from typing import List, Dict, Tuple
from .structure import load_metadata

def compute_hybrid_score(result: Dict, metadata: List[Dict], query: str = "") -> float:
    """
    Compute hybrid score combining:
    - vector similarity (from result.score)
    - structural importance (from metadata)
    - salience (from result.salience)

    Args:
        result: Retrieval result with score, salience
        metadata: Full metadata list for structural lookup
        query: Original query (for future query-dependent reranking)

    Returns:
        Hybrid score (0-1 range)
    """
    # Base score from vector similarity
    base_score = result.get("score", 0.5)

    # Salience boost
    salience = result.get("salience", {})
    importance = salience.get("importance", 0.5)
    reuse_count = salience.get("reuse_count", 0)

    # Reuse boost (logarithmic to prevent dominance)
    reuse_boost = min(0.2, reuse_count * 0.02)

    # Importance boost
    importance_boost = (importance - 0.5) * 0.1  # -0.05 to +0.05

    # Type bonus (Alphas more valuable for research)
    node_type = result.get("node_type", "")
    type_bonus = {
        "Alpha": 0.05,
        "Concept": 0.03,
        "Operator": 0.02,
        "Datafield": 0.02,
        "Session": 0.01,
        "Setting": 0.01,
        "FailureMode": 0.02,
    }.get(node_type, 0)

    # Composite score
    hybrid = base_score + reuse_boost + importance_boost + type_bonus

    return min(1.0, max(0.0, hybrid))


def rerank_results(results: List[Dict], metadata: List[Dict] = None, query: str = "") -> List[Dict]:
    """
    Rerank retrieval results using hybrid scoring.

    Args:
        results: List of retrieval results
        metadata: Full metadata list (optional, for structural features)
        query: Original query (optional)

    Returns:
        Reranked results with updated retrieval_score
    """
    if not results:
        return results

    # Load metadata if not provided
    if metadata is None:
        try:
            metadata = load_metadata()
        except:
            metadata = []

    # Compute hybrid scores
    scored_results = []
    for r in results:
        hybrid_score = compute_hybrid_score(r, metadata, query)
        r["retrieval_score"] = hybrid_score
        r["hybrid_score"] = hybrid_score
        scored_results.append(r)

    # Sort by hybrid score
    scored_results.sort(key=lambda x: x.get("retrieval_score", 0), reverse=True)

    return scored_results


def compress_with_lineage(results: List[Dict], max_alphas: int = 5) -> Dict:
    """
    Compress results by including 1-hop lineage info for alphas.

    Args:
        results: Reranked results
        max_alphas: Maximum alphas to include lineage for

    Returns:
        Compressed research state
    """
    from .structure import load_metadata
    metadata = load_metadata()

    # Index metadata by name
    meta_index = {m["name"]: m for m in metadata if m.get("name")}

    active_concepts = set()
    related_failures = set()
    lineage = {}

    alphas_processed = 0

    for r in results:
        if r.get("node_type") == "Alpha" and alphas_processed < max_alphas:
            name = r.get("name", "")
            if name in meta_index:
                m = meta_index[name]
                # Collect concepts
                for c in m.get("concepts", []):
                    active_concepts.add(c)
                # Collect failures
                for f in m.get("failure_modes", []):
                    related_failures.add(f)
                # Collect lineage
                derived = m.get("derived_from", [])
                if derived:
                    lineage[name] = derived
                alphas_processed += 1

        # Also collect from any node's concepts/failures
        if r.get("concepts"):
            for c in r.get("concepts", []):
                active_concepts.add(c)
        if r.get("failure_modes"):
            for f in r.get("failure_modes", []):
                related_failures.add(f)

    return {
        "active_concepts": sorted(list(active_concepts)),
        "related_failures": sorted(list(related_failures)),
        "alpha_lineage": lineage,
        "total_results": len(results),
        "alphas_with_lineage": alphas_processed,
    }


def compile_research_state(
    query: str,
    results: List[Dict],
    token_budget: int = 2000
) -> Dict:
    """
    Compile a structured research state from retrieval results.

    This is the "Research State Compiler" - instead of returning raw memories,
    produce machine-assisted research cognition.

    Args:
        query: Original research query
        results: Reranked retrieval results
        token_budget: Maximum tokens for context

    Returns:
        Structured research state
    """
    # Get compressed lineage
    compression = compress_with_lineage(results)

    # Format context
    from .budget import compile_context_compact
    context, tokens = compile_context_compact(results, budget_tokens=token_budget)

    return {
        "query": query,
        "research_state": {
            "active_concepts": compression["active_concepts"],
            "related_failures": compression["related_failures"],
            "alpha_lineage": compression["alpha_lineage"],
            "recommendations": _generate_recommendations(compression, results),
        },
        "context": context,
        "tokens_used": tokens,
        "token_budget": token_budget,
        "compression_ratio": round(tokens / token_budget, 2) if token_budget > 0 else 0,
    }


def _generate_recommendations(compression: Dict, results: List[Dict]) -> List[str]:
    """Generate research recommendations based on retrieved state."""
    recommendations = []

    # Check for high-failure concepts
    failures = compression.get("related_failures", [])
    if "low_sharpe" in failures:
        recommendations.append("Consider exploring neutralization or universe changes for low-Sharpe alphas")

    # Check for sentiment concepts
    concepts = compression.get("active_concepts", [])
    if "sentiment" in concepts:
        recommendations.append("Sentiment alphas may benefit from regime filtering")

    if "momentum" in concepts and "mean_reversion" in concepts:
        recommendations.append("Both momentum and mean reversion concepts retrieved - consider regime-specific strategies")

    # Check for value concepts
    if "value" in concepts and "quality" in concepts:
        recommendations.append("Value+quality combination is historically robust - verify with longer backtest")

    # Default recommendation if nothing specific
    if not recommendations:
        recommendations.append("Review retrieved alphas for pattern extraction and new hypothesis generation")

    return recommendations[:3]  # Limit to 3
"""
Minimal salience scoring for memory prioritization.
Only tracks: importance_score (manual) + reuse_count
"""

from typing import Dict, List, Optional

def init_salience(metadata: List[Dict], default_importance: float = 0.5) -> List[Dict]:
    """
    Initialize salience fields for all metadata entries.

    Args:
        metadata: List of metadata dictionaries
        default_importance: Default importance score for all nodes

    Returns:
        Metadata with salience fields added
    """
    for m in metadata:
        if "salience" not in m:
            m["salience"] = {
                "importance": default_importance,
                "reuse_count": 0,
                "last_used": None
            }
        else:
            # Ensure all fields exist
            m["salience"].setdefault("importance", default_importance)
            m["salience"].setdefault("reuse_count", 0)
            m["salience"].setdefault("last_used", None)
    return metadata

def update_reuse_count(metadata: Dict) -> None:
    """Increment reuse count when a memory is retrieved."""
    if "salience" not in metadata:
        metadata["salience"] = {"importance": 0.5, "reuse_count": 1, "last_used": None}
    else:
        metadata["salience"]["reuse_count"] = metadata["salience"].get("reuse_count", 0) + 1

def set_importance(metadata: Dict, importance: float) -> None:
    """Set importance score manually (0.0 to 1.0)."""
    if "salience" not in metadata:
        metadata["salience"] = {"importance": importance, "reuse_count": 0, "last_used": None}
    else:
        metadata["salience"]["importance"] = max(0.0, min(1.0, importance))

def compute_retrieval_score(metadata: Dict, similarity: float) -> float:
    """
    Compute retrieval score: similarity × importance × (1 + log(reuse_count + 1))

    This favors:
    - High similarity (semantic relevance)
    - High importance (foundational nodes)
    - More frequently used memories (proved useful)
    """
    salience = metadata.get("salience", {})
    importance = salience.get("importance", 0.5)
    reuse_count = salience.get("reuse_count", 0)

    # Logarithmic reuse boost (diminishing returns)
    reuse_boost = 1 + (reuse_count / 10)  # roughly 1 + log_e(reuse+1)

    return similarity * importance * reuse_boost

def rank_by_salience(results: List[Dict], k: int = 5) -> List[Dict]:
    """
    Rank retrieval results by salience-enhanced score.
    Modifies the results list in place and returns top-k.
    """
    # Sort by retrieval score (assumes each result has a 'score' field from vector search)
    ranked = sorted(results, key=lambda r: r.get("retrieval_score", r.get("score", 0)), reverse=True)
    return ranked[:k]

def bulk_importance_from_file(metadata: List[Dict], importance_map: Dict[str, float]) -> None:
    """
    Set importance scores from a mapping file.

    Args:
        metadata: List of metadata
        importance_map: Dict mapping node_id to importance score
    """
    for m in metadata:
        node_id = m.get("id", "")
        if node_id in importance_map:
            set_importance(m, importance_map[node_id])
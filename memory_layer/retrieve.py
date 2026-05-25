"""
Hybrid retrieval: vector + metadata filtering + shallow graph expansion
"""

from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .config import config, get_qdrant_url
from .embed import embed_text
from .salience import compute_retrieval_score

_client = None

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=get_qdrant_url())
    return _client

def retrieve(
    query: str,
    k: int = 5,
    node_types: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Semantic retrieval using vector similarity with optional filtering.

    Args:
        query: Natural language query
        k: Number of results to return
        node_types: Filter to specific node types (e.g., ["Alpha"])
        filters: Additional filters (operators, concepts, datafields, etc.)

    Returns:
        List of retrieved memory records
    """
    # Generate query embedding
    query_vector = embed_text(query).tolist()

    # Build filter
    qdrant_filter = _build_filter(node_types, filters)

    # Search
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        limit=k * 2,  # Get more for re-ranking
        query_filter=qdrant_filter,
        with_payload=True
    )

    # Process results - response.points contains the ScoredPoint list
    memories = []
    for r in response.points:
        payload = r.payload
        # Compute salience-enhanced score
        retrieval_score = compute_retrieval_score(payload, r.score)

        memory = {
            "id": payload.get("id"),
            "node_type": payload.get("node_type"),
            "name": payload.get("name"),
            "structured_summary": payload.get("structured_summary"),
            "relationships": payload.get("relationships", []),
            "score": r.score,
            "retrieval_score": retrieval_score,
            "salience": payload.get("salience", {}),
        }

        # Add type-specific fields
        if payload.get("node_type") == "Alpha":
            memory.update({
                "operators": payload.get("operators", []),
                "concepts": payload.get("concepts", []),
                "datafields": payload.get("datafields", []),
                "failure_modes": payload.get("failure_modes", []),
                "rating": payload.get("rating"),
                "sharpe": payload.get("sharpe"),
            })

        memories.append(memory)

    return memories

def _build_filter(node_types: Optional[List[str]], filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    """Build Qdrant filter from node types and metadata filters."""
    conditions = []

    if node_types:
        conditions.append(FieldCondition(
            key="node_type",
            match=MatchValue(value=node_types[0])
        ))

    if filters:
        # Filter by specific operators
        if "operators" in filters and filters["operators"]:
            # This is complex - we'd need to check if any of the operators are in the list
            # For now, use simple equality or search
            pass

        # Filter by concepts
        if "concepts" in filters and filters["concepts"]:
            pass

        # Filter by datafields
        if "datafields" in filters and filters["datafields"]:
            pass

        # Filter by failure modes
        if "failure_modes" in filters and filters["failure_modes"]:
            pass

        # Filter by universe (from structured summary)
        if "universe" in filters:
            # Search in structured_summary
            pass

        # Filter by rating
        if "rating" in filters:
            conditions.append(FieldCondition(
                key="rating",
                match=MatchValue(value=filters["rating"])
            ))

    if conditions:
        return Filter(must=conditions)
    return None

def retrieve_hybrid(
    query: str,
    k: int = 5,
    operators: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
    node_types: Optional[List[str]] = None,
    include_structured: bool = True,
    include_raw: bool = False
) -> List[Dict]:
    """
    Hybrid retrieval combining vector search with metadata filtering.

    This is the main retrieval function for agents.
    """
    # Build filters
    filters = {}
    if operators:
        filters["operators"] = operators
    if concepts:
        filters["concepts"] = concepts

    # Initial vector search (broader)
    results = retrieve(query, k=k * 2, node_types=node_types, filters=filters)

    # Apply additional filtering if needed
    if operators or concepts:
        filtered = []
        for r in results:
            if operators:
                result_ops = set(r.get("operators", []))
                if not result_ops.intersection(set(operators)):
                    continue
            if concepts:
                result_concepts = set(r.get("concepts", []))
                if not result_concepts.intersection(set(concepts)):
                    continue
            filtered.append(r)
        results = filtered

    # Rank by salience
    from .salience import rank_by_salience
    results = rank_by_salience(results, k=k)

    return results

def format_context(
    memories: List[Dict],
    include_structured: bool = True,
    include_raw: bool = False,
    max_memories: int = 5
) -> str:
    """
    Format retrieved memories into a context string for LLM injection.

    Args:
        memories: List of retrieved memory records
        include_structured: Include structured metadata
        include_raw: Include raw graph relationships (requires resolving from graph)
        max_memories: Maximum number of memories to include

    Returns:
        Formatted context string
    """
    if not memories:
        return "No relevant memories found."

    lines = ["## Retrieved Memories\n"]
    lines.append(f"Total retrieved: {len(memories)}\n")

    for i, mem in enumerate(memories[:max_memories]):
        lines.append(f"\n### {i+1}. {mem.get('name')} ({mem.get('node_type')})")
        lines.append(f"Relevance score: {mem.get('score', 0):.3f}")

        if include_structured:
            lines.append(f"\n**Structured Summary:**")
            lines.append(f"```\n{mem.get('structured_summary', 'N/A')}\n```")

        # Type-specific details
        if mem.get("node_type") == "Alpha":
            ops = mem.get("operators", [])
            concepts = mem.get("concepts", [])
            failures = mem.get("failure_modes", [])
            rating = mem.get("rating")
            sharpe = mem.get("sharpe")

            if ops:
                lines.append(f"\n**Operators:** {', '.join(ops)}")
            if concepts:
                lines.append(f"**Concepts:** {', '.join(concepts)}")
            if failures:
                lines.append(f"**Failures:** {', '.join(failures)}")
            if rating:
                lines.append(f"**Rating:** {rating}")
            if sharpe:
                lines.append(f"**Sharpe:** {sharpe}")

        relationships = mem.get("relationships", [])
        if relationships:
            lines.append(f"\n**Relationships:**")
            for rel in relationships[:5]:
                lines.append(f"  - {rel}")

        salience = mem.get("salience", {})
        if salience:
            importance = salience.get("importance", 0)
            reuse = salience.get("reuse_count", 0)
            lines.append(f"\n*Salience: importance={importance:.2f}, reuse={reuse}*")

    return "\n".join(lines)

def resolve_full_neighborhood(node_id: str) -> Dict:
    """
    Resolve full graph neighborhood from graph.json.
    This is called when raw context is needed.
    """
    from .structure import load_graph, build_edge_lookup

    graph = load_graph()
    edges = build_edge_lookup(graph)

    # Find the node
    node = None
    for n in graph.get("nodes", []):
        if n.get("id") == node_id:
            node = n
            break

    if not node:
        return {}

    # Get all connected nodes
    outgoing = edges.get(node_id, [])

    # Resolve targets (basic - just IDs for now)
    neighbors = {
        "outgoing": [{"target": e["target"], "relation": e["relation"]} for e in outgoing]
    }

    return {
        "node": node,
        "neighbors": neighbors
    }
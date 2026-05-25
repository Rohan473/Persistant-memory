"""
Deterministic structured metadata extraction from graph.
This is the core of the memory layer - converts graph nodes to structured metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from .config import config

# Node type to structured field mapping
NODE_TYPE_FIELDS = {
    "Alpha": ["name", "expression", "sharpe", "fitness", "rating", "is_ceiling"],
    "Concept": ["name", "hypothesis"],
    "Datafield": ["name"],
    "Operator": ["name"],
    "Setting": ["name"],
    "FailureMode": ["name"],
    "Session": ["name", "timestamp"],
}

def load_graph() -> Dict[str, Any]:
    """Load the graph from graph.json."""
    with open(config.graph_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_edge_lookup(graph: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Build a lookup of edges grouped by source node."""
    edges_by_source = defaultdict(list)
    for edge in graph.get("links", []):
        source = edge.get("source")
        if isinstance(source, dict):
            source = source.get("id", str(source))
        target = edge.get("target")
        if isinstance(target, dict):
            target = target.get("id", str(target))
        edges_by_source[source].append({
            "target": target,
            "relation": edge.get("relation", "")
        })
    return edges_by_source

def extract_node_metadata(node: Dict, edges: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Extract structured metadata from a single node."""
    node_type = node.get("node_type", "")
    node_id = node.get("id", "")
    node_name = node.get("name", "")

    # Get outgoing edges (relationships)
    outgoing = edges.get(node_id, [])

    # Extract based on node type
    metadata = {
        "id": node_id,
        "node_type": node_type,
        "name": node_name,
    }

    if node_type == "Alpha":
        metadata.update({
            "expression": node.get("expression", ""),
            "sharpe": node.get("sharpe"),
            "fitness": node.get("fitness"),
            "rating": node.get("rating"),
            "is_ceiling": node.get("is_ceiling", False),
            "ceiling_blocked_by": node.get("ceiling_blocked_by"),
        })
        # Extract relationships
        metadata["operators"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "APPLIES"]
        metadata["concepts"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "IMPLEMENTS"]
        metadata["datafields"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "USES"]
        metadata["settings"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "TESTED_UNDER"]
        metadata["failure_modes"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "FAILED_BY"]
        metadata["correlated_with"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "CORRELATED_WITH"]
        metadata["derived_from"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "DERIVED_FROM"]

    elif node_type == "Concept":
        metadata["hypothesis"] = node.get("hypothesis", "")
        # Related alphas
        related = [e["target"] for e in outgoing]
        metadata["related_alphas"] = [a.split("::")[1] for a in related if a.startswith("Alpha::")]
        metadata["datafields_used"] = [e["target"].split("::")[1] for e in outgoing if e["target"].startswith("Datafield::")]
        metadata["operators_used"] = [e["target"].split("::")[1] for e in outgoing if e["target"].startswith("Operator::")]

    elif node_type == "Datafield":
        metadata["alphas_used_in"] = [e["target"].split("::")[1] for e in outgoing if e["target"].startswith("Alpha::")]

    elif node_type == "Operator":
        metadata["alphas_used_in"] = [e["target"].split("::")[1] for e in outgoing if e["target"].startswith("Alpha::")]

    elif node_type == "Setting":
        metadata["alphas_tested"] = [e["target"].split("::")[1] for e in outgoing if e["target"].startswith("Alpha::")]

    elif node_type == "FailureMode":
        metadata["failed_alphas"] = [e["target"].split("::")[1] for e in outgoing if e["target"].startswith("Alpha::")]

    elif node_type == "Session":
        metadata["timestamp"] = node.get("timestamp", "")
        metadata["alphas_produced"] = [e["target"].split("::")[1] for e in outgoing if e["relation"] == "PRODUCED"]

    return metadata

def generate_structured_summary(metadata: Dict) -> str:
    """
    Generate a deterministic structured summary for embedding.
    Format: NODE_TYPE:NAME|FIELD:VALUE|...
    """
    node_type = metadata.get("node_type", "")
    name = metadata.get("name", "")

    parts = [f"{node_type}:{name}"]

    if node_type == "Alpha":
        ops = metadata.get("operators", [])
        concepts = metadata.get("concepts", [])
        dfs = metadata.get("datafields", [])
        failures = metadata.get("failure_modes", [])
        derived = metadata.get("derived_from", [])
        rating = metadata.get("rating", "")

        if ops:
            parts.append(f"OPS:{','.join(ops[:5])}")
        if concepts:
            parts.append(f"CONCEPTS:{','.join(concepts[:5])}")
        if dfs:
            parts.append(f"DF:{','.join(dfs[:5])}")
        if failures:
            parts.append(f"FAIL:{','.join(failures[:3])}")
        if derived:
            parts.append(f"DERIVED:{derived[0]}")
        if rating:
            parts.append(f"RATING:{rating}")

    elif node_type == "Concept":
        related = metadata.get("related_alphas", [])
        if related:
            parts.append(f"ALPHAS:{','.join(related[:5])}")
        hypothesis = metadata.get("hypothesis", "")
        if hypothesis:
            parts.append(f"HYP:{hypothesis[:50]}")

    elif node_type == "Datafield":
        alphas = metadata.get("alphas_used_in", [])
        if alphas:
            parts.append(f"USED_IN:{','.join(alphas[:5])}")

    elif node_type == "Operator":
        alphas = metadata.get("alphas_used_in", [])
        if alphas:
            parts.append(f"APPLIED_TO:{','.join(alphas[:5])}")

    elif node_type == "FailureMode":
        failed = metadata.get("failed_alphas", [])
        if failed:
            parts.append(f"FAILED:{','.join(failed[:5])}")

    elif node_type == "Session":
        produced = metadata.get("alphas_produced", [])
        if produced:
            parts.append(f"PRODUCED:{len(produced)}_alphas")

    return "|".join(parts)

def extract_all_metadata() -> List[Dict[str, Any]]:
    """
    Extract structured metadata from all nodes in the graph.

    Returns:
        List of metadata dictionaries, one per node
    """
    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = build_edge_lookup(graph)

    all_metadata = []
    for node in nodes:
        metadata = extract_node_metadata(node, edges)
        metadata["structured_summary"] = generate_structured_summary(metadata)
        all_metadata.append(metadata)

    return all_metadata

def save_metadata(metadata: List[Dict], output_path: Optional[Path] = None) -> None:
    """Save metadata to JSON file."""
    if output_path is None:
        output_path = config.metadata_output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_metadata(path: Optional[Path] = None) -> List[Dict]:
    """Load metadata from JSON file."""
    if path is None:
        path = config.metadata_output
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
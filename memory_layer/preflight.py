"""
Pre-flight context for a candidate alpha expression.

Before submitting a simulation, look up similar prior attempts in the
NetworkX graph so the user can decide whether to skip ("already tried that")
or proceed with awareness of what failed/worked before.

Similarity score combines:
  - datafield overlap (Jaccard)
  - operator overlap (Jaccard)
  - shared parent / shared concept
  - exact normalized-expression match (caps score at 1.0)
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

BASE = Path(__file__).resolve().parent.parent
GRAPH_PATH = BASE / "graph" / "graph.gpickle"


@dataclass
class SimilarAttempt:
    alpha_id: str
    sharpe: Optional[float]
    status: str
    expression: str
    score: float
    shared_datafields: List[str]
    shared_operators: List[str]
    failure_modes: List[str]
    note: str = ""


def _norm(expr: str) -> str:
    expr = re.sub(r"#[^\n]*", "", expr or "")
    return re.sub(r"\s+", " ", expr).strip()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def find_similar_attempts(
    G,
    expression: str,
    datafields: List[str],
    operators: List[str],
    *,
    concepts: Optional[List[str]] = None,
    parent: Optional[str] = None,
    top_n: int = 5,
) -> List[SimilarAttempt]:
    """Return top-N alphas similar to a candidate. Higher score = more similar."""
    if G is None:
        return []
    df_set = {d.lower() for d in (datafields or []) if d}
    op_set = {o.lower() for o in (operators or []) if o}
    concept_set = {c.lower() for c in (concepts or []) if c}
    target_norm = _norm(expression)

    out: List[SimilarAttempt] = []
    for nid in G.nodes:
        d = G.nodes[nid]
        if d.get("node_type") != "Alpha":
            continue
        # Pull this alpha's outgoing edges to score
        alpha_dfs, alpha_ops, alpha_concepts = set(), set(), set()
        alpha_parent = None
        for _, tgt, ed in G.out_edges(nid, data=True):
            tdata = G.nodes[tgt]
            rel = ed.get("relation")
            tname = (tdata.get("name") or "").lower()
            if rel == "USES" and tdata.get("node_type") == "Datafield":
                alpha_dfs.add(tname)
            elif rel == "APPLIES" and tdata.get("node_type") == "Operator":
                alpha_ops.add(tname)
            elif rel == "IMPLEMENTS" and tdata.get("node_type") == "Concept":
                alpha_concepts.add(tname)
            elif rel == "DERIVED_FROM":
                alpha_parent = tdata.get("name")

        score = 0.0
        score += 0.5 * _jaccard(df_set, alpha_dfs)
        score += 0.3 * _jaccard(op_set, alpha_ops)
        score += 0.15 * _jaccard(concept_set, alpha_concepts) if concept_set else 0.0
        if parent and parent == alpha_parent:
            score += 0.05
        if score == 0:
            continue

        # Exact-expression bonus
        note = ""
        existing_norm = _norm(d.get("expression") or "")
        if existing_norm and existing_norm == target_norm:
            score = 1.0
            note = "EXACT MATCH"

        # Pull failure_modes via FAILED_BY edges
        fm = []
        for _, tgt, ed in G.out_edges(nid, data=True):
            if ed.get("relation") == "FAILED_BY":
                fm.append(G.nodes[tgt].get("name") or "")

        out.append(SimilarAttempt(
            alpha_id=d.get("name") or nid.split("::", 1)[-1],
            sharpe=d.get("sharpe"),
            status=d.get("status") or "?",
            expression=(d.get("expression") or "")[:90],
            score=round(score, 3),
            shared_datafields=sorted(df_set & alpha_dfs),
            shared_operators=sorted(op_set & alpha_ops),
            failure_modes=sorted(set(fm)),
            note=note,
        ))

    out.sort(key=lambda s: s.score, reverse=True)
    return out[:top_n]


def load_graph():
    if not GRAPH_PATH.exists():
        return None
    try:
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def format_report(attempts: List[SimilarAttempt]) -> str:
    """Render a short text report suitable for printing before a submission."""
    if not attempts:
        return "Pre-flight: no similar prior attempts found in the graph.\n"
    lines = [f"Pre-flight context ({len(attempts)} related attempt(s)):"]
    for s in attempts:
        sh = f"{s.sharpe:+.2f}" if isinstance(s.sharpe, (int, float)) else "n/a"
        flag = f" [{s.note}]" if s.note else ""
        lines.append(
            f"  {s.alpha_id:<14} score={s.score:.2f}  Sharpe={sh}  "
            f"status={s.status}{flag}"
        )
        if s.shared_datafields:
            lines.append(f"    shared datafields: {', '.join(s.shared_datafields)}")
        if s.shared_operators:
            lines.append(f"    shared operators:  {', '.join(s.shared_operators)}")
        if s.failure_modes:
            lines.append(f"    prior failures:    {', '.join(s.failure_modes)}")
        if s.expression:
            lines.append(f"    expression: {s.expression}")
    return "\n".join(lines) + "\n"

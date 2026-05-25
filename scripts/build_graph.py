"""
Phase 4: Build the NetworkX knowledge graph from all alpha markdown files.
Usage: python scripts/build_graph.py
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import csv
import pickle
import frontmatter
import networkx as nx

from collections import defaultdict, Counter

def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    _ml = BASE / "memory_layer"
    reg_mod = _load_module("memory_layer.regime_analysis", _ml / "regime_analysis.py")
    sys.modules.setdefault("memory_layer", type(sys)("memory_layer"))
    sys.modules["memory_layer"].regime_analysis = reg_mod
    yearly_performance = _load_module("memory_layer.yearly_performance", _ml / "yearly_performance.py")
    YEARLY_PARSER_AVAILABLE = True
except Exception as _e:
    print(f"  WARN: yearly performance parser unavailable: {_e}")
    YEARLY_PARSER_AVAILABLE = False

BASE = Path(__file__).resolve().parent.parent
NODES_DIR = BASE / "private" / "nodes"
ALPHAS_DIR = NODES_DIR / "alphas"
CONCEPTS_DIR = NODES_DIR / "concepts"
DATAFIELDS_DIR = NODES_DIR / "datafields"
OPERATORS_DIR = NODES_DIR / "operators"
SETTINGS_DIR = NODES_DIR / "settings"
FAILURE_MODES_DIR = NODES_DIR / "failure_modes"
SESSIONS_DIR = NODES_DIR / "sessions"
GRAPH_DIR = BASE / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def safe_list(val):
    if not val:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v and str(v) != "null"]
    if isinstance(val, str) and val != "null":
        return [val]
    return []


# WQ Brain IS cutoffs (standard Challenge competition values)
CUTOFFS = {
    "sharpe":      1.25,
    "fitness":     1.00,
    "turnover_hi": 70.0,   # upper bound (%)
    "turnover_lo": 1.0,    # lower bound (%)
}

def ceiling_analysis(meta, sharpe, fitness, turnover):
    """For Needs-Improvement alphas with Sharpe >= 1.5, identify what blocks them."""
    if not sharpe or sharpe < 1.5:
        return None
    status = str(meta.get("status") or "")
    if status not in ("rejected", "iterating"):
        return None

    blockers, fixes = [], []
    fm = safe_list(meta.get("failure_modes"))

    if fitness is not None and fitness < CUTOFFS["fitness"]:
        gap = CUTOFFS["fitness"] - fitness
        blockers.append(f"Fitness {fitness:.2f} < 1.0 cutoff (gap {gap:+.2f})")
    if turnover is not None and turnover > CUTOFFS["turnover_hi"]:
        excess = turnover - CUTOFFS["turnover_hi"]
        blockers.append(f"Turnover {turnover:.1f}% > 70% cutoff (excess {excess:+.1f}pp)")
        fixes.append(f"Apply ts_decay_linear(..., {max(5, int(excess/6))}) or sim Decay setting")
    if "correlated" in fm:
        blockers.append("Self-correlation > 0.7")
        fixes.append("Use sim Decay setting (not ts_decay_linear in formula)")
    if "sub_universe_failure" in fm:
        blockers.append("Sub-universe Sharpe below cutoff")
        fixes.append("Try smaller universe (TOP1000 or TOP500)")
    if "sector_bias" in fm:
        blockers.append("Sector bias contaminating signal")
        fixes.append("Add subindustry or industry neutralization")

    if not blockers:
        return None

    return {
        "is_ceiling": True,
        "sharpe_excess": round(sharpe - CUTOFFS["sharpe"], 2),
        "blocked_by": " | ".join(blockers),
        "unblock_try": " | ".join(fixes) if fixes else "Iterate on the blocker above",
    }


def safe_val(val):
    if val is None or val == "null" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def node_id(ntype, name):
    return f"{ntype}::{name}"


def load_entity_nodes(G, directory, ntype):
    count = 0
    for md_file in sorted(directory.glob("*.md")):
        if md_file.name.startswith("_"):
            continue
        try:
            post = frontmatter.load(str(md_file))
            name = post.metadata.get("name", md_file.stem)
            nid = node_id(ntype, name)
            G.add_node(nid, node_type=ntype, name=name, file=str(md_file))
            count += 1
        except Exception as e:
            print(f"  WARN: could not parse {md_file.name}: {e}")
    return count


def _load_brain_catalogue():
    """Best-effort load of the WQ Brain catalogue cache. Returns None if absent."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "brain_catalogue", BASE / "memory_layer" / "brain_catalogue.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        snap = mod.CatalogueSnapshot.load()
        return mod, snap
    except Exception as e:
        print(f"  WARN: brain catalogue not loaded: {e}")
        return None, None


def _add_catalogue_nodes(G, cat_mod, snap):
    """
    Add operator + datafield nodes from the catalogue snapshot.
    Existing nodes (markdown-sourced) gain metadata; new nodes get availability='available'.
    The alpha loop later overrides availability='used' for nodes referenced by any alpha.
    """
    if not snap or not cat_mod:
        return 0, 0

    op_added = 0
    for op in snap.operators:
        name = op.get("name")
        if not name:
            continue
        nid = node_id("Operator", name)
        if nid not in G:
            G.add_node(nid, node_type="Operator", name=name, file="catalogue")
            op_added += 1
        G.nodes[nid].update({
            "availability": "available",
            "category": op.get("category"),
            "scope": op.get("scope"),
            "definition": op.get("definition"),
            "description": op.get("description"),
            "level": op.get("level"),
        })

    # Datafields: prefer TOP3000 as the canonical snapshot for global metadata.
    canonical_key = snap.settings_key("USA", 1, "TOP3000")
    seen_in_universes = defaultdict(list)
    canonical_rows = snap.datafields.get(canonical_key, [])
    other_rows = []
    for key, rows in snap.datafields.items():
        if key == canonical_key:
            continue
        for row in rows:
            other_rows.append((key, row))
            if row.get("id"):
                seen_in_universes[row["id"]].append(key)

    for row in canonical_rows:
        if row.get("id"):
            seen_in_universes[row["id"]].append(canonical_key)

    df_added = 0
    all_rows = [(canonical_key, r) for r in canonical_rows] + other_rows
    for key, row in all_rows:
        df_id = row.get("id")
        if not df_id:
            continue
        nid = node_id("Datafield", df_id)
        if nid not in G:
            G.add_node(nid, node_type="Datafield", name=df_id, file="catalogue")
            df_added += 1
        # Only set rich metadata from canonical snapshot to avoid clobbering.
        if key == canonical_key or "availability" not in G.nodes[nid]:
            ds = row.get("dataset") or {}
            cat = row.get("category") or {}
            sub = row.get("subcategory") or {}
            G.nodes[nid].update({
                "availability": "available",
                "description": row.get("description"),
                "dataset_id": ds.get("id"),
                "dataset_name": ds.get("name"),
                "category_id": cat.get("id"),
                "category_name": cat.get("name"),
                "subcategory_id": sub.get("id"),
                "subcategory_name": sub.get("name"),
                "df_type": row.get("type"),
                "coverage": row.get("coverage"),
                "date_coverage": row.get("dateCoverage"),
                "user_count": row.get("userCount"),
                "alpha_count_global": row.get("alphaCount"),
                "frequency": cat_mod.infer_frequency(ds.get("id", "")),
                "available_in_universes": sorted(set(seen_in_universes.get(df_id, []))),
                "themes": row.get("themes") or [],
            })

    return op_added, df_added


def build_graph():
    G = nx.DiGraph()

    # ── Load entity nodes ─────────────────────────────────────────────────────
    n_concepts = load_entity_nodes(G, CONCEPTS_DIR, "Concept")
    n_datafields = load_entity_nodes(G, DATAFIELDS_DIR, "Datafield")
    n_operators = load_entity_nodes(G, OPERATORS_DIR, "Operator")
    n_settings = load_entity_nodes(G, SETTINGS_DIR, "Setting")
    n_failure_modes = load_entity_nodes(G, FAILURE_MODES_DIR, "FailureMode")

    # ── Merge WQ Brain catalogue (operators + datafields available on platform) ─
    cat_mod, cat_snap = _load_brain_catalogue()
    n_ops_cat, n_dfs_cat = _add_catalogue_nodes(G, cat_mod, cat_snap)
    if n_ops_cat or n_dfs_cat:
        print(f"  catalogue: +{n_ops_cat} operators, +{n_dfs_cat} datafields")

    # Load session nodes from manifest
    manifest_path = SESSIONS_DIR / "_manifest.json"
    n_sessions = 0
    if manifest_path.exists():
        import json
        with open(manifest_path, encoding="utf-8") as f:
            sessions = json.load(f)
        for s in sessions:
            nid = node_id("Session", s["session_id"])
            G.add_node(nid, node_type="Session", name=s["title"],
                       date=s["date"], turn_count=s["turn_count"],
                       file=str(SESSIONS_DIR / f"{s['safe_id']}.md"))
            n_sessions += 1

    # ── Load alpha nodes + build edges ────────────────────────────────────────
    alpha_files = sorted(ALPHAS_DIR.glob("alpha_*.md"))
    n_alphas = 0
    edge_counts = Counter()

    # Helpers: ensure entity node exists even if agent forgot to write it
    def ensure_node(ntype, name):
        nid = node_id(ntype, name)
        if nid not in G:
            G.add_node(nid, node_type=ntype, name=name, file="auto-generated")
        return nid

    for alpha_file in alpha_files:
        try:
            post = frontmatter.load(str(alpha_file), encoding="utf-8")
            meta = post.metadata
            alpha_id = meta.get("id", alpha_file.stem)
            anid = node_id("Alpha", alpha_id)

            sharpe   = safe_val(meta.get("sharpe"))
            fitness  = safe_val(meta.get("fitness"))
            turnover = safe_val(meta.get("turnover"))
            ceiling  = ceiling_analysis(meta, sharpe, fitness, turnover)
            G.add_node(
                anid,
                node_type="Alpha",
                name=alpha_id,
                expression=meta.get("expression", ""),
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                returns=safe_val(meta.get("returns")),
                drawdown=safe_val(meta.get("drawdown")),
                margin=safe_val(meta.get("margin")),
                universe=meta.get("universe", "TOP3000"),
                region=meta.get("region", "USA"),
                delay=meta.get("delay", 1),
                neutralization=meta.get("neutralization", "market"),
                decay=meta.get("decay"),
                status=meta.get("status", "unknown"),
                rating=meta.get("rating", "Needs Improvement"),
                hypothesis=meta.get("hypothesis", ""),
                is_ceiling     = bool(ceiling),
                ceiling_blocked_by = ceiling["blocked_by"] if ceiling else None,
                ceiling_unblock_try = ceiling["unblock_try"] if ceiling else None,
                file=str(alpha_file),
            )
            n_alphas += 1

            # ── Parse yearly performance table from markdown body ─────────────────
            if YEARLY_PARSER_AVAILABLE:
                try:
                    content = post.content if post.content else ""
                    yearly_metrics = yearly_performance.parse_yearly_table(content)
                    for ym in yearly_metrics:
                        ypid = node_id("YearPerformance", f"{alpha_id}_{ym.year}")
                        G.add_node(
                            ypid,
                            node_type="YearPerformance",
                            name=f"{alpha_id}_{ym.year}",
                            year=ym.year,
                            sharpe=ym.sharpe,
                            turnover=ym.turnover,
                            fitness=ym.fitness,
                            returns=ym.returns,
                            drawdown=ym.drawdown,
                            margin=ym.margin,
                            regime=ym.regime,
                        )
                        G.add_edge(anid, ypid, relation="HAS_YEAR")
                        edge_counts["HAS_YEAR"] += 1
                except Exception:
                    pass

            # Alpha → Datafield (USES) — also marks the field as 'used'
            for df in safe_list(meta.get("datafields")):
                tgt = ensure_node("Datafield", df)
                G.add_edge(anid, tgt, relation="USES")
                G.nodes[tgt]["availability"] = "used"
                edge_counts["USES"] += 1

            # Alpha → Operator (APPLIES) — also marks the operator as 'used'
            for op in safe_list(meta.get("operators")):
                tgt = ensure_node("Operator", op)
                G.add_edge(anid, tgt, relation="APPLIES")
                G.nodes[tgt]["availability"] = "used"
                edge_counts["APPLIES"] += 1

            # Alpha → Concept (IMPLEMENTS)
            for c in safe_list(meta.get("concepts")):
                tgt = ensure_node("Concept", c)
                G.add_edge(anid, tgt, relation="IMPLEMENTS")
                edge_counts["IMPLEMENTS"] += 1

            # Alpha → Setting (TESTED_UNDER)
            univ = meta.get("universe", "TOP3000")
            delay = meta.get("delay", 1)
            neut = meta.get("neutralization", "market")
            if univ and univ != "null":
                setting_name = f"{univ}_{delay}_{neut}"
                tgt = ensure_node("Setting", setting_name)
                if not G.nodes[tgt].get("universe"):
                    G.nodes[tgt].update({"universe": univ, "delay": delay, "neutralization": neut})
                G.add_edge(anid, tgt, relation="TESTED_UNDER")
                edge_counts["TESTED_UNDER"] += 1

            # Alpha → FailureMode (FAILED_BY)
            for fm in safe_list(meta.get("failure_modes")):
                tgt = ensure_node("FailureMode", fm)
                G.add_edge(anid, tgt, relation="FAILED_BY")
                edge_counts["FAILED_BY"] += 1

            # Alpha → Alpha (DERIVED_FROM)
            parent = meta.get("parent_alpha")
            if parent and parent != "null":
                parent_nid = node_id("Alpha", parent)
                ensure_node("Alpha", parent)
                G.add_edge(anid, parent_nid, relation="DERIVED_FROM")
                edge_counts["DERIVED_FROM"] += 1

            # Alpha ↔ Alpha (CORRELATED_WITH — undirected, both ways)
            for corr in safe_list(meta.get("correlated_with")):
                corr_nid = node_id("Alpha", corr)
                ensure_node("Alpha", corr)
                G.add_edge(anid, corr_nid, relation="CORRELATED_WITH")
                G.add_edge(corr_nid, anid, relation="CORRELATED_WITH")
                edge_counts["CORRELATED_WITH"] += 2

            # Session → Alpha (PRODUCED)
            session_id = meta.get("session")
            if session_id and session_id != "null":
                snid = node_id("Session", session_id)
                if snid not in G:
                    G.add_node(snid, node_type="Session", name=session_id, file="unknown")
                G.add_edge(snid, anid, relation="PRODUCED")
                edge_counts["PRODUCED"] += 1

        except Exception as e:
            print(f"  WARN: could not parse {alpha_file.name}: {e}")

    # ── Save graph ────────────────────────────────────────────────────────────
    gpickle_path = GRAPH_DIR / "graph.gpickle"
    with open(gpickle_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    edges_path = GRAPH_DIR / "edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "relation"])
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get("relation", "")])

    # ── Print stats ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH STATS")
    print("=" * 60)

    type_counts = Counter(G.nodes[n]["node_type"] for n in G.nodes)
    print("\nNodes by type:")
    for ntype, cnt in sorted(type_counts.items()):
        print(f"  {ntype:<15} {cnt:>4}")
    print(f"  {'TOTAL':<15} {sum(type_counts.values()):>4}")

    print("\nEdges by relation:")
    for rel, cnt in sorted(edge_counts.items()):
        print(f"  {rel:<20} {cnt:>4}")
    print(f"  {'TOTAL':<20} {sum(edge_counts.values()):>4}")

    # Top 10 most-connected datafields
    df_degree = {
        n: G.in_degree(n)
        for n in G.nodes
        if G.nodes[n]["node_type"] == "Datafield"
    }
    print("\nTop 10 most-used datafields (by alpha count):")
    for name, deg in sorted(df_degree.items(), key=lambda x: -x[1])[:10]:
        display = G.nodes[name]["name"]
        print(f"  {display:<30} {deg:>3} alphas")

    # Top 10 operators
    op_degree = {
        n: G.in_degree(n)
        for n in G.nodes
        if G.nodes[n]["node_type"] == "Operator"
    }
    print("\nTop 10 most-used operators:")
    for name, deg in sorted(op_degree.items(), key=lambda x: -x[1])[:10]:
        display = G.nodes[name]["name"]
        print(f"  {display:<30} {deg:>3} alphas")

    # ── Catalogue coverage ────────────────────────────────────────────────
    df_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "Datafield"]
    op_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "Operator"]
    df_used = [n for n in df_nodes if G.nodes[n].get("availability") == "used"]
    df_avail = [n for n in df_nodes if G.nodes[n].get("availability") == "available"]
    op_used = [n for n in op_nodes if G.nodes[n].get("availability") == "used"]
    op_avail = [n for n in op_nodes if G.nodes[n].get("availability") == "available"]
    df_total_cat = len(df_used) + len(df_avail)
    op_total_cat = len(op_used) + len(op_avail)

    if df_total_cat > 0 or op_total_cat > 0:
        print("\nCatalogue coverage (used vs. available on WQ Brain):")
        if df_total_cat > 0:
            pct = 100.0 * len(df_used) / df_total_cat
            print(f"  Datafields: {len(df_used):>4} used / {df_total_cat:>4} catalogued  "
                  f"({pct:5.2f}% explored, {len(df_avail)} untried)")
        if op_total_cat > 0:
            pct = 100.0 * len(op_used) / op_total_cat
            print(f"  Operators:  {len(op_used):>4} used / {op_total_cat:>4} catalogued  "
                  f"({pct:5.2f}% explored, {len(op_avail)} untried)")

        # Per-dataset coverage for datafields
        ds_stats = defaultdict(lambda: {"used": 0, "total": 0, "name": ""})
        for n in df_nodes:
            d = G.nodes[n]
            ds_id = d.get("dataset_id")
            if not ds_id:
                continue
            ds_stats[ds_id]["total"] += 1
            ds_stats[ds_id]["name"] = d.get("dataset_name") or ds_id
            if d.get("availability") == "used":
                ds_stats[ds_id]["used"] += 1

        if ds_stats:
            print("\n  Per-dataset coverage (sorted by used count):")
            ranked = sorted(ds_stats.items(),
                            key=lambda kv: (-kv[1]["used"], -kv[1]["total"]))
            for ds_id, s in ranked[:15]:
                pct = 100.0 * s["used"] / s["total"] if s["total"] else 0.0
                name = (s["name"] or "")[:32]
                bar_n = int(pct / 5)
                bar = "#" * bar_n + "." * (20 - bar_n)
                print(f"    {ds_id:<14} {name:<32} {s['used']:>3}/{s['total']:>4} "
                      f"{pct:5.1f}% {bar}")
            unranked_used = sum(1 for n in df_used if not G.nodes[n].get("dataset_id"))
            if unranked_used:
                print(f"    {'(uncatalogued)':<14} {'':<32} {unranked_used:>3}/{unranked_used:<4} "
                      f"  n/a  (used but not in catalogue snapshot)")

    # Top 5 concepts
    concept_degree = {
        n: G.in_degree(n)
        for n in G.nodes
        if G.nodes[n]["node_type"] == "Concept"
    }
    print("\nTop 5 concepts by alpha count:")
    for name, deg in sorted(concept_degree.items(), key=lambda x: -x[1])[:5]:
        display = G.nodes[name]["name"]
        print(f"  {display:<30} {deg:>3} alphas")

    # Alphas per universe/delay combo
    combos = Counter()
    for n in G.nodes:
        if G.nodes[n]["node_type"] == "Alpha":
            univ = G.nodes[n].get("universe", "unknown")
            delay = G.nodes[n].get("delay", "?")
            combos[(univ, delay)] += 1
    print("\nAlphas per universe/delay combo:")
    for (univ, delay), cnt in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"  {univ} delay={delay}: {cnt}")

    # Failure mode frequency
    fm_degree = {
        n: G.in_degree(n)
        for n in G.nodes
        if G.nodes[n]["node_type"] == "FailureMode"
    }
    print("\nFailure mode frequency:")
    for name, deg in sorted(fm_degree.items(), key=lambda x: -x[1]):
        display = G.nodes[name]["name"]
        print(f"  {display:<30} {deg:>3}")

    # YearPerformance nodes
    year_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "YearPerformance"]
    if year_nodes:
        print(f"\nYearPerformance nodes: {len(year_nodes)}")
        regime_counts = Counter(G.nodes[n].get("regime") for n in year_nodes)
        print("  By regime:")
        for reg, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {reg:<20} {cnt}")

    # Ceiling alphas summary
    ceiling_alphas = [n for n in G.nodes
                      if G.nodes[n].get("node_type") == "Alpha"
                      and G.nodes[n].get("is_ceiling")]
    if ceiling_alphas:
        print(f"\nCeiling alphas (Sharpe >=1.5 but blocked): {len(ceiling_alphas)}")
        for nid in sorted(ceiling_alphas,
                          key=lambda n: -(G.nodes[n].get("sharpe") or 0))[:10]:
            d = G.nodes[nid]
            print(f"  {d['name']:<14} Sharpe={d['sharpe']:.2f}  "
                  f"blocked: {d.get('ceiling_blocked_by','')[:80]}")

    print(f"\nGraph saved to {gpickle_path}")
    print(f"Edges CSV saved to {edges_path}")
    return G


if __name__ == "__main__":
    build_graph()

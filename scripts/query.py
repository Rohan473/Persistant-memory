"""
Phase 5: CLI retrieval helper for the WQ Brain knowledge graph.

Usage:
  python query.py concept mean_reversion
  python query.py datafield close
  python query.py setting TOP3000 1 industry
  python query.py failures
  python query.py gaps
  python query.py lineage alpha_0042
  python query.py best 10
  python query.py regime 2022
  python query.py year alpha_0023
  python query.py crisis-robust
  python query.py gaps-catalogue          # underexplored datafields (vs WQ Brain catalogue)
  python query.py gaps-catalogue 30 quality  # top 30 in 'quality' category/subcategory
  python query.py operators-available     # untried operators
  python query.py dataset model77         # all fields in a dataset
  python query.py quality-gaps            # high-coverage (>=0.8) untried fields, all datasets
  python query.py quality-gaps option8    # same, restricted to one dataset
  python query.py quality-gaps option8 0.9  # same, custom min coverage
  python query.py search volatility       # fuzzy search across names + descriptions
  python query.py search "earnings revision" 30  # custom result limit
  python query.py memory                  # list all auto-memory entries
  python query.py memory momentum         # show entries matching a keyword
"""

import pickle
import sys
import importlib.util
from pathlib import Path
from collections import Counter

# Lazy load project_memory to avoid the heavy memory_layer/__init__ chain
_MEMORIES_CACHE = None
def _memories():
    global _MEMORIES_CACHE
    if _MEMORIES_CACHE is None:
        try:
            spec = importlib.util.spec_from_file_location(
                "project_memory",
                Path(__file__).resolve().parent.parent / "memory_layer" / "project_memory.py",
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            _MEMORIES_CACHE = (mod, mod.load_memories())
        except Exception:
            _MEMORIES_CACHE = (None, [])
    return _MEMORIES_CACHE


def _memory_hint(*terms, types=None, header="Project memory", limit=3):
    """Print a small block of relevant project-memory entries, if any."""
    mod, mems = _memories()
    if not mod or not mems:
        return
    hits = mod.find_relevant(mems, *terms, types=types, limit=limit)
    block = mod.format_memory_block(hits, header=header)
    if block:
        print(block, end="")


def _log_to_active_session():
    """If a research session is open, log this CLI invocation. Silent on failure."""
    try:
        spec = importlib.util.spec_from_file_location(
            "sessions",
            Path(__file__).resolve().parent.parent / "memory_layer" / "sessions.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mod.log_invocation(sys.argv)
    except Exception:
        pass

BASE = Path(__file__).resolve().parent.parent
GRAPH_PATH = BASE / "graph" / "graph.gpickle"


def load_graph():
    if not GRAPH_PATH.exists():
        print("Graph not built yet. Run: python scripts/build_graph.py")
        sys.exit(1)
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def node_id(ntype, name):
    return f"{ntype}::{name}"


def get_alpha_display(G, anid):
    d = G.nodes[anid]
    sharpe = d.get("sharpe")
    sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "n/a"
    return (
        f"  {d['name']:<15} "
        f"Sharpe={sharpe_str:<7} "
        f"status={d.get('status','?'):<12} "
        f"{d.get('expression','')[:60]}"
    )


def cmd_concept(G, concept_name):
    target = node_id("Concept", concept_name)
    if target not in G:
        # fuzzy match
        matches = [n for n in G.nodes if G.nodes[n]["node_type"] == "Concept"
                   and concept_name.lower() in G.nodes[n]["name"].lower()]
        if not matches:
            print(f"Concept '{concept_name}' not found.")
            return
        target = matches[0]
        print(f"(matched: {G.nodes[target]['name']})")

    alphas = [u for u, v, d in G.in_edges(target, data=True)
              if d.get("relation") == "IMPLEMENTS"]
    alphas_data = [(G.nodes[a].get("sharpe") or -99, a) for a in alphas]
    alphas_data.sort(reverse=True)

    print(f"\nConcept: {G.nodes[target]['name']} — {len(alphas)} alpha(s)")
    print("-" * 70)
    for _, anid in alphas_data:
        print(get_alpha_display(G, anid))
        print(f"    hypothesis: {G.nodes[anid].get('hypothesis','')}")
        print()
    _memory_hint(G.nodes[target]['name'])


def cmd_datafield(G, field_name):
    target = node_id("Datafield", field_name)
    if target not in G:
        q = field_name.lower()
        # Tier 1: substring in field name
        matches = [n for n in G.nodes
                   if G.nodes[n].get("node_type") == "Datafield"
                   and q in G.nodes[n].get("name", "").lower()]
        # Tier 2: fall back to substring in description
        if not matches:
            matches = [n for n in G.nodes
                       if G.nodes[n].get("node_type") == "Datafield"
                       and q in (G.nodes[n].get("description") or "").lower()]
        if not matches:
            print(f"Datafield '{field_name}' not found.")
            print(f"Hint: try `python scripts/query.py search {field_name}` for broader matching.")
            return
        # Sort: exact name → starts-with → contains → description-only, then by user_count
        def rank(nid):
            name = G.nodes[nid].get("name", "").lower()
            if name == q:        tier = 0
            elif name.startswith(q): tier = 1
            elif q in name:      tier = 2
            else:                tier = 3
            return (tier, -(G.nodes[nid].get("user_count") or 0))
        matches.sort(key=rank)
        target = matches[0]
        if G.nodes[target].get("name", "").lower() != q:
            print(f"(no exact match — showing best of {len(matches)} candidates)\n")
            if len(matches) > 1:
                print("Other candidates:")
                for nid in matches[1:6]:
                    n = G.nodes[nid]
                    desc = (n.get("description") or "")[:48]
                    print(f"  {n['name']:<36} users={(n.get('user_count') or 0):>6}  {desc}")
                if len(matches) > 6:
                    print(f"  ... and {len(matches) - 6} more — run "
                          f"`python scripts/query.py search {field_name}` to see all")
                print()

    d = G.nodes[target]
    alphas = [u for u, v, ed in G.in_edges(target, data=True)
              if ed.get("relation") == "USES"]

    # Header: catalogue metadata
    print(f"\nDatafield: {d['name']}")
    print("=" * 78)
    desc = d.get("description")
    if desc:
        # Wrap long descriptions
        words = desc.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > 74:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            lines.append(line)
        for ln in lines:
            print(f"  {ln}")
        print()

    rows = []
    if d.get("dataset_id"):
        rows.append(("Dataset",      f"{d.get('dataset_id')} ({d.get('dataset_name') or '?'})"))
    if d.get("category_name"):
        sub = d.get("subcategory_name") or ""
        cat_str = d["category_name"] + (f" > {sub}" if sub else "")
        rows.append(("Category",     cat_str))
    if d.get("df_type"):
        rows.append(("Type",         d["df_type"]))
    if d.get("frequency"):
        rows.append(("Frequency",    d["frequency"]))
    if d.get("coverage") is not None:
        rows.append(("Coverage",     f"{d['coverage']:.2f} (universe), date={d.get('date_coverage', '?')}"))
    if d.get("user_count") is not None:
        rows.append(("Global usage", f"{d['user_count']:>6} users, "
                                     f"{d.get('alpha_count_global', 0):>6} alphas"))
    rows.append(("Availability",     d.get("availability") or "unknown"))
    rows.append(("Your alphas",      f"{len(alphas)}"))
    for label, value in rows:
        print(f"  {label:<14} {value}")

    if alphas:
        print(f"\nUsed by {len(alphas)} of your alpha(s):")
        print("-" * 78)
        alphas_data = [(G.nodes[a].get("sharpe") or -99, a) for a in alphas]
        alphas_data.sort(reverse=True)
        for _, anid in alphas_data:
            print(get_alpha_display(G, anid))
    _memory_hint(d["name"], d.get("dataset_id") or "")


def cmd_setting(G, universe, delay, neutralization):
    setting_name = f"{universe}_{delay}_{neutralization}"
    target = node_id("Setting", setting_name)
    if target not in G:
        # Try partial match
        matches = [n for n in G.nodes if G.nodes[n]["node_type"] == "Setting"
                   and universe.lower() in n.lower()]
        if not matches:
            print(f"Setting '{setting_name}' not found.")
            print("Available settings:")
            for n in G.nodes:
                if G.nodes[n]["node_type"] == "Setting":
                    print(f"  {G.nodes[n]['name']}")
            return
        target = matches[0]
        print(f"(matched: {G.nodes[target]['name']})")

    alphas = [u for u, v, d in G.in_edges(target, data=True)
              if d.get("relation") == "TESTED_UNDER"]
    alphas_data = [(G.nodes[a].get("sharpe") or -99, a) for a in alphas]
    alphas_data.sort(reverse=True)

    print(f"\nSetting: {G.nodes[target]['name']} — {len(alphas)} alpha(s)")
    print("-" * 70)
    for _, anid in alphas_data:
        print(get_alpha_display(G, anid))


def cmd_failures(G):
    fm_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "FailureMode"]
    counts = [(G.in_degree(n), G.nodes[n]["name"]) for n in fm_nodes]
    counts.sort(reverse=True)

    print("\nFailure mode frequency table:")
    print("-" * 45)
    print(f"  {'Failure Mode':<30} {'Count':>5}")
    print(f"  {'-'*30} {'-----':>5}")
    for cnt, name in counts:
        print(f"  {name:<30} {cnt:>5}")


def cmd_gaps(G):
    df_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "Datafield"]
    gaps = [(G.in_degree(n), G.nodes[n]["name"]) for n in df_nodes if G.in_degree(n) < 3]
    gaps.sort()

    print("\nDatafields used in fewer than 3 alphas (exploration gaps):")
    print("-" * 50)
    for cnt, name in gaps:
        print(f"  {name:<30} {cnt:>2} alpha(s)")


def cmd_lineage(G, alpha_id):
    # Normalize
    if not alpha_id.startswith("Alpha::"):
        anid = node_id("Alpha", alpha_id)
    else:
        anid = alpha_id

    if anid not in G:
        print(f"Alpha '{alpha_id}' not found.")
        return

    def print_tree(nid, depth=0, visited=None):
        if visited is None:
            visited = set()
        if nid in visited:
            return
        visited.add(nid)
        d = G.nodes[nid]
        sharpe = d.get("sharpe")
        sharpe_str = f"Sharpe={sharpe:.2f}" if sharpe is not None else ""
        prefix = "  " * depth + ("+-- " if depth > 0 else "")
        print(f"{prefix}{d['name']} {sharpe_str} [{d.get('status','?')}]")
        print(f"{'  '*(depth+1)}expr: {d.get('expression','')[:55]}")
        # Find children (alphas derived FROM this one)
        children = [u for u, v, ed in G.out_edges(nid, data=True)
                    if ed.get("relation") == "DERIVED_FROM" and v == nid]
        # Find alphas that have DERIVED_FROM edges pointing TO this alpha
        children2 = [u for u, v, ed in G.in_edges(nid, data=True)
                     if ed.get("relation") == "DERIVED_FROM"]
        # Actually: DERIVED_FROM means "A was derived from B", so edge is A→B.
        # To find children of nid (alphas derived from nid), look at in-edges:
        derived_children = [u for u, v, ed in G.edges(data=True)
                            if v == nid and ed.get("relation") == "DERIVED_FROM"]
        for child in derived_children:
            print_tree(child, depth + 1, visited)

    # Walk up to root
    def find_root(nid, visited=None):
        if visited is None:
            visited = set()
        if nid in visited:
            return nid
        visited.add(nid)
        parents = [v for u, v, ed in G.out_edges(nid, data=True)
                   if ed.get("relation") == "DERIVED_FROM"]
        if not parents:
            return nid
        return find_root(parents[0], visited)

    root = find_root(anid)
    print(f"\nLineage tree rooted at {G.nodes[root]['name']}:")
    print("-" * 60)
    print_tree(root)


def cmd_best(G, n):
    alpha_nodes = [(G.nodes[nid].get("sharpe"), nid)
                   for nid in G.nodes if G.nodes[nid]["node_type"] == "Alpha"
                   and G.nodes[nid].get("sharpe") is not None]
    alpha_nodes.sort(reverse=True)

    print(f"\nTop {n} alphas by Sharpe:")
    print("=" * 80)
    RATING_LABEL = {
        "Good":             "[GOOD]",
        "Average":          "[AVG] ",
        "Needs Improvement":"[NI]  ",
    }
    for sharpe, anid in alpha_nodes[:n]:
        d      = G.nodes[anid]
        rating = d.get("rating", "Needs Improvement")
        rlabel = RATING_LABEL.get(rating, "[?]  ")
        print(f"\n{rlabel} {d['name']}")
        print(f"  Expression  : {d.get('expression','')}")
        print(f"  Sharpe      : {sharpe:.3f}")
        print(f"  Fitness     : {d.get('fitness')}")
        print(f"  Turnover    : {d.get('turnover')}%")
        print(f"  Returns     : {d.get('returns')}%  Drawdown: {d.get('drawdown')}%  Margin: {d.get('margin')}bps")
        print(f"  Universe    : {d.get('universe')} | Region: {d.get('region')} | Delay: {d.get('delay')}")
        print(f"  Neutraliz.  : {d.get('neutralization')}")
        print(f"  Status      : {d.get('status')}  Rating: {rating}")
        print(f"  Hypothesis  : {d.get('hypothesis','')}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    _log_to_active_session()

    G = load_graph()
    cmd = args[0].lower()

    if cmd == "concept" and len(args) >= 2:
        cmd_concept(G, args[1])
    elif cmd == "datafield" and len(args) >= 2:
        cmd_datafield(G, args[1])
    elif cmd == "setting" and len(args) >= 4:
        cmd_setting(G, args[1], args[2], args[3])
    elif cmd == "failures":
        cmd_failures(G)
    elif cmd == "gaps":
        cmd_gaps(G)
    elif cmd == "lineage" and len(args) >= 2:
        cmd_lineage(G, args[1])
    elif cmd == "best":
        n = int(args[1]) if len(args) >= 2 else 10
        cmd_best(G, n)
    elif cmd == "ceiling":
        cmd_ceiling(G)
    elif cmd == "regime" and len(args) >= 2:
        cmd_regime(G, args[1])
    elif cmd == "year" and len(args) >= 2:
        cmd_year(G, args[1])
    elif cmd == "crisis-robust":
        cmd_crisis_robust(G)
    elif cmd == "gaps-catalogue":
        n = int(args[1]) if len(args) >= 2 and args[1].isdigit() else 25
        filt = args[2] if len(args) >= 3 else None
        cmd_gaps_catalogue(G, n, filt)
    elif cmd == "operators-available":
        cmd_operators_available(G)
    elif cmd == "dataset" and len(args) >= 2:
        cmd_dataset(G, args[1])
    elif cmd == "quality-gaps":
        dataset = args[1] if len(args) >= 2 else None
        try:
            min_cov = float(args[2]) if len(args) >= 3 else 0.8
        except ValueError:
            min_cov = 0.8
        cmd_quality_gaps(G, dataset=dataset, min_coverage=min_cov)
    elif cmd == "search" and len(args) >= 2:
        pattern = args[1]
        try:
            limit = int(args[2]) if len(args) >= 3 else 20
        except ValueError:
            limit = 20
        cmd_search(G, pattern, limit=limit)
    elif cmd == "memory":
        cmd_memory(args[1:])
    else:
        print(__doc__)


def cmd_ceiling(G):
    nodes = [(G.nodes[n].get("sharpe") or 0, n) for n in G.nodes
             if G.nodes[n].get("node_type") == "Alpha"
             and G.nodes[n].get("is_ceiling")]
    nodes.sort(reverse=True)
    print(f"\nCeiling alphas — high Sharpe but blocked ({len(nodes)} total)")
    print("=" * 80)
    for sh, anid in nodes:
        d = G.nodes[anid]
        print(f"\n{d['name']}  Sharpe={sh:.2f}  Status={d.get('status')}")
        print(f"  Blocked by  : {d.get('ceiling_blocked_by','')}")
        if d.get("ceiling_unblock_try"):
            print(f"  Try         : {d['ceiling_unblock_try']}")
        print(f"  Expression  : {(d.get('expression') or '')[:75]}")


def cmd_regime(G, year):
    """Show alphas that performed in a specific year/regime."""
    try:
        target_year = int(year)
    except ValueError:
        print(f"Invalid year: {year}")
        return

    year_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "YearPerformance"
                  and G.nodes[n].get("year") == target_year]

    if not year_nodes:
        print(f"No YearPerformance data found for year {target_year}")
        return

    alphas_data = []
    for yn in year_nodes:
        d = G.nodes[yn]
        sharpe = d.get("sharpe", 0) or 0
        regime = d.get("regime", "unknown")
        alpha_nid = list(G.predecessors(yn))[0] if G.predecessors(yn) else None
        if alpha_nid:
            alpha_name = G.nodes[alpha_nid].get("name", "")
            alphas_data.append((sharpe, alpha_name, d.get("returns"), regime))

    alphas_data.sort(reverse=True)

    print(f"\nAlphas in {target_year} (sorted by Sharpe):")
    print("=" * 70)
    for sharpe, alpha_name, returns, regime in alphas_data:
        print(f"  {alpha_name:<15} Sharpe={sharpe:+.2f}  Returns={returns or 0:+.2f}%  [{regime}]")


def cmd_year(G, alpha_id):
    """Show year-by-year breakdown for an alpha."""
    if not alpha_id.startswith("Alpha::"):
        anid = node_id("Alpha", alpha_id)
    else:
        anid = alpha_id

    if anid not in G:
        print(f"Alpha '{alpha_id}' not found.")
        return

    year_edges = [(v, G.edges[anid, v]) for v in G.successors(anid)
                  if G.nodes[v].get("node_type") == "YearPerformance"]

    if not year_edges:
        print(f"No yearly performance data for {alpha_id}")
        return

    year_edges.sort(key=lambda x: G.nodes[x[0]].get("year", 0))

    d = G.nodes[anid]
    print(f"\n{alpha_id} — Year-by-year breakdown:")
    print("=" * 70)
    print(f"  {'Year':<6} {'Sharpe':>8} {'Turnover':>10} {'Fitness':>8} {'Returns':>10} {'Regime'}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*12}")
    def fmt(v, spec):
        if v is None:
            return "    n/a "
        try:
            return format(v, spec)
        except (TypeError, ValueError):
            return "    n/a "
    for ypid, _ in year_edges:
        yd = G.nodes[ypid]
        print(f"  {yd.get('year',''):<6} {fmt(yd.get('sharpe'),'>+8.2f')} {fmt(yd.get('turnover'),'>9.1f')}% {fmt(yd.get('fitness'),'>+8.2f')} {fmt(yd.get('returns'),'>+9.2f')}%  {yd.get('regime','')}")


def cmd_crisis_robust(G):
    """Show alphas that performed well in crisis years (2008, 2020, 2022)."""
    crisis_years = [2008, 2020, 2022]

    alpha_sharpes = {}
    for year in crisis_years:
        year_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "YearPerformance"
                      and G.nodes[n].get("year") == year]
        for yn in year_nodes:
            sharpe = G.nodes[yn].get("sharpe") or 0
            preds = list(G.predecessors(yn))
            if preds:
                anid = preds[0]
                alpha_name = G.nodes[anid].get("name", "")
                if alpha_name not in alpha_sharpes:
                    alpha_sharpes[alpha_name] = []
                alpha_sharpes[alpha_name].append(sharpe)

    avg_sharpes = [(sum(sharpes) / len(sharpes), name, sharpes)
                   for name, sharpes in alpha_sharpes.items() if len(sharpes) >= 2]
    avg_sharpes.sort(reverse=True)

    print(f"\nCrisis-robust alphas (avg Sharpe in 2008, 2020, 2022):")
    print("=" * 70)
    for avg, name, sharpes in avg_sharpes[:15]:
        print(f"  {name:<15} avg={avg:+.2f}  ({(', '.join(f'{s:+.2f}' for s in sharpes))})")


def cmd_gaps_catalogue(G, n=25, filter_str=None):
    """List datafields that are catalogued in WQ Brain but unused by any alpha.
    Sorted by global user_count × coverage (most-validated underexplored fields first)."""
    candidates = []
    for nid in G.nodes:
        d = G.nodes[nid]
        if d.get("node_type") != "Datafield":
            continue
        if d.get("availability") != "available":
            continue
        if filter_str:
            f = filter_str.lower()
            hay = " ".join(str(d.get(k, "")) for k in
                           ("name", "dataset_name", "category_name", "subcategory_name")).lower()
            if f not in hay:
                continue
        score = (d.get("user_count") or 0) * (d.get("coverage") or 0)
        candidates.append((score, nid, d))
    candidates.sort(reverse=True, key=lambda x: x[0])

    print(f"\nUnderexplored datafields (catalogued but unused){' — filter: ' + filter_str if filter_str else ''}:")
    print(f"Total: {len(candidates)}  showing top {min(n, len(candidates))}")
    print("=" * 90)
    print(f"  {'name':<32} {'dataset':<16} {'freq':<9} {'users':>6} {'cov':>5} {'alphas':>7}")
    print(f"  {'-'*32} {'-'*16} {'-'*9} {'-'*6} {'-'*5} {'-'*7}")
    for _, nid, d in candidates[:n]:
        print(f"  {d['name'][:32]:<32} {(d.get('dataset_id') or '')[:16]:<16} "
              f"{(d.get('frequency') or '?'):<9} "
              f"{(d.get('user_count') or 0):>6} "
              f"{(d.get('coverage') or 0):>5.2f} "
              f"{(d.get('alpha_count_global') or 0):>7}")


def cmd_operators_available(G):
    """List operators that exist on the WQ Brain platform but you've never used."""
    untried = []
    for nid in G.nodes:
        d = G.nodes[nid]
        if d.get("node_type") != "Operator":
            continue
        if d.get("availability") != "available":
            continue
        untried.append(d)
    untried.sort(key=lambda d: (d.get("category") or "", d.get("name") or ""))

    print(f"\nUntried operators ({len(untried)} of {sum(1 for n in G.nodes if G.nodes[n].get('node_type')=='Operator')} catalogued):")
    print("=" * 90)
    by_cat = {}
    for d in untried:
        by_cat.setdefault(d.get("category") or "Uncategorized", []).append(d)
    for cat, items in sorted(by_cat.items()):
        print(f"\n[{cat}]")
        for d in items:
            defn = (d.get("definition") or "")[:55]
            print(f"  {d['name']:<28} {defn}")


def cmd_dataset(G, dataset_id):
    """Show all catalogued datafields in a given dataset, with usage status."""
    rows = []
    for nid in G.nodes:
        d = G.nodes[nid]
        if d.get("node_type") != "Datafield":
            continue
        if d.get("dataset_id") != dataset_id:
            continue
        rows.append(d)
    if not rows:
        print(f"No datafields found for dataset '{dataset_id}'.")
        print("Hint: dataset IDs look like 'model77', 'fundamental6', 'analyst4', 'pv1'.")
        return

    used = [r for r in rows if r.get("availability") == "used"]
    avail = [r for r in rows if r.get("availability") != "used"]
    avail.sort(key=lambda d: (d.get("user_count") or 0), reverse=True)

    ds_name = rows[0].get("dataset_name", "")
    print(f"\nDataset {dataset_id} ({ds_name}): {len(rows)} fields, {len(used)} used by you, {len(avail)} untried")
    print("=" * 90)
    if used:
        print(f"\nUSED ({len(used)}):")
        for d in used:
            print(f"  {d['name']:<32} cov={d.get('coverage', 0):.2f}  users={d.get('user_count',0)}")
    print(f"\nUNTRIED ({len(avail)}, top 15 by user_count):")
    for d in avail[:15]:
        print(f"  {d['name']:<32} cov={d.get('coverage',0):.2f}  users={d.get('user_count',0):>5}  alphas={d.get('alpha_count_global',0):>5}")


def cmd_quality_gaps(G, dataset=None, min_coverage=0.8, n=25):
    """High-coverage untried datafields — the 'ready to use today' shortlist.

    Filters: availability=='available' AND coverage>=min_coverage.
    Optionally restrict to a single dataset_id. Excludes group-key fields
    (sector/industry/subindustry/market) since those are neutralization keys,
    not signal candidates.
    """
    EXCLUDE_NAMES = {"sector", "industry", "subindustry", "market", "country"}
    candidates = []
    for nid in G.nodes:
        d = G.nodes[nid]
        if d.get("node_type") != "Datafield":
            continue
        if d.get("availability") != "available":
            continue
        if (d.get("coverage") or 0) < min_coverage:
            continue
        if d.get("name", "").lower() in EXCLUDE_NAMES:
            continue
        if dataset and d.get("dataset_id") != dataset:
            continue
        candidates.append(d)

    candidates.sort(key=lambda d: (d.get("user_count") or 0,
                                   d.get("alpha_count_global") or 0),
                    reverse=True)

    scope = f"dataset={dataset}" if dataset else "all datasets"
    print(f"\nHigh-quality untried datafields ({scope}, coverage >= {min_coverage}):")
    print(f"Total matching: {len(candidates)}  showing top {min(n, len(candidates))}")
    print("=" * 100)
    print(f"  {'name':<36} {'dataset':<14} {'freq':<10} {'cov':>5} {'users':>7} {'alphas':>8}")
    print(f"  {'-'*36} {'-'*14} {'-'*10} {'-'*5} {'-'*7} {'-'*8}")
    for d in candidates[:n]:
        print(f"  {d['name'][:36]:<36} "
              f"{(d.get('dataset_id') or '')[:14]:<14} "
              f"{(d.get('frequency') or '?'):<10} "
              f"{(d.get('coverage') or 0):>5.2f} "
              f"{(d.get('user_count') or 0):>7} "
              f"{(d.get('alpha_count_global') or 0):>8}")

    if not dataset and candidates:
        # Tail: how the gap distributes by dataset
        from collections import Counter as _Counter
        by_ds = _Counter()
        for d in candidates:
            by_ds[d.get("dataset_id") or "?"] += 1
        print(f"\n  By dataset (matching fields):")
        for ds, cnt in by_ds.most_common(10):
            print(f"    {ds:<14} {cnt}")


def cmd_memory(args):
    """List or search auto-memory entries kept by Claude Code."""
    mod, mems = _memories()
    if not mod:
        print("project_memory module unavailable.")
        return
    if not mems:
        print("No auto-memory entries found. "
              f"(Expected directory: {mod.DEFAULT_MEMORY_DIR})")
        return
    if args:
        hits = mod.find_relevant(mems, *args, limit=25)
        if not hits:
            print(f"No memory entries match: {' '.join(args)}")
            return
        print(f"\nMemory entries matching '{' '.join(args)}' ({len(hits)}):")
    else:
        hits = mems
        print(f"\nAll memory entries ({len(hits)}):")
    print("=" * 78)
    for e in hits:
        print(f"  [{e.mem_type or 'unknown':<8}] {e.title}")
        if e.description:
            print(f"             {e.description}")
        print(f"             ({e.file_path.name})")


def cmd_search(G, pattern, limit=20):
    """Fuzzy search across datafield names + descriptions.

    Ranking:
      exact name           → 1000
      name starts-with     →  500
      name substring       →  200
      description substring →  50
    Then add (user_count / 1000) so popular fields surface within each tier.
    """
    q = pattern.lower()
    scored = []
    for nid in G.nodes:
        d = G.nodes[nid]
        if d.get("node_type") != "Datafield":
            continue
        name = (d.get("name") or "").lower()
        desc = (d.get("description") or "").lower()
        if name == q:                 tier = 1000
        elif name.startswith(q):      tier = 500
        elif q in name:               tier = 200
        elif q in desc:               tier = 50
        else:                         continue
        score = tier + (d.get("user_count") or 0) / 1000.0
        scored.append((score, d))
    scored.sort(reverse=True, key=lambda x: x[0])

    import shutil, textwrap
    term_w = max(shutil.get_terminal_size((120, 24)).columns, 80)

    print(f"\nSearch for '{pattern}':")
    print(f"Total: {len(scored)}, showing top {min(limit, len(scored))}")
    print("=" * min(term_w, 120))
    if not scored:
        print("  (no matches)")
        return
    print(f"  {'name':<36} {'dataset':<14} {'avail':<10} {'cov':>5} {'users':>7}")
    print(f"  {'-'*36} {'-'*14} {'-'*10} {'-'*5} {'-'*7}")
    desc_width = max(40, term_w - 8)
    for _, d in scored[:limit]:
        print(f"  {(d.get('name') or '')[:36]:<36} "
              f"{(d.get('dataset_id') or '')[:14]:<14} "
              f"{(d.get('availability') or '?'):<10} "
              f"{(d.get('coverage') or 0):>5.2f} "
              f"{(d.get('user_count') or 0):>7}")
        desc = (d.get("description") or "").strip()
        if desc:
            for ln in textwrap.wrap(desc, width=desc_width,
                                    initial_indent="      ",
                                    subsequent_indent="      "):
                print(ln)
    # Footer: where the matches live
    if len(scored) > limit:
        from collections import Counter as _C
        ds_hist = _C((d.get("dataset_id") or "?") for _, d in scored)
        print(f"\n  Datasets containing matches:")
        for ds, cnt in ds_hist.most_common(8):
            print(f"    {ds:<14} {cnt}")


if __name__ == "__main__":
    main()

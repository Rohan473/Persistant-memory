"""
visualize.py — WQ Brain knowledge graph viewer.

Modes
  --lineage <alpha_id>             ASCII parent/child tree (rich)
  --explore                        D3 force-directed graph (default: --top 50)
    --type <NodeType>              keep only nodes of this type
    --concept <name>               alphas implementing concept + neighbors
    --top <N>                      top N alphas by Sharpe + neighbors
    --max-nodes <N>                hard cap on subgraph (default 500, drops least-connected)
"""

import argparse
import json
import pickle
import sys
import webbrowser
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE      = Path(__file__).resolve().parent.parent
GRAPH_PKL = BASE / "graph" / "graph.gpickle"
HTML_OUT  = BASE / "graph" / "graph.html"
JSON_OUT  = BASE / "graph" / "graph.json"

NODE_COLORS = {
    "Alpha": "#4A90E2", "Concept": "#7ED321", "Datafield": "#F5A623",
    "Operator": "#9B59B6", "Setting": "#1ABC9C", "FailureMode": "#E74C3C",
    "Session": "#95A5A6",
}


def load_graph():
    if not GRAPH_PKL.exists():
        sys.exit("Graph not built. Run: python scripts/build_graph.py")
    with open(GRAPH_PKL, "rb") as f:
        return pickle.load(f)


# ─── MODE 1: --lineage ────────────────────────────────────────────────────────

def lineage_mode(G, alpha_id):
    from rich.console import Console
    from rich.tree import Tree
    console = Console(file=sys.stdout)

    nid = alpha_id if alpha_id.startswith("Alpha::") else f"Alpha::{alpha_id}"
    if nid not in G:
        cand = [n for n in G if G.nodes[n].get("node_type") == "Alpha"
                and alpha_id.lower() in n.lower()]
        if not cand:
            console.print(f"[red]Alpha '{alpha_id}' not found.[/red]"); return
        nid = cand[0]

    def find_root(n, seen=None):
        seen = seen or set()
        if n in seen: return n
        seen.add(n)
        parents = [v for _, v, d in G.out_edges(n, data=True)
                   if d.get("relation") == "DERIVED_FROM"]
        return find_root(parents[0], seen) if parents else n

    icons = {"submitted":"[green][PASS][/green]", "rejected":"[red][FAIL][/red]",
             "iterating":"[yellow][WIP][/yellow]", "idea_only":"[dim][IDEA][/dim]"}

    def label(n):
        d = G.nodes[n]; name = d.get("name", n); sharpe = d.get("sharpe")
        icon = icons.get(d.get("status","?"), "[dim][?][/dim]")
        s = f"[cyan]Sharpe={sharpe:.2f}[/cyan]" if sharpe is not None else "[dim]Sharpe=n/a[/dim]"
        concepts = [G.nodes[v].get("name","") for _, v, ed in G.out_edges(n, data=True)
                    if ed.get("relation") == "IMPLEMENTS"]
        c = f"  [green dim][{', '.join(concepts[:3])}][/green dim]" if concepts else ""
        marker = " [bold magenta]<< queried[/bold magenta]" if n == nid else ""
        return (f"{icon} [bold blue]{name}[/bold blue]  {s}{c}{marker}\n"
                f"   [dim]{(d.get('expression') or '')[:60]}[/dim]")

    def build(n, branch, seen=None):
        seen = seen or set()
        if n in seen: return
        seen.add(n)
        kids = sorted([u for u, v, d in G.edges(data=True)
                       if v == n and d.get("relation") == "DERIVED_FROM"],
                      key=lambda x: -(G.nodes[x].get("sharpe") or -99))
        for k in kids:
            build(k, branch.add(label(k)), seen)

    root = find_root(nid)
    tree = Tree(label(root)); build(root, tree)
    console.print(); console.rule(f"[bold]Lineage — [blue]{alpha_id}[/blue]")
    console.print(tree); console.print()


# ─── MODE 2: --explore (D3) ───────────────────────────────────────────────────

def filter_subgraph(G, args):
    if args.type:
        keep = {n for n in G if G.nodes[n].get("node_type") == args.type}
        # For Alpha-only, also include alpha-to-alpha edges by keeping them subgraph-relative
        which = f"type={args.type}"
    elif args.concept:
        nid = f"Concept::{args.concept}"
        if nid not in G:
            cand = [n for n in G if G.nodes[n].get("node_type")=="Concept"
                    and args.concept.lower() in n.lower()]
            if not cand: sys.exit(f"Concept '{args.concept}' not found.")
            nid = cand[0]
        alphas = [u for u, v, d in G.in_edges(nid, data=True)
                  if d.get("relation") == "IMPLEMENTS"]
        keep = {nid, *alphas}
        for a in alphas:
            keep.update(G.successors(a)); keep.update(G.predecessors(a))
        which = f"concept={args.concept} ({len(alphas)} alphas + neighbors)"
    else:
        n_top = args.top or 50
        alphas = sorted(
            [(G.nodes[n].get("sharpe") if G.nodes[n].get("sharpe") is not None else -99, n)
             for n in G if G.nodes[n].get("node_type") == "Alpha"],
            reverse=True)[:n_top]
        top = [n for _, n in alphas]
        keep = set(top)
        for a in top:
            keep.update(G.successors(a)); keep.update(G.predecessors(a))
        which = f"top {n_top} alphas by Sharpe + neighbors"

    capped = False
    if args.max_nodes and len(keep) > args.max_nodes:
        keep = set(sorted(keep, key=lambda n: -G.degree(n))[:args.max_nodes])
        capped = True

    return G.subgraph(keep).copy(), which, capped


def build_graph_data(SG):
    """Return the dict that will be embedded into HTML and saved as JSON."""
    try:
        data = json_graph.node_link_data(SG, edges="links")
    except TypeError:
        data = json_graph.node_link_data(SG)
        if "links" not in data and "edges" in data:
            data["links"] = data.pop("edges")
    for n in data["nodes"]:
        ntype = n.get("node_type", "?")
        n["name"] = n.get("name", n["id"].split("::")[-1])
        size = 8
        if ntype == "Alpha":
            try:
                s = max(0.0, min(float(n.get("sharpe") or 0), 3.0))
                size = 7 + s * 7
            except Exception:
                pass
        n["size"] = size
        for k in ("expression", "hypothesis"):
            if n.get(k):
                n[k] = str(n[k])[:200]
    return data


def export_json(data, path):
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return len(data["nodes"]), len(data["links"])


HTML_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>WQ Brain Graph</title>
<style>
 body{margin:0;background:#0f0f1e;color:#eee;font-family:-apple-system,Segoe UI,sans-serif;overflow:hidden}
 svg{display:block;cursor:grab}
 .node{stroke:rgba(255,255,255,0.4);stroke-width:0.5px;cursor:pointer}
 .node.dim{opacity:0.08}
 .node:hover,.node.hl{stroke:#fff;stroke-width:2.5px}
 .node.rating-Good{stroke:#00ff88;stroke-width:3px;filter:drop-shadow(0 0 6px #00ff88)}
 .node.rating-Average{stroke:#ffd700;stroke-width:2.5px;filter:drop-shadow(0 0 4px #ffd700)}
 .node.ceiling{stroke:#ff6b35;stroke-width:2.5px;stroke-dasharray:3 2;filter:drop-shadow(0 0 5px #ff6b35)}
 .link{stroke:#888;stroke-opacity:0.32}
 .link.dim{stroke-opacity:0.04}
 .link.hl{stroke:#fff;stroke-opacity:0.85;stroke-width:1.2px}
 .label{font-size:9px;fill:rgba(255,255,255,0.78);pointer-events:none;text-shadow:0 0 3px #000}
 #legend,#tooltip,#search,#stats{background:rgba(20,20,40,0.92);border:1px solid #333;border-radius:6px;padding:8px 10px;font-size:11px;color:#eee}
 #legend{position:fixed;top:12px;left:12px;line-height:1.55}
 #legend .sw{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:50%;vertical-align:middle}
 #search{position:fixed;top:12px;right:12px;width:230px;font-size:12px;outline:none}
 #stats{position:fixed;bottom:10px;left:12px;font-size:10px;color:#aaa}
 #tooltip{position:fixed;display:none;max-width:360px;pointer-events:none;z-index:99;line-height:1.5}
 code{background:#222;padding:1px 4px;border-radius:3px;font-size:10px;color:#9af}
</style></head><body>
<svg></svg>
<div id="legend"><strong>WQ Brain Graph</strong><br><br>
 <div><span class="sw" style="background:#4A90E2"></span>Alpha &nbsp;<em style="color:#888">size=Sharpe</em></div>
 <div><span class="sw" style="background:#7ED321"></span>Concept</div>
 <div><span class="sw" style="background:#F5A623"></span>Datafield</div>
 <div><span class="sw" style="background:#9B59B6"></span>Operator</div>
 <div><span class="sw" style="background:#1ABC9C"></span>Setting</div>
 <div><span class="sw" style="background:#E74C3C"></span>FailureMode</div>
 <div><span class="sw" style="background:#95A5A6"></span>Session</div>
 <hr style="border-color:#333;margin:6px 0">
 <div style="font-size:10px;color:#888;margin-bottom:4px"><b style="color:#ccc">Ring around alpha:</b></div>
 <div><span class="sw" style="background:#4A90E2;border:2px solid #00ff88;box-shadow:0 0 4px #00ff88"></span><b style="color:#00ff88">Good</b> rated alpha</div>
 <div><span class="sw" style="background:#4A90E2;border:2px solid #ffd700;box-shadow:0 0 4px #ffd700"></span><b style="color:#ffd700">Average</b> rated alpha</div>
 <div><span class="sw" style="background:#4A90E2;border:2px dashed #ff6b35;box-shadow:0 0 4px #ff6b35"></span><b style="color:#ff6b35">Ceiling</b> alpha (high Sharpe, blocked)</div>
 <hr style="border-color:#333;margin:6px 0">
 <span style="color:#888;font-size:10px">scroll=zoom · drag=move · click=highlight neighbors</span>
</div>
<input id="search" placeholder="search node name...">
<div id="stats"></div><div id="tooltip"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script id="graph-data" type="application/json">__GRAPH_JSON__</script>
<script>
const COLORS = __COLORS__;
const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select("svg").attr("width", W).attr("height", H);
const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1, 10]).on("zoom", e => g.attr("transform", e.transform)));

(function() {
  const data = JSON.parse(document.getElementById("graph-data").textContent);
  document.getElementById("stats").textContent =
    data.nodes.length + " nodes · " + data.links.length + " edges";

  const sim = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(70).strength(0.35))
    .force("charge", d3.forceManyBody().strength(-260).distanceMax(420))
    .force("center", d3.forceCenter(W/2, H/2))
    .force("collide", d3.forceCollide().radius(d => d.size + 3));

  const link = g.append("g").selectAll("line").data(data.links).enter()
    .append("line").attr("class", "link");
  const node = g.append("g").selectAll("circle").data(data.nodes).enter()
    .append("circle")
    .attr("class", d => "node" + (d.rating === "Good" ? " rating-Good"
                                  : d.rating === "Average" ? " rating-Average"
                                  : d.is_ceiling ? " ceiling" : ""))
    .attr("r", d => d.size)
    .attr("fill", d => COLORS[d.node_type] || "#bbb")
    .call(d3.drag()
      .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end",   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
  const label = g.append("g").selectAll("text").data(data.nodes).enter()
    .append("text").attr("class", "label").text(d => d.name);

  const tip = document.getElementById("tooltip");
  node.on("mouseover", (e, d) => {
    let h = `<strong style="color:${COLORS[d.node_type]}">${d.name}</strong> <span style="color:#888">(${d.node_type})</span>`;
    if (d.sharpe   != null) h += `<br>Sharpe: <b>${d.sharpe}</b>`;
    if (d.fitness  != null) h += ` · Fitness: ${d.fitness}`;
    if (d.status)           h += `<br>Status: ${d.status}`;
    if (d.rating)           h += ` · Rating: <b>${d.rating}</b>`;
    if (d.is_ceiling) {
      h += `<br><span style="color:#ff6b35">🚧 CEILING ALPHA</span>`;
      if (d.ceiling_blocked_by) h += `<br><span style="color:#ff6b35">Blocked by: ${d.ceiling_blocked_by}</span>`;
      if (d.ceiling_unblock_try) h += `<br><span style="color:#aaffaa">Try: ${d.ceiling_unblock_try}</span>`;
    }
    if (d.expression)       h += `<br><code>${d.expression}</code>`;
    if (d.hypothesis)       h += `<br><em style="color:#aaa">${d.hypothesis}</em>`;
    tip.innerHTML = h; tip.style.display = "block";
  }).on("mousemove", e => {
    tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";
  }).on("mouseout", () => tip.style.display = "none")
    .on("click", (e, d) => {
      const nbrs = new Set([d.id]);
      data.links.forEach(l => {
        if (l.source.id === d.id) nbrs.add(l.target.id);
        if (l.target.id === d.id) nbrs.add(l.source.id);
      });
      node.classed("hl", n => n.id === d.id).classed("dim", n => !nbrs.has(n.id));
      link.classed("hl", l => l.source.id === d.id || l.target.id === d.id)
          .classed("dim", l => !(l.source.id === d.id || l.target.id === d.id));
    });
  svg.on("dblclick.zoom", null);
  svg.on("dblclick", () => { node.classed("hl", false).classed("dim", false);
                              link.classed("hl", false).classed("dim", false); });

  sim.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x + d.size + 2).attr("y", d => d.y + 3);
  });

  // Stop physics after initial layout converges
  setTimeout(() => { sim.stop(); }, 6000);

  document.getElementById("search").addEventListener("input", e => {
    const q = e.target.value.toLowerCase();
    node.classed("dim", d => q && !d.name.toLowerCase().includes(q));
    label.style("opacity", d => !q || d.name.toLowerCase().includes(q) ? 1 : 0.1);
  });
})();
</script></body></html>"""


def explore_mode(G, args):
    SG, which, capped = filter_subgraph(G, args)
    data = build_graph_data(SG)
    n, e = export_json(data, JSON_OUT)

    # Embed JSON directly to avoid file:// CORS issues with d3.json()
    embedded = json.dumps(data, default=str).replace("</", "<\\/")
    html = (HTML_TEMPLATE
        .replace("__GRAPH_JSON__", embedded)
        .replace("__COLORS__", json.dumps(NODE_COLORS)))
    HTML_OUT.write_text(html, encoding="utf-8")

    print(f"Filter:    {which}")
    if capped: print(f"Cap:       {args.max_nodes} (dropped least-connected)")
    print(f"Nodes:     {n}")
    print(f"Edges:     {e}")
    print(f"JSON:      {JSON_OUT}  (also embedded in HTML)")
    print(f"HTML:      {HTML_OUT}")
    print("Opening in browser...")
    webbrowser.open(HTML_OUT.as_uri())


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--lineage", metavar="ALPHA_ID")
    p.add_argument("--explore", action="store_true")
    p.add_argument("--type")
    p.add_argument("--concept")
    p.add_argument("--top", type=int)
    p.add_argument("--max-nodes", type=int, default=500)
    args = p.parse_args()

    if not (args.lineage or args.explore or args.type or args.concept or args.top):
        p.print_help(); return

    G = load_graph()
    if args.lineage:
        lineage_mode(G, args.lineage)
    if args.explore or args.type or args.concept or args.top:
        explore_mode(G, args)


if __name__ == "__main__":
    main()

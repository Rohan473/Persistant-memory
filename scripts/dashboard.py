"""
Local Streamlit dashboard for the WQ Brain knowledge graph.

Usage:
  pip install streamlit pandas
  streamlit run scripts/dashboard.py
  # opens browser at http://localhost:8501

Six pages, switchable from the sidebar:
  - Home        : summary stats, catalogue coverage, top untouched datasets
  - Alphas      : filterable / searchable alpha table with drill-down
  - Datafields  : catalogue browser with quality + availability filters
  - Concepts    : per-concept Sharpe distribution
  - Regimes     : yearly Sharpe by alpha + crisis-robust ranking
  - Sessions    : research-session browser
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
GRAPH_PATH = BASE / "graph" / "graph.gpickle"
SESSIONS_DIR = BASE / "private" / "nodes" / "sessions"

st.set_page_config(page_title="WQB Knowledge Graph", layout="wide")


# ── loaders ──────────────────────────────────────────────────────────────────

def _graph_mtime() -> float:
    """Used as a cache key so any rebuild of graph.gpickle invalidates caches."""
    return GRAPH_PATH.stat().st_mtime if GRAPH_PATH.exists() else 0.0


@st.cache_resource
def load_graph(mtime: float):
    if not GRAPH_PATH.exists():
        return None
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def get_graph():
    return load_graph(_graph_mtime())


@st.cache_data
def alphas_df(mtime: float = None):
    if mtime is None:
        mtime = _graph_mtime()
    G = get_graph()
    if G is None:
        return pd.DataFrame()
    rows = []
    for n in G.nodes:
        d = G.nodes[n]
        if d.get("node_type") != "Alpha":
            continue
        concepts, datafields, operators = [], [], []
        for _, tgt, ed in G.out_edges(n, data=True):
            tn = G.nodes[tgt].get("name") or ""
            rel = ed.get("relation")
            if   rel == "IMPLEMENTS": concepts.append(tn)
            elif rel == "USES":       datafields.append(tn)
            elif rel == "APPLIES":    operators.append(tn)
        rows.append({
            "id":          d.get("name"),
            "sharpe":      d.get("sharpe"),
            "fitness":     d.get("fitness"),
            "turnover":    d.get("turnover"),
            "returns":     d.get("returns"),
            "drawdown":    d.get("drawdown"),
            "status":      d.get("status"),
            "rating":      d.get("rating"),
            "universe":    d.get("universe"),
            "concepts":    ", ".join(concepts),
            "datafields":  ", ".join(datafields[:5]),
            "operators":   ", ".join(operators[:5]),
            "hypothesis":  d.get("hypothesis", ""),
            "expression":  d.get("expression", ""),
            "is_ceiling":  bool(d.get("is_ceiling")),
        })
    return pd.DataFrame(rows)


@st.cache_data
def datafields_df(mtime: float = None):
    if mtime is None: mtime = _graph_mtime()
    G = get_graph()
    if G is None:
        return pd.DataFrame()
    rows = []
    for n in G.nodes:
        d = G.nodes[n]
        if d.get("node_type") != "Datafield":
            continue
        rows.append({
            "name":         d.get("name"),
            "dataset":      d.get("dataset_id"),
            "dataset_name": d.get("dataset_name"),
            "category":     d.get("category_name"),
            "subcategory":  d.get("subcategory_name"),
            "frequency":    d.get("frequency"),
            "coverage":     d.get("coverage"),
            "users":        d.get("user_count"),
            "alphas":       d.get("alpha_count_global"),
            "availability": d.get("availability"),
            "description":  d.get("description") or "",
        })
    return pd.DataFrame(rows)


@st.cache_data
def yearly_df(mtime: float = None):
    if mtime is None: mtime = _graph_mtime()
    G = get_graph()
    if G is None:
        return pd.DataFrame()
    rows = []
    for n in G.nodes:
        d = G.nodes[n]
        if d.get("node_type") != "YearPerformance":
            continue
        # find parent alpha
        parents = [u for u, _, ed in G.in_edges(n, data=True)
                   if ed.get("relation") == "HAS_YEAR"]
        parent_name = G.nodes[parents[0]].get("name") if parents else None
        rows.append({
            "alpha":    parent_name,
            "year":     d.get("year"),
            "sharpe":   d.get("sharpe"),
            "turnover": d.get("turnover"),
            "fitness":  d.get("fitness"),
            "returns":  d.get("returns"),
            "regime":   d.get("regime"),
        })
    return pd.DataFrame(rows)


@st.cache_data
def operators_df(mtime: float = None):
    if mtime is None: mtime = _graph_mtime()
    G = get_graph()
    if G is None:
        return pd.DataFrame()
    rows = []
    for n in G.nodes:
        d = G.nodes[n]
        if d.get("node_type") != "Operator":
            continue
        rows.append({
            "name":         d.get("name"),
            "category":     d.get("category"),
            "level":        d.get("level"),
            "definition":   d.get("definition"),
            "description":  d.get("description"),
            "availability": d.get("availability"),
        })
    return pd.DataFrame(rows)


def list_sessions():
    rows = []
    if not SESSIONS_DIR.exists():
        return pd.DataFrame()
    for p in sorted(SESSIONS_DIR.glob("sess_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append({
                "id":      d.get("id"),
                "title":   d.get("title"),
                "start":   d.get("start", "")[:19],
                "end":     d.get("end", "")[:19] if d.get("end") else "(open)",
                "events":  len(d.get("events", [])),
                "alphas":  ", ".join(d.get("alphas_touched", [])),
                "note":    d.get("conclusion") or "",
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ── pages ────────────────────────────────────────────────────────────────────

def page_home():
    st.title("WQ Brain Knowledge Graph")
    G = get_graph()
    if G is None:
        st.warning("Graph not built yet. Run `python scripts/build_graph.py` first.")
        return

    a = alphas_df(_graph_mtime())
    df = datafields_df(_graph_mtime())
    op = operators_df(_graph_mtime())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alphas", len(a))
    c2.metric("Datafields", len(df))
    c3.metric("Operators", len(op))
    c4.metric("Submitted alphas", int((a["status"] == "submitted").sum()))

    c1, c2, c3 = st.columns(3)
    df_used = int((df["availability"] == "used").sum())
    c1.metric("Datafields used / catalogued", f"{df_used} / {len(df)}",
              f"{100*df_used/max(len(df),1):.2f}% explored")
    op_used = int((op["availability"] == "used").sum())
    c2.metric("Operators used / catalogued", f"{op_used} / {len(op)}",
              f"{100*op_used/max(len(op),1):.2f}% explored")
    best = a.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
    c3.metric("Best Sharpe", f"{best.iloc[0]['sharpe']:.2f}" if len(best) else "n/a",
              best.iloc[0]["id"] if len(best) else "")

    st.markdown("### Catalogue coverage by dataset")
    ds = (df.groupby("dataset")
            .agg(used=("availability", lambda s: int((s == "used").sum())),
                 total=("name", "count"))
            .reset_index()
            .sort_values("used", ascending=False))
    ds["pct"] = (100 * ds["used"] / ds["total"]).round(2)
    st.dataframe(ds, use_container_width=True, hide_index=True)

    st.markdown("### Top 10 alphas by Sharpe")
    st.dataframe(
        best[["id", "sharpe", "fitness", "turnover", "status", "concepts", "hypothesis"]].head(10),
        use_container_width=True, hide_index=True,
    )


def page_alphas():
    st.title("Alphas")
    a = alphas_df(_graph_mtime())
    if a.empty:
        st.warning("No alphas in graph.")
        return

    c1, c2, c3, c4 = st.columns(4)
    status = c1.multiselect("Status", sorted(a["status"].dropna().unique()),
                            default=["submitted", "iterating", "rejected"])
    universe = c2.multiselect("Universe", sorted(a["universe"].dropna().unique()),
                              default=sorted(a["universe"].dropna().unique()))
    min_sharpe = c3.number_input("Min Sharpe", value=-99.0, step=0.5)
    text = c4.text_input("Search expression / hypothesis", "")

    filt = a[a["status"].isin(status) & a["universe"].isin(universe)]
    filt = filt[filt["sharpe"].fillna(-99) >= min_sharpe]
    if text:
        m = (filt["expression"].str.contains(text, case=False, na=False) |
             filt["hypothesis"].str.contains(text, case=False, na=False))
        filt = filt[m]
    st.caption(f"Showing {len(filt)} of {len(a)}")
    st.dataframe(
        filt.sort_values("sharpe", ascending=False)[
            ["id", "sharpe", "fitness", "turnover", "returns", "status",
             "rating", "universe", "concepts", "hypothesis"]
        ],
        use_container_width=True, hide_index=True, height=420,
    )

    sel = st.selectbox("Drill-down alpha", [""] + list(filt["id"].dropna()))
    if sel:
        row = a[a["id"] == sel].iloc[0]
        st.subheader(sel)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", f"{row['sharpe']:.2f}" if pd.notna(row["sharpe"]) else "n/a")
        c2.metric("Fitness", f"{row['fitness']:.2f}" if pd.notna(row["fitness"]) else "n/a")
        c3.metric("Turnover", f"{row['turnover']:.2f}%" if pd.notna(row["turnover"]) else "n/a")
        st.code(row["expression"] or "(no expression)")
        if row["hypothesis"]:
            st.write(f"**Hypothesis:** {row['hypothesis']}")
        st.write(f"**Concepts:** {row['concepts'] or '—'}")
        st.write(f"**Datafields:** {row['datafields'] or '—'}")
        st.write(f"**Operators:** {row['operators'] or '—'}")
        yr = yearly_df(_graph_mtime())
        yr = yr[yr["alpha"] == sel].sort_values("year")
        if not yr.empty:
            st.markdown("**Year-by-year:**")
            st.dataframe(yr.drop(columns=["alpha"]), use_container_width=True, hide_index=True)


def page_datafields():
    st.title("Datafields")
    df = datafields_df(_graph_mtime())
    if df.empty:
        st.warning("No datafields in graph.")
        return

    c1, c2, c3, c4 = st.columns(4)
    ds = c1.multiselect("Dataset", sorted(df["dataset"].dropna().unique()))
    avail = c2.multiselect("Availability", ["used", "available"],
                           default=["used", "available"])
    min_cov = c3.slider("Min coverage", 0.0, 1.0, 0.0, step=0.05)
    text = c4.text_input("Search name / description")

    filt = df[df["availability"].isin(avail)]
    if ds:    filt = filt[filt["dataset"].isin(ds)]
    filt = filt[filt["coverage"].fillna(0) >= min_cov]
    if text:
        m = (filt["name"].str.contains(text, case=False, na=False) |
             filt["description"].str.contains(text, case=False, na=False))
        filt = filt[m]
    st.caption(f"Showing {len(filt)} of {len(df)}")
    st.dataframe(
        filt.sort_values("users", ascending=False)[
            ["name", "dataset", "category", "frequency", "coverage",
             "users", "alphas", "availability", "description"]
        ],
        use_container_width=True, hide_index=True, height=500,
    )


def page_concepts():
    st.title("Concepts")
    a = alphas_df(_graph_mtime())
    if a.empty:
        st.warning("No alphas in graph.")
        return
    rows = []
    for _, r in a.iterrows():
        for c in (r["concepts"] or "").split(","):
            c = c.strip()
            if not c:
                continue
            rows.append({"concept": c, "sharpe": r["sharpe"], "alpha": r["id"]})
    cf = pd.DataFrame(rows)
    if cf.empty:
        st.warning("No concept edges.")
        return
    agg = (cf.groupby("concept")
             .agg(count=("alpha", "count"),
                  mean_sharpe=("sharpe", "mean"),
                  max_sharpe=("sharpe", "max"))
             .reset_index()
             .sort_values("count", ascending=False))
    st.dataframe(agg, use_container_width=True, hide_index=True)


def page_regimes():
    st.title("Regimes")
    yr = yearly_df(_graph_mtime())
    if yr.empty:
        st.warning("No YearPerformance nodes — rebuild the graph.")
        return
    c1, c2 = st.columns(2)
    regimes = c1.multiselect("Regime", sorted(yr["regime"].dropna().unique()),
                             default=sorted(yr["regime"].dropna().unique()))
    years = c2.multiselect("Year", sorted(yr["year"].dropna().unique()),
                           default=sorted(yr["year"].dropna().unique()))
    filt = yr[yr["regime"].isin(regimes) & yr["year"].isin(years)]

    st.markdown("### Sharpe heatmap (alpha × year)")
    pivot = filt.pivot_table(index="alpha", columns="year", values="sharpe", aggfunc="mean")
    st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", axis=None),
                 use_container_width=True, height=480)

    st.markdown("### Crisis-robust alphas (avg Sharpe in 2008/2020/2022)")
    crisis = yr[yr["year"].isin([2008, 2020, 2022])]
    agg = (crisis.groupby("alpha")
                 .agg(avg_sharpe=("sharpe", "mean"),
                      years=("year", "count"))
                 .reset_index())
    agg = agg[agg["years"] >= 2].sort_values("avg_sharpe", ascending=False)
    st.dataframe(agg, use_container_width=True, hide_index=True)


def page_sessions():
    st.title("Research sessions")
    sf = list_sessions()
    if sf.empty:
        st.info("No sessions yet. Start one with `python scripts/session.py start \"title\"`.")
        return
    st.dataframe(sf, use_container_width=True, hide_index=True)
    sel = st.selectbox("Open session", [""] + list(sf["id"]))
    if sel:
        path = SESSIONS_DIR / f"{sel}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            st.subheader(data.get("title", sel))
            st.caption(f"{data.get('start','')} → {data.get('end','(open)')}")
            if data.get("conclusion"):
                st.success(f"Conclusion: {data['conclusion']}")
            evs = data.get("events", [])
            if evs:
                st.markdown(f"**Events ({len(evs)}):**")
                st.dataframe(pd.DataFrame(evs)[["time", "type", "summary"]],
                             use_container_width=True, hide_index=True)


# ── router ───────────────────────────────────────────────────────────────────

PAGES = {
    "Home":       page_home,
    "Alphas":     page_alphas,
    "Datafields": page_datafields,
    "Concepts":   page_concepts,
    "Regimes":    page_regimes,
    "Sessions":   page_sessions,
}

with st.sidebar:
    st.markdown("### Navigate")
    choice = st.radio("page", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Graph: `{GRAPH_PATH.relative_to(BASE)}`")
    st.caption("Rebuild with `python scripts/build_graph.py`")

PAGES[choice]()

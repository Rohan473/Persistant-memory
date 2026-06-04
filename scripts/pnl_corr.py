"""
Local PnL self-correlation — reproduce WQB self-corr from daily PnL vectors.

WQB self-correlation IS the Pearson correlation of two alphas' daily PnL series.
This pulls /alphas/{id}/recordsets/pnl (with retry; the endpoint is async and often
returns an empty body until ready) and computes correlation locally — so we can
predict self-corr BEFORE submission, against any chosen set of alphas.

Subcommands:
  pnl <alpha_id>                 fetch + cache one alpha's daily PnL, print summary
  validate <alpha_id>            recompute every WQB self-corr pair locally, compare
  corr <alpha_id> <id2> [id3..]  local Pearson corr of alpha_id vs each other id
  cache-list                     show what's cached

Cache: memory_layer/pnl_cache/<remote_id>.json  ({date: daily_pnl})
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "memory_layer" / "pnl_cache"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


brain_api = _load("brain_api", "memory_layer/brain_api.py")


def _get_with_retry(client, path, tries=12, delay=5.0):
    """GET an async recordset endpoint; retry while body is empty/non-JSON.

    Returns None (no crash) on 404 / HTTP errors — many node remote_ids are
    sim-ids from ERRORed sims that have no /recordsets/pnl resource.
    """
    for _ in range(tries):
        try:
            r = client._request("GET", path)
        except Exception:
            return None  # 404 or not-an-alpha: permanent, don't retry
        if r.status_code == 200 and r.text.strip():
            try:
                return r.json()
            except Exception:
                pass
        time.sleep(delay)
    return None


def fetch_pnl(client, aid, use_cache=True, tries=12, delay=5.0):
    """Return {date: daily_pnl} for one alpha (cached). Auto-diffs if cumulative."""
    cf = CACHE / f"{aid}.json"
    if use_cache and cf.exists():
        return {k: float(v) for k, v in json.loads(cf.read_text()).items()}
    d = _get_with_retry(client, f"/alphas/{aid}/recordsets/pnl", tries=tries, delay=delay)
    if not d or "records" not in d:
        return None
    recs = [(row[0], float(row[1])) for row in d["records"] if row[1] is not None]
    recs.sort(key=lambda x: x[0])
    vals = [v for _, v in recs]
    # WQB /recordsets/pnl is the CUMULATIVE PnL curve; correlate daily increments.
    # Detect cumulative by scale: a running total sits far from zero relative to its
    # day-to-day step size; a daily series oscillates around ~0.
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    mean_abs_step = (sum(abs(x) for x in diffs) / len(diffs)) if diffs else 0.0
    mean_level = abs(sum(vals) / len(vals))
    is_cumulative = mean_abs_step > 0 and mean_level > 3 * mean_abs_step
    if is_cumulative:
        daily = {recs[0][0]: vals[0]}
        for i in range(1, len(recs)):
            daily[recs[i][0]] = vals[i] - vals[i - 1]
    else:
        daily = dict(recs)
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(daily))
    return daily


def pearson(a_map, b_map, min_overlap=60):
    common = sorted(set(a_map) & set(b_map))
    n = len(common)
    if n < min_overlap:
        return None, n
    a = [a_map[d] for d in common]
    b = [b_map[d] for d in common]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    if va == 0 or vb == 0:
        return None, n
    return cov / (va * vb), n


def cmd_pnl(client, args):
    p = fetch_pnl(client, args.alpha_id, use_cache=not args.refresh)
    if not p:
        print(f"no PnL for {args.alpha_id} (endpoint empty after retries)")
        return
    vals = list(p.values())
    print(f"{args.alpha_id}: {len(p)} daily points  "
          f"sum={sum(vals):,.0f}  mean={sum(vals)/len(vals):,.1f}  "
          f"min={min(vals):,.0f}  max={max(vals):,.0f}")
    print(f"  cached: {CACHE / (args.alpha_id + '.json')}")


def cmd_corr(client, args):
    base = fetch_pnl(client, args.alpha_id, use_cache=not args.refresh)
    if not base:
        print(f"no PnL for {args.alpha_id}"); return
    print(f"\nLocal PnL corr vs {args.alpha_id}:")
    rows = []
    for other in args.others:
        op = fetch_pnl(client, other, use_cache=not args.refresh)
        if not op:
            print(f"  {other}: (no PnL)"); continue
        c, n = pearson(base, op)
        rows.append((other, c, n))
    for other, c, n in sorted(rows, key=lambda r: -(r[1] or -9)):
        print(f"  {other}: corr={c:+.4f}  (n={n})" if c is not None else f"  {other}: n/a")


def cmd_validate(client, args):
    """Recompute each WQB self-corr pair locally and compare."""
    aid = args.alpha_id
    sc = _get_with_retry(client, f"/alphas/{aid}/correlations/self")
    if not sc or "records" not in sc:
        print(f"no WQB self-corr for {aid} (still pending?)"); return
    base = fetch_pnl(client, aid, use_cache=not args.refresh)
    if not base:
        print(f"no PnL for candidate {aid}"); return
    print(f"\nValidating local PnL corr vs WQB self-corr for {aid}")
    print(f"{'prod_id':<12} {'WQB':>8} {'local':>8} {'diff':>7} {'n':>5}")
    print("-" * 44)
    diffs = []
    for row in sorted(sc["records"], key=lambda r: -(r[5] if isinstance(r[5], (int, float)) else -9)):
        pid, wqb = row[0], row[5]
        if not isinstance(wqb, (int, float)):
            continue
        op = fetch_pnl(client, pid, use_cache=not args.refresh)
        if not op:
            print(f"{pid:<12} {wqb:>8.4f} {'(no pnl)':>8}")
            continue
        loc, n = pearson(base, op)
        if loc is None:
            print(f"{pid:<12} {wqb:>8.4f} {'n/a':>8} {'':>7} {n:>5}")
            continue
        diffs.append(abs(loc - wqb))
        print(f"{pid:<12} {wqb:>8.4f} {loc:>8.4f} {loc-wqb:>+7.4f} {n:>5}")
    if diffs:
        print("-" * 44)
        print(f"mean |WQB-local| = {sum(diffs)/len(diffs):.4f}   max = {max(diffs):.4f}")
        print("MATCH (method reproduces WQB)" if max(diffs) < 0.05
              else "MISMATCH - check daily/cumulative handling or corr window")


import re


def _node_index():
    """Map remote_alpha_id -> (alpha_stem, status) from the alpha nodes."""
    idx = {}
    nd = BASE / "private" / "nodes" / "alphas"
    for f in nd.glob("*.md"):
        txt = f.read_text(errors="replace")
        m = re.search(r"^remote_alpha_id:\s*(.+)$", txt, re.M)
        if not m:
            continue
        rid = m.group(1).strip().strip("'\"")
        if not rid or rid.lower() in ("null", "none"):
            continue
        sm = re.search(r"^status:\s*(.+)$", txt, re.M)
        status = sm.group(1).strip().strip("'\"") if sm else ""
        idx[rid] = (f.stem, status)
    return idx


def novelty_rows(client, target_rid, use_cache=True, tries=12, delay=5.0):
    """Local PnL corr of target vs every cached alpha. Returns (sorted_rows, base)."""
    base = fetch_pnl(client, target_rid, use_cache=use_cache, tries=tries, delay=delay)
    if not base:
        return None, None
    rows = []
    for cf in CACHE.glob("*.json"):
        rid = cf.stem
        if rid == target_rid:
            continue
        op = fetch_pnl(client, rid, use_cache=True)
        if not op:
            continue
        c, n = pearson(base, op)
        if c is not None:
            rows.append((rid, c, n))
    rows.sort(key=lambda r: -r[1])
    return rows, base


def novelty_summary(client, target_rid, tries=2, delay=4.0):
    """Best-effort novelty dict for programmatic use (e.g. sweep.py post-sim)."""
    rows, base = novelty_rows(client, target_rid, use_cache=False, tries=tries, delay=delay)
    if rows is None:
        return None
    mx = rows[0][1] if rows else 0.0
    over5 = sum(1 for _, c, _ in rows if c > 0.5)
    over7 = sum(1 for _, c, _ in rows if c > 0.7)
    verdict = "NOVEL" if mx < 0.5 else ("CROWDED" if over5 >= 3 else "BORDERLINE")
    top = rows[0] if rows else (None, 0.0, 0)
    return {"max": mx, "novelty": 1 - mx, "over5": over5, "over7": over7,
            "verdict": verdict, "n_book": len(rows), "nearest": top[0]}


def cmd_backfill(client, args):
    idx = _node_index()
    items = list(idx.items())
    if args.status:
        want = {s.strip().lower() for s in args.status.split(",")}
        items = [(r, v) for r, v in items if v[1].lower() in want]
    if args.limit:
        items = items[:args.limit]
    have = len(list(CACHE.glob("*.json"))) if CACHE.exists() else 0
    print(f"backfilling {len(items)} alphas (cache has {have})")
    ok = skip = fail = 0
    for rid, (stem, status) in items:
        if (CACHE / f"{rid}.json").exists() and not args.refresh:
            skip += 1
            continue
        p = fetch_pnl(client, rid, use_cache=False, tries=3, delay=3.0)
        if p:
            ok += 1
        else:
            fail += 1
        if (ok + fail) % 25 == 0 and (ok + fail) > 0:
            print(f"  ... {ok} ok, {fail} fail, {skip} skip", flush=True)
    total = len(list(CACHE.glob("*.json")))
    print(f"DONE: {ok} newly cached, {skip} already, {fail} failed. cache total = {total}")


def active_alpha_ids(client, statuses=("ACTIVE",)):
    """Real prod set: ids from /users/self/alphas with status in `statuses`."""
    want = {s.upper() for s in statuses}
    ids, offset = set(), 0
    while True:
        r = client._request("GET", "/users/self/alphas",
                             params={"limit": 100, "offset": offset})
        try:
            d = r.json()
        except Exception:
            break
        rows = d.get("results", []) if isinstance(d, dict) else d
        if not rows:
            break
        for rec in rows:
            if str(rec.get("status", "")).upper() in want:
                ids.add(rec.get("id"))
        offset += len(rows)
        count = d.get("count") if isinstance(d, dict) else None
        if count is not None and offset >= count:
            break
    return ids


def cmd_novelty(client, args):
    idx = _node_index()
    base = fetch_pnl(client, args.alpha_id, use_cache=not args.refresh)
    if base is None:
        print(f"no PnL for {args.alpha_id} (endpoint empty after retries)")
        return
    if args.portfolio:
        active = active_alpha_ids(client)
        active.discard(args.alpha_id)
        rows = []
        for rid in active:
            op = fetch_pnl(client, rid, use_cache=True) or fetch_pnl(client, rid, use_cache=False, tries=2)
            if not op:
                continue
            c, n = pearson(base, op)
            if c is not None:
                rows.append((rid, c, n))
        rows.sort(key=lambda r: -r[1])
        print(f"(portfolio scope: {len(rows)} live ACTIVE alphas from /users/self/alphas)")
    else:
        rows = []
        for cf in CACHE.glob("*.json"):
            rid = cf.stem
            if rid == args.alpha_id:
                continue
            op = fetch_pnl(client, rid, use_cache=True)
            if not op:
                continue
            c, n = pearson(base, op)
            if c is not None:
                rows.append((rid, c, n))
        rows.sort(key=lambda r: -r[1])
        if args.status:
            want = {s.strip().lower() for s in args.status.split(",")}
            rows = [(r, c, n) for r, c, n in rows if idx.get(r, ("", ""))[1].lower() in want]
            print(f"(scoped to status in {sorted(want)} - {len(rows)} alphas)")
    mx = rows[0][1] if rows else 0.0
    over5 = sum(1 for _, c, _ in rows if c > 0.5)
    over7 = sum(1 for _, c, _ in rows if c > 0.7)
    verdict = "NOVEL" if mx < 0.5 else ("CROWDED" if over5 >= 3 else "BORDERLINE")
    stem = idx.get(args.alpha_id, ("?", ""))[0]
    print(f"\nNovelty for {args.alpha_id} ({stem}) vs {len(rows)} cached alphas")
    print(f"  max_corr = {mx:.4f}   novelty(1-max) = {1-mx:.4f}   #>0.5 = {over5}   #>0.7 = {over7}")
    note = "  (>=3 alphas >0.5 -> negative-points risk)" if over5 >= 3 else ""
    print(f"  verdict: {verdict}{note}")
    print("  nearest neighbors:")
    for rid, c, n in rows[:args.top]:
        s, st = idx.get(rid, ("?", ""))
        print(f"    {c:+.4f}  {rid}  {s} [{st}]")


def cmd_cache_list(client, args):
    if not CACHE.exists():
        print("(cache empty)"); return
    files = sorted(CACHE.glob("*.json"))
    print(f"{len(files)} cached PnL series in {CACHE}")
    for f in files[:50]:
        print(f"  {f.stem}")


def main():
    ap = argparse.ArgumentParser(description="Local PnL self-correlation")
    ap.add_argument("--refresh", action="store_true", help="bypass cache, re-fetch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("pnl"); sp.add_argument("alpha_id"); sp.set_defaults(fn=cmd_pnl)
    sp = sub.add_parser("validate"); sp.add_argument("alpha_id"); sp.set_defaults(fn=cmd_validate)
    sp = sub.add_parser("corr"); sp.add_argument("alpha_id"); sp.add_argument("others", nargs="+"); sp.set_defaults(fn=cmd_corr)
    sp = sub.add_parser("backfill"); sp.add_argument("--status", default=None,
        help="comma-list to filter by node status, e.g. submitted,active"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(fn=cmd_backfill)
    sp = sub.add_parser("novelty"); sp.add_argument("alpha_id"); sp.add_argument("--top", type=int, default=8)
    sp.add_argument("--status", default=None, help="scope comparison set by node frontmatter status (approx)")
    sp.add_argument("--portfolio", action="store_true", help="scope to the REAL live ACTIVE set from /users/self/alphas (exact portfolio-additivity)")
    sp.set_defaults(fn=cmd_novelty)
    sp = sub.add_parser("cache-list"); sp.set_defaults(fn=cmd_cache_list)
    args = ap.parse_args()
    client = brain_api.BrainAPIClient.from_disk()
    args.fn(client, args)


if __name__ == "__main__":
    main()

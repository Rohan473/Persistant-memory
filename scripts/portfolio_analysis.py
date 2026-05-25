"""
Portfolio diversification analysis across all submitted (ACTIVE) alphas.

Pulls pairwise self-correlations from the WQ Brain API, builds the full
N x N correlation matrix, and reports:
  - The matrix itself (saved to CSV)
  - Effective number of independent bets: N_eff = (sum lambda)^2 / sum lambda^2
  - Dominant risk modes (top-k eigenvectors and their alpha loadings)
  - Redundancy flags: alphas whose max pairwise correlation exceeds a threshold

Usage:
  python scripts/portfolio_analysis.py                   # all ACTIVE alphas
  python scripts/portfolio_analysis.py --threshold 0.6   # custom redundancy cutoff
  python scripts/portfolio_analysis.py --csv out.csv     # save matrix to a path
  python scripts/portfolio_analysis.py id1 id2 id3 ...   # restrict to specific IDs
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Project import path (assumes torch/torchvision are aligned; see scripts/get_alpha.py
# for the importlib fallback if memory_layer/__init__.py ever breaks again).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory_layer.brain_api import BrainAPIClient, BrainAuthError


def list_active_alphas(client):
    """Return list of dicts for all ACTIVE alphas (paginates if needed)."""
    out, offset, page = [], 0, 100
    while True:
        r = client._request("GET", f"/users/self/alphas?limit={page}&offset={offset}&status=ACTIVE")
        d = r.json()
        rows = d.get("results", [])
        out.extend(rows)
        if len(rows) < page or offset + len(rows) >= (d.get("count") or 0):
            break
        offset += len(rows)
    return out


def fetch_self_correlations(client, alpha_id, max_wait=60.0):
    """Poll /alphas/{id}/correlations/self until a non-empty body is returned."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = client._request("GET", f"/alphas/{alpha_id}/correlations/self")
        if r.text.strip():
            return r.json()
        retry = float(r.headers.get("Retry-After") or 1.5)
        time.sleep(min(retry, 3.0))
    raise TimeoutError(f"correlations/self for {alpha_id} never populated")


def build_matrix(client, alpha_ids, verbose=True):
    """Build symmetric correlation matrix by querying each alpha's correlations."""
    n = len(alpha_ids)
    idx = {a: i for i, a in enumerate(alpha_ids)}
    raw = np.full((n, n), np.nan)
    np.fill_diagonal(raw, 1.0)

    for i, a in enumerate(alpha_ids):
        if verbose:
            print(f"  [{i+1}/{n}] {a} ...", end=" ", flush=True)
        try:
            data = fetch_self_correlations(client, a)
        except Exception as e:
            if verbose:
                print(f"FAIL ({e})")
            continue
        cols = [p["name"] for p in data["schema"]["properties"]]
        id_col = cols.index("id")
        corr_col = cols.index("correlation")
        hits = 0
        for rec in data.get("records", []):
            other = rec[id_col]
            corr = rec[corr_col]
            if other in idx and corr is not None:
                j = idx[other]
                # If both A->B and B->A are reported, average them (should match closely)
                existing = raw[i, j]
                raw[i, j] = corr if np.isnan(existing) else 0.5 * (existing + corr)
                existing2 = raw[j, i]
                raw[j, i] = corr if np.isnan(existing2) else 0.5 * (existing2 + corr)
                hits += 1
        if verbose:
            print(f"got {hits} pairs (min={data.get('min')}, max={data.get('max')})")

    # Fill remaining NaN with 0 — WQB only reports non-trivial correlations,
    # so missing pairs are effectively below the reporting threshold.
    missing = int(np.isnan(raw).sum())
    raw = np.where(np.isnan(raw), 0.0, raw)
    return raw, missing


def effective_n(eigvals):
    """N_eff = (sum lambda)^2 / sum lambda^2 — participation ratio."""
    eigvals = np.clip(eigvals, 0, None)  # numerical noise can give tiny negatives
    s = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return float(s * s / s2) if s2 > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Portfolio diversification analysis")
    ap.add_argument("alpha_ids", nargs="*", help="Specific alpha IDs (default: all ACTIVE)")
    ap.add_argument("--threshold", type=float, default=0.6, help="Redundancy cutoff (default 0.6)")
    ap.add_argument("--csv", default="exports/portfolio_correlation_matrix.csv",
                    help="Path to save the correlation matrix CSV")
    ap.add_argument("--top-modes", type=int, default=3, help="Number of dominant modes to print")
    args = ap.parse_args()

    try:
        client = BrainAPIClient.from_disk()
    except BrainAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.alpha_ids:
        alpha_ids = args.alpha_ids
        meta = {a: {} for a in alpha_ids}
    else:
        print("Discovering ACTIVE alphas ...")
        rows = list_active_alphas(client)
        alpha_ids = [r["id"] for r in rows]
        meta = {r["id"]: r for r in rows}
        print(f"  found {len(alpha_ids)} ACTIVE alpha(s)")

    if len(alpha_ids) < 2:
        print("Need at least 2 alphas for portfolio analysis.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFetching pairwise correlations for {len(alpha_ids)} alphas ...")
    M, missing = build_matrix(client, alpha_ids)
    n = len(alpha_ids)
    print(f"\n{missing} of {n*n - n} off-diagonal cells unreported (filled with 0)")

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + alpha_ids)
        for i, a in enumerate(alpha_ids):
            w.writerow([a] + [f"{M[i,j]:.4f}" for j in range(n)])
    print(f"matrix saved -> {csv_path}")

    # ── Eigendecomposition ───────────────────────────────────────────────
    # Symmetrize to clean numerical asymmetry, then eigh.
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_eff = effective_n(eigvals)
    print()
    print("=" * 70)
    print(f"PORTFOLIO DIVERSIFICATION REPORT — {n} alphas")
    print("=" * 70)
    print(f"\nEffective number of independent bets (N_eff): {n_eff:.2f}")
    print(f"  -> ratio: {n_eff/n*100:.0f}% of nominal portfolio size")
    if n_eff / n < 0.5:
        print(f"  -> WARN: less than half of nominal — heavy concentration in shared risk modes")
    elif n_eff / n < 0.7:
        print(f"  -> moderate diversification — room to improve")
    else:
        print(f"  -> well-diversified")

    print(f"\nEigenvalue spectrum (descending):")
    total = eigvals.sum()
    cum = 0.0
    for i, lam in enumerate(eigvals):
        cum += max(lam, 0)
        share = max(lam, 0) / total * 100 if total > 0 else 0
        print(f"  mode {i+1:2d}: lambda={lam:7.4f}  variance share={share:5.1f}%  cum={cum/total*100:5.1f}%")

    # ── Dominant risk modes ──────────────────────────────────────────────
    print(f"\nTop {args.top_modes} dominant risk modes (alpha loadings, |w| > 0.25):")
    for k in range(min(args.top_modes, n)):
        v = eigvecs[:, k]
        if v[np.argmax(np.abs(v))] < 0:
            v = -v  # canonical sign: largest |loading| is positive
        loadings = sorted(
            [(alpha_ids[i], float(v[i])) for i in range(n)],
            key=lambda t: -abs(t[1]),
        )
        share = max(eigvals[k], 0) / total * 100 if total > 0 else 0
        print(f"\n  Mode {k+1} (lambda={eigvals[k]:.3f}, {share:.1f}% of variance):")
        for aid, w in loadings:
            if abs(w) >= 0.25:
                grade = meta.get(aid, {}).get("grade") or "?"
                sharpe = (meta.get(aid, {}).get("is") or {}).get("sharpe")
                print(f"    {aid:10s}  load={w:+.3f}  grade={grade:8s}  IS-sharpe={sharpe}")

    # ── Redundancy flags ─────────────────────────────────────────────────
    print(f"\nRedundancy flags (max pairwise correlation > {args.threshold}):")
    flagged = []
    for i in range(n):
        off = [(alpha_ids[j], M[i, j]) for j in range(n) if j != i]
        worst = max(off, key=lambda t: abs(t[1]))
        if abs(worst[1]) > args.threshold:
            flagged.append((alpha_ids[i], worst[0], worst[1]))
    if not flagged:
        print(f"  none — all pairs below {args.threshold}")
    else:
        flagged.sort(key=lambda t: -abs(t[2]))
        for a, b, c in flagged:
            grade_a = meta.get(a, {}).get("grade") or "?"
            print(f"  {a} <-> {b}   corr={c:+.3f}   (grade of {a}: {grade_a})")

    print()


if __name__ == "__main__":
    main()

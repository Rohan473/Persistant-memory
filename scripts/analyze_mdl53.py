"""
Rank model53 sweep results and propose the next 5 variants on the winning branch.

Reads alpha .md files in private/nodes/alphas/, filters those whose `hypothesis`
frontmatter starts with one of the four mdl53_* template tags, groups by branch,
scores by the project's stated priorities (Fitness >=1.15, Turnover <=45%, Sharpe
sweet spot ~1.55), and prints:
  1. Per-branch top alpha with metrics
  2. Overall winner across all 4 branches
  3. Five proposed follow-up variants on the winning branch

Usage:
  python scripts/analyze_mdl53.py
  python scripts/analyze_mdl53.py --since 2026-05-27
  python scripts/analyze_mdl53.py --prefix mdl53_                 # default
  python scripts/analyze_mdl53.py --top 3                         # show top-N per branch
  python scripts/analyze_mdl53.py --write-template                # emit next-variants as a template JSON
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ALPHAS_DIR = BASE / "private" / "nodes" / "alphas"
TEMPLATES_DIR = BASE / "private" / "templates"

# User's optimization priorities from project memory
TARGET_FITNESS = 1.15
MAX_TURNOVER = 45.0
SHARPE_SWEET_SPOT = 1.55  # midpoint of 1.4-1.7 range
SHARPE_MIN_FOR_SUBMIT = 1.25  # WQB's "Average" gate

BRANCHES = [
    "mdl53_pd_momentum",
    "mdl53_term_structure",
    "mdl53_model_dispersion",
    "mdl53_implied_vs_model",
]


# ---------- loading ----------

def load_alphas(prefix: str, since: str | None):
    import frontmatter
    out = []
    if not ALPHAS_DIR.exists():
        return out
    cutoff = datetime.fromisoformat(since).date() if since else None
    for f in sorted(ALPHAS_DIR.glob("alpha_*.md")):
        try:
            post = frontmatter.load(str(f))
        except Exception:
            continue
        m = post.metadata
        hyp = (m.get("hypothesis") or "").strip()
        if not hyp.startswith("["):
            continue
        tag = hyp.split("]", 1)[0].lstrip("[").strip()
        if not tag.startswith(prefix):
            continue
        if cutoff:
            try:
                created = datetime.fromisoformat(str(m.get("created", ""))).date()
                if created < cutoff:
                    continue
            except Exception:
                pass
        m["_branch"] = tag
        m["_file"] = f.name
        out.append(m)
    return out


# ---------- scoring ----------

def _num(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def score(alpha: dict) -> tuple[int, float, float]:
    """
    Return (tier, primary, secondary) — higher tier first, then primary desc, then secondary desc.
    Tier 3: passes user's stated criteria (fitness>=1.15, turnover<=45, sharpe>=1.25)
    Tier 2: submitted but missing one criterion
    Tier 1: any sim with usable metrics
    Tier 0: rejected / missing metrics
    """
    sharpe = _num(alpha.get("sharpe"))
    fitness = _num(alpha.get("fitness"))
    turnover = _num(alpha.get("turnover"))
    status = alpha.get("status", "")
    fm = alpha.get("failure_modes") or []

    if sharpe is None or fitness is None or turnover is None:
        return (0, -9e9, -9e9)

    # Tier check on user's hard criteria
    passes_fitness = fitness >= TARGET_FITNESS
    passes_turnover = turnover <= MAX_TURNOVER
    passes_sharpe = sharpe >= SHARPE_MIN_FOR_SUBMIT

    if status == "submitted" and passes_fitness and passes_turnover and passes_sharpe:
        tier = 3
        # within tier 3: closeness to sweet spot, fitness as tiebreaker
        primary = -abs(sharpe - SHARPE_SWEET_SPOT)
        secondary = fitness
    elif status == "submitted" or (not fm and sharpe > 0):
        tier = 2
        # composite that mirrors user's priorities
        primary = fitness - max(0, turnover - MAX_TURNOVER) / 100.0
        secondary = sharpe
    else:
        tier = 1
        primary = sharpe
        secondary = fitness

    return (tier, primary, secondary)


# ---------- presentation ----------

def fmt_row(a: dict) -> str:
    s = _num(a.get("sharpe"))
    f = _num(a.get("fitness"))
    t = _num(a.get("turnover"))
    status = a.get("status", "?")
    fm = a.get("failure_modes") or []
    fm_str = ",".join(fm) if fm else "-"
    return (f"  {a['_file']:<16} "
            f"Sharpe={s if s is None else f'{s:+.2f}':<6} "
            f"Fitness={f if f is None else f'{f:+.2f}':<6} "
            f"Turnover={'-' if t is None else f'{t:.1f}%':<7} "
            f"[{status}] fm={fm_str}\n"
            f"    {a.get('expression', '')}")


def print_branch_summary(branch: str, alphas: list, top_n: int):
    print(f"\n--- {branch} ({len(alphas)} alphas) ---")
    if not alphas:
        print("  (no results yet)")
        return None
    alphas.sort(key=score, reverse=True)
    for a in alphas[:top_n]:
        print(fmt_row(a))
    return alphas[0]


# ---------- follow-up variant proposals ----------

def propose_next(branch: str, winner: dict) -> list[str]:
    """
    5 paper-informed next variants per branch. Botha & Verster (2025) emphasize:
      - PD is a hazard, not a level: trade proportional / log changes, not raw deltas
      - Forward hazard h(t1→t2) ~ (PD_t2 − PD_t1) / (1 − PD_t1) is the right object
      - Scale-invariance matters: log-ratios beat raw differences on bounded PDs
    """
    if branch == "mdl53_pd_momentum":
        return [
            # Log-momentum: trade % change in PD, not absolute change.
            "group_rank(-ts_delta(log(annualized_pd_1_year + 0.000001), 66), industry)",
            # Proportional momentum: relative change scales across PD levels.
            "group_rank(-ts_delta(annualized_pd_1_year, 66) / (annualized_pd_1_year + 0.000001), industry)",
            # Cox-style hazard scaling per paper §3.4.
            "group_rank(-ts_delta(annualized_pd_1_year, 66) / (1 - annualized_pd_1_year), industry)",
            # Log-level long-window zscore (no delta noise).
            "group_rank(-ts_zscore(log(annualized_pd_1_year + 0.000001), 132), industry)",
            # Shorter horizon log-momentum (test if short-horizon PD has more signal).
            "group_rank(-ts_delta(log(annualized_pd_3_month + 0.000001), 66), industry)",
        ]
    if branch == "mdl53_term_structure":
        return [
            # Forward hazard between 1y and 2y — paper's recommended object.
            "group_rank((annualized_pd_2_year - annualized_pd_1_year) / (1 - annualized_pd_1_year), industry)",
            # Forward hazard between 1m and 1y (near-term marginal hazard).
            "group_rank((annualized_pd_1_year - annualized_pd_1_month) / (1 - annualized_pd_1_month), industry)",
            # Log-ratio curve slope — scale-invariant alternative to raw subtraction.
            "group_rank(log(annualized_pd_2_year + 0.000001) - log(annualized_pd_1_year + 0.000001), industry)",
            # Curve curvature (3-point): convexity in PD term structure.
            "group_rank(2 * annualized_pd_1_year - annualized_pd_1_month - annualized_pd_2_year, industry)",
            # Within-model forward hazard using uncrowded JC5 family.
            "group_rank((mdl53_jc5_2year - mdl53_jc5_1year) / (1 - mdl53_jc5_1year), industry)",
        ]
    if branch == "mdl53_model_dispersion":
        return [
            # Proportional disagreement: dispersion normalized by mean PD level.
            "group_rank(abs(mdl53_jc5_1year - mdl53_ms5_1year) / (mdl53_jc5_1year + mdl53_ms5_1year + 0.000001), industry)",
            # Log-ratio disagreement (scale-invariant).
            "group_rank(abs(log(mdl53_jc5_1year + 0.000001) - log(mdl53_ms5_1year + 0.000001)), industry)",
            # ts_zscore the dispersion — flag novel disagreement vs historical baseline.
            "group_rank(ts_zscore(abs(mdl53_jc5_1year - mdl53_ms5_1year), 22), industry)",
            # 3y horizon: longer-dated disagreement is structural, not noise.
            "group_rank(abs(mdl53_jc5_3year - mdl53_ms5_3year) / (mdl53_jc5_3year + mdl53_ms5_3year + 0.000001), industry)",
            # Cross-family (JM vs JC) at 1y — different model class.
            "group_rank(abs(mdl53_jc5_1year - mdl53_jm5_1year) / (mdl53_jc5_1year + mdl53_jm5_1year + 0.000001), industry)",
        ]
    if branch == "mdl53_implied_vs_model":
        return [
            # Log market vs log model — scale-invariant dissent.
            "group_rank(log(mdl53_implied_spreads + 0.000001) - log(annualized_pd_1_year + 0.000001), industry)",
            # Market spread vs forward hazard 1y→2y (paper-informed model side).
            "group_rank(zscore(mdl53_implied_spreads) - zscore((annualized_pd_2_year - annualized_pd_1_year) / (1 - annualized_pd_1_year)), industry)",
            # Use uncrowded expected_rating_value vs uncrowded JC5.
            "group_rank(zscore(expected_rating_value) - zscore(mdl53_jc5_1year), industry)",
            # ts_zscore the dissent — flag fresh divergence vs baseline.
            "group_rank(ts_zscore(zscore(mdl53_implied_spreads) - zscore(mdl53_jc5_1year), 22), industry)",
            # Term-structured dissent: market spread vs 2y PD.
            "group_rank(zscore(mdl53_implied_spreads) - zscore(mdl53_jc5_2year), industry)",
        ]
    return []


def write_next_template(branch: str, exprs: list[str]) -> Path:
    tmpl_id = f"{branch}_phase2"
    payload = {
        "id": tmpl_id,
        "description": (
            f"Phase-2 follow-up on the winning {branch} branch, informed by Botha & "
            "Verster (2025) on PD term-structure modeling: trade proportional/log PD "
            "changes (not raw deltas), use forward-hazard normalization (PD_t2 − PD_t1) "
            "/ (1 − PD_t1), and prefer log-ratios over raw subtractions on bounded PDs."
        ),
        "concepts": ["credit_risk", "phase2_followup", "forward_hazard", "log_pd"],
        "form": "{expr}",
        "slots": {
            "expr": {"type": "literal", "values": exprs},
        },
        "constraints": [],
        "settings": {
            "universe": "TOP1000",
            "delay": 1,
            "neutralization": "INDUSTRY",
            "truncation": 0.08,
        },
        "operators_hint": ["group_rank"],
    }
    out = TEMPLATES_DIR / f"{tmpl_id}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Analyze model53 sweep results")
    ap.add_argument("--prefix", default="mdl53_", help="Template-id prefix to filter on")
    ap.add_argument("--since", default=None, help="ISO date — only alphas created on/after")
    ap.add_argument("--top", type=int, default=2, help="Top-N per branch to print")
    ap.add_argument("--write-template", action="store_true",
                    help="Write phase-2 follow-up template for the winning branch")
    args = ap.parse_args()

    alphas = load_alphas(args.prefix, args.since)
    if not alphas:
        print(f"No alphas matching prefix '{args.prefix}'"
              f"{f' since {args.since}' if args.since else ''}.")
        print("Run the 4 sweeps first:")
        for b in BRANCHES:
            print(f"  python scripts/sweep.py run {b}")
        return

    by_branch: dict[str, list] = {b: [] for b in BRANCHES}
    for a in alphas:
        by_branch.setdefault(a["_branch"], []).append(a)

    print(f"\nLoaded {len(alphas)} alpha(s) across {sum(1 for v in by_branch.values() if v)} branch(es).")
    print(f"Priorities: Fitness >= {TARGET_FITNESS}, Turnover <= {MAX_TURNOVER}%, "
          f"Sharpe ~ {SHARPE_SWEET_SPOT} (gate: >= {SHARPE_MIN_FOR_SUBMIT}).")

    branch_winners: dict[str, dict] = {}
    for b in BRANCHES:
        w = print_branch_summary(b, by_branch.get(b, []), args.top)
        if w is not None:
            branch_winners[b] = w

    if not branch_winners:
        print("\nNo branch had a usable winner.")
        return

    # Overall winner across branches
    overall = max(branch_winners.items(), key=lambda kv: score(kv[1]))
    branch, winner = overall
    tier, p, s = score(winner)

    print("\n" + "=" * 70)
    print(f"OVERALL WINNER: {branch}  (tier {tier})")
    print("=" * 70)
    print(fmt_row(winner))

    fitness = _num(winner.get("fitness"))
    turnover = _num(winner.get("turnover"))
    sharpe = _num(winner.get("sharpe"))
    notes = []
    if fitness is not None and fitness < TARGET_FITNESS:
        notes.append(f"Fitness {fitness:.2f} is below target {TARGET_FITNESS} — phase 2 should focus on lifting it.")
    if turnover is not None and turnover > MAX_TURNOVER:
        notes.append(f"Turnover {turnover:.1f}% breaches {MAX_TURNOVER}% — try wider windows / less reactive forms.")
    if sharpe is not None and sharpe > 1.7:
        notes.append(f"Sharpe {sharpe:.2f} is above the sweet spot — risk of overfitting; widen / robust-check.")
    if sharpe is not None and 1.4 <= sharpe <= 1.7 and fitness and fitness >= TARGET_FITNESS:
        notes.append("In the sweet spot. Push for TOP500 port for IQC points after phase 2 lands.")
    for n in notes:
        print(f"  - {n}")

    print(f"\nProposed 5 follow-up variants on {branch}:")
    proposed = propose_next(branch, winner)
    for i, e in enumerate(proposed, 1):
        print(f"  [{i}] {e}")

    if args.write_template:
        path = write_next_template(branch, proposed)
        print(f"\nWrote phase-2 template: {path.relative_to(BASE)}")
        print(f"Run with: python scripts/sweep.py run {branch}_phase2")


if __name__ == "__main__":
    main()

"""
Strategy-template grid sweeper.

Subcommands:
  list                                  list available templates
  show <id>                             show slot expansion + sample combos
  expand <id> [--limit N]               print every expression a template produces
  run <id> [--limit N] [--no-preflight] submit expressions to WQ Brain (budget-aware)
  run <id> --filter "decay >= 10"       submit only the slice matching a constraint
  run <id> --dry-run                    print what would be submitted, don't submit

Examples:
  python scripts/sweep.py list
  python scripts/sweep.py show vol_anomaly_iv_rank
  python scripts/sweep.py expand fundamental_composite
  python scripts/sweep.py run xs_reversal_decay --limit 5
  python scripts/sweep.py run vol_anomaly_iv_rank --filter "window == 252"
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


templates  = _load("templates",  "memory_layer/templates.py")
brain_api  = _load("brain_api",  "memory_layer/brain_api.py")
simulator  = _load("simulator",  "memory_layer/simulator.py")
preflight  = _load("preflight",  "memory_layer/preflight.py")
sessions   = _load("sessions",   "memory_layer/sessions.py")


def cmd_list(args):
    tmpls = templates.load_all()
    if not tmpls:
        print(f"No templates found under {templates.TEMPLATES_DIR}")
        return
    print(f"\n{len(tmpls)} template(s):")
    print("=" * 90)
    for t in tmpls:
        s = templates.stats(t)
        print(f"\n  {t.id}")
        print(f"    {t.description}")
        print(f"    form     : {t.form}")
        print(f"    concepts : {', '.join(t.concepts) or '(none)'}")
        slot_str = " × ".join(f"{x['slot']}({x['size']})" for x in s["slots"])
        print(f"    combos   : {s['total_combos']}  =  {slot_str}")


def cmd_show(args):
    t = templates.load(args.id)
    if not t:
        print(f"template not found: {args.id}", file=sys.stderr); sys.exit(1)
    s = templates.stats(t)
    print(f"\n{t.id}")
    print(f"  {t.description}\n")
    print(f"  form: {t.form}")
    print(f"  concepts: {', '.join(t.concepts) or '(none)'}")
    print(f"  total combos: {s['total_combos']}")
    if t.constraints:
        print(f"  constraints (skip if):")
        for c in t.constraints:
            print(f"    - {c}")
    print("  slots:")
    for b in s["slots"]:
        print(f"    {b['slot']:<10} type={b['type']:<10} size={b['size']:>3}  sample={b['sample']}")
    print("\n  Sample expansions (first 5):")
    for ex in templates.expand(t, max_results=5):
        print(f"    {ex.expression}")


def cmd_expand(args):
    t = templates.load(args.id)
    if not t:
        print(f"template not found: {args.id}", file=sys.stderr); sys.exit(1)
    exprs = templates.expand(t, max_results=args.limit)
    for ex in exprs:
        print(ex.expression)
    print(f"\n# {len(exprs)} expressions{' (truncated)' if args.limit else ''}", file=sys.stderr)


def cmd_run(args):
    t = templates.load(args.id)
    if not t:
        print(f"template not found: {args.id}", file=sys.stderr); sys.exit(1)

    expanded = templates.expand(t)
    if args.filter:
        kept = []
        for ex in expanded:
            try:
                if eval(args.filter, {"__builtins__": {}}, dict(ex.bindings)):
                    kept.append(ex)
            except Exception:
                continue
        expanded = kept
    if args.limit:
        expanded = expanded[:args.limit]

    print(f"\nTemplate: {t.id}")
    print(f"Selected {len(expanded)} expression(s) to submit\n")
    if not expanded:
        return

    if args.dry_run:
        for i, ex in enumerate(expanded, 1):
            print(f"  [{i}] {ex.expression}")
            print(f"      bindings: {ex.bindings}")
        return

    try:
        client = brain_api.BrainAPIClient.from_disk()
    except brain_api.BrainAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr); sys.exit(1)
    try:
        sessions.log_invocation(sys.argv)
    except Exception:
        pass

    budget = simulator.DailyBudget(
        limit=args.budget,
        quiet_hours_before_midnight=0 if args.ignore_quiet_hours else args.quiet_hours,
    )
    print(f"Daily budget: {budget.remaining()}/{args.budget} remaining")
    is_quiet, msg = budget.quiet_window()
    if is_quiet:
        print(f"BLOCKED: {msg}")
        print(f"Pass --ignore-quiet-hours to override.")
        return
    if budget.remaining() < len(expanded):
        print(f"NOTE: budget allows only {budget.remaining()} of {len(expanded)} — "
              f"remainder will be skipped.")

    G = preflight.load_graph() if not args.no_preflight else None

    submitted = 0
    for i, ex in enumerate(expanded, 1):
        print(f"\n=== [{i}/{len(expanded)}] {ex.expression} ===")
        if G is not None:
            sim = preflight.find_similar_attempts(
                G, ex.expression,
                datafields=ex.datafields,
                operators=ex.operators,
                concepts=ex.concepts,
                top_n=3,
            )
            if sim and sim[0].note == "EXACT MATCH" and not args.force:
                print(f"  SKIP — exact prior attempt: {sim[0].alpha_id} (use --force to override)")
                continue
            if sim:
                print(f"  Pre-flight: top similar = {sim[0].alpha_id} "
                      f"(score={sim[0].score}, Sharpe={sim[0].sharpe})")

        try:
            result = simulator.run_simulation(
                client, ex.expression,
                settings=ex.settings,
                poll_interval=args.poll, timeout=args.timeout,
                budget=budget,
                on_progress=lambda m: print(f"  {m}"),
            )
        except simulator.BudgetExhausted as e:
            print(f"  STOP — {e}")
            break
        except brain_api.BrainAuthError as e:
            print(f"  STOP — auth: {e}", file=sys.stderr); break
        except Exception as e:
            print(f"  FAILED — {type(e).__name__}: {e}", file=sys.stderr)
            continue

        m = result.metrics
        print(f"  → status={result.status} "
              f"Sharpe={m.get('sharpe')} Fitness={m.get('fitness')} "
              f"Turnover={m.get('turnover')}%")
        if result.failure_modes:
            print(f"    failure_modes: {', '.join(result.failure_modes)}")

        out_path = simulator.write_back(
            result,
            hypothesis=f"[{t.id}] {t.description[:120]}",
            concepts=ex.concepts,
            datafields=ex.datafields,
            operators=t.operators_hint or ex.operators,
        )
        print(f"    wrote: {out_path.name}")
        try:
            sessions.log_event(
                "simulation",
                summary=f"{out_path.stem}: {ex.expression[:80]}",
                template=t.id, bindings=ex.bindings,
                sim_id=result.sim_id,
                sharpe=m.get("sharpe"), failure_modes=result.failure_modes,
            )
            sessions.attach_alpha(out_path.stem)
        except Exception:
            pass
        submitted += 1

        if args.sleep > 0 and i < len(expanded):
            time.sleep(args.sleep)

    print(f"\nDone. Submitted {submitted}/{len(expanded)}. "
          f"Budget remaining: {budget.remaining()}/{args.budget}.")


def main():
    ap = argparse.ArgumentParser(description="Strategy-template grid sweeper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list"); sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("show"); sp.add_argument("id"); sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("expand"); sp.add_argument("id")
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(fn=cmd_expand)

    sp = sub.add_parser("run"); sp.add_argument("id")
    sp.add_argument("--limit", type=int, default=None,
                    help="Cap number of submissions for this run")
    sp.add_argument("--filter", default=None,
                    help='Python expression on slot bindings, e.g. "decay >= 10"')
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--no-preflight", action="store_true")
    sp.add_argument("--force", action="store_true",
                    help="Submit even when EXACT prior attempt exists")
    sp.add_argument("--budget", type=int, default=30)
    sp.add_argument("--poll", type=float, default=5.0)
    sp.add_argument("--timeout", type=float, default=600.0)
    sp.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between submissions (rate-limit safety)")
    sp.add_argument("--ignore-quiet-hours", action="store_true",
                    help="Override the quiet-hours block")
    sp.add_argument("--quiet-hours", type=int, default=3,
                    help="Hours before local midnight to block submissions (default 3)")
    sp.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

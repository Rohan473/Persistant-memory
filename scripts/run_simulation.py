"""
Submit an alpha expression (or batch of expressions) to WQ Brain, wait for
the result, and write it back into private/nodes/alphas/ as markdown.

Single-shot:
  python scripts/run_simulation.py --expression "rank(close/ts_mean(close,20))"
  python scripts/run_simulation.py --expression "..." --universe TOP1000 --neutralization INDUSTRY

Batch (one expression per non-empty, non-comment line; lines starting with # are skipped):
  python scripts/run_simulation.py --batch ideas.txt

Optional knobs:
  --hypothesis "..."        attach a one-line rationale to new alphas
  --concepts a,b,c          tag concepts for new alphas
  --datafields close,vwap   tag datafields
  --operators rank,ts_mean  tag operators
  --parent alpha_0029       link as derived-from
  --dry-run                 don't submit, just print what would be sent
  --budget N                override daily cap (default 30)
  --poll N                  seconds between status checks (default 5)
  --timeout N               max seconds to wait per sim (default 600)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


brain_api = _load("brain_api", "memory_layer/brain_api.py")
simulator = _load("simulator", "memory_layer/simulator.py")
preflight = _load("preflight", "memory_layer/preflight.py")
sessions  = _load("sessions",  "memory_layer/sessions.py")


def _settings_from_args(args) -> dict:
    s = {}
    if args.universe:        s["universe"] = args.universe
    if args.region:          s["region"] = args.region
    if args.delay is not None: s["delay"] = args.delay
    if args.neutralization:  s["neutralization"] = args.neutralization.upper()
    if args.decay is not None:   s["decay"] = args.decay
    if args.truncation is not None: s["truncation"] = args.truncation
    return s


def _csv_to_list(s):
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def _run_one(client, expression: str, args, budget):
    settings = _settings_from_args(args)

    # Pre-flight context: scan the graph for similar prior attempts
    if not args.no_preflight:
        G = preflight.load_graph()
        attempts = preflight.find_similar_attempts(
            G, expression,
            datafields=_csv_to_list(args.datafields),
            operators=_csv_to_list(args.operators),
            concepts=_csv_to_list(args.concepts),
            parent=args.parent,
        )
        if attempts:
            print(preflight.format_report(attempts))
            if any(a.note == "EXACT MATCH" for a in attempts) and not args.force:
                print("EXACT match exists. Re-run with --force to submit anyway.")
                return None

    if args.dry_run:
        print(f"[dry-run] would submit: {expression}")
        print(f"          settings: {settings}")
        return None

    def on_progress(msg):
        print(msg)

    try:
        result = simulator.run_simulation(
            client, expression, settings=settings,
            poll_interval=args.poll, timeout=args.timeout,
            budget=budget, on_progress=on_progress,
        )
    except simulator.QuietHours as e:
        print(f"\nBLOCKED: {e}", file=sys.stderr)
        sys.exit(4)
    except simulator.BudgetExhausted as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except brain_api.BrainAuthError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"\nResult: status={result.status}")
    if result.metrics:
        m = result.metrics
        print(f"  Sharpe={m.get('sharpe')}  Fitness={m.get('fitness')}  "
              f"Turnover={m.get('turnover')}%  Returns={m.get('returns')}%")
    if result.failure_modes:
        print(f"  failure_modes: {', '.join(result.failure_modes)}")

    out_path = simulator.write_back(
        result,
        hypothesis=args.hypothesis or "",
        concepts=_csv_to_list(args.concepts),
        datafields=_csv_to_list(args.datafields),
        operators=_csv_to_list(args.operators),
        parent_alpha=args.parent,
    )
    print(f"  wrote: {out_path}")
    try:
        sessions.log_event(
            "simulation",
            summary=f"{out_path.stem}: status={result.status} sharpe={result.metrics.get('sharpe')}",
            expression=expression,
            sim_id=result.sim_id,
            alpha_md=out_path.name,
            failure_modes=result.failure_modes,
        )
        sessions.attach_alpha(out_path.stem)
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser(description="Submit alpha simulations to WQ Brain")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--expression", help="Single expression to simulate")
    src.add_argument("--batch", help="Path to file with one expression per line")

    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--concepts", default="")
    ap.add_argument("--datafields", default="")
    ap.add_argument("--operators", default="")
    ap.add_argument("--parent", default=None)

    ap.add_argument("--universe", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--delay", type=int, default=None)
    ap.add_argument("--neutralization", default=None)
    ap.add_argument("--decay", type=int, default=None)
    ap.add_argument("--truncation", type=float, default=None)

    ap.add_argument("--budget", type=int, default=30, help="Daily submission cap")
    ap.add_argument("--poll", type=float, default=5.0, help="Seconds between status polls")
    ap.add_argument("--timeout", type=float, default=600.0, help="Max seconds per simulation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-preflight", action="store_true",
                    help="Skip the 'similar prior attempts' check")
    ap.add_argument("--force", action="store_true",
                    help="Submit even when an EXACT prior attempt exists")
    ap.add_argument("--ignore-quiet-hours", action="store_true",
                    help="Override the quiet-hours block (default 3h before local midnight)")
    ap.add_argument("--quiet-hours", type=int, default=3,
                    help="Hours before local midnight during which submissions are blocked (default 3)")

    args = ap.parse_args()

    try:
        sessions.log_invocation(sys.argv)
    except Exception:
        pass

    try:
        client = brain_api.BrainAPIClient.from_disk()
    except brain_api.BrainAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        sys.exit(1)

    budget = simulator.DailyBudget(
        limit=args.budget,
        quiet_hours_before_midnight=0 if args.ignore_quiet_hours else args.quiet_hours,
    )
    if not args.dry_run:
        print(f"Daily budget: {budget.remaining()}/{args.budget} remaining")
        is_quiet, msg = budget.quiet_window()
        if is_quiet:
            print(f"NOTE: {msg}")

    if args.expression:
        _run_one(client, args.expression, args, budget)
        return

    # Batch mode
    batch_path = Path(args.batch)
    if not batch_path.exists():
        print(f"Batch file not found: {batch_path}", file=sys.stderr)
        sys.exit(1)
    exprs = []
    for line in batch_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exprs.append(line)
    print(f"Batch: {len(exprs)} expressions from {batch_path}")
    print(f"Daily cap leaves room for {budget.remaining()} more today.\n")

    for i, expr in enumerate(exprs, 1):
        print(f"\n=== [{i}/{len(exprs)}] ===")
        try:
            _run_one(client, expr, args, budget)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Fetch one alpha's full record from the WQ Brain API.

Usage:
  python scripts/get_alpha.py <alpha_id>
  python scripts/get_alpha.py <alpha_id> --field status
  python scripts/get_alpha.py <alpha_id> --raw       # full JSON
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _load_brain_api():
    spec = importlib.util.spec_from_file_location(
        "brain_api", BASE / "memory_layer" / "brain_api.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Fetch one alpha from WQ Brain")
    ap.add_argument("alpha_id", help="Platform alpha ID (e.g. Grk6No1x)")
    ap.add_argument("--field", help="Print only this top-level field")
    ap.add_argument("--raw", action="store_true", help="Print full JSON, not summary")
    args = ap.parse_args()

    brain_api = _load_brain_api()
    try:
        client = brain_api.BrainAPIClient.from_disk()
    except brain_api.BrainAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        sys.exit(1)

    r = client._request("GET", f"/alphas/{args.alpha_id}")
    data = r.json()

    if args.field:
        print(json.dumps(data.get(args.field), indent=2, default=str))
        return

    if args.raw:
        print(json.dumps(data, indent=2, default=str))
        return

    # Summary view
    print(f"alpha: {data.get('id')}  status: {data.get('status')}  stage: {data.get('stage')}")
    if (g := data.get("grade")):
        print(f"  grade: {g}")
        if (r_ := data.get("rating")):
            print(f"  rating: {r_}")
    if (s := data.get("settings")):
        print(f"  region={s.get('region')} universe={s.get('universe')} delay={s.get('delay')} "
              f"neutralization={s.get('neutralization')} decay={s.get('decay')} "
              f"truncation={s.get('truncation')}")
    def _print_checks(label, checks):
        if not checks:
            return
        print(f"  {label} checks:")
        for c in checks:
            name = c.get("name", "?")
            res = c.get("result", "?")
            val = c.get("value")
            lim = c.get("limit")
            extra = ""
            if val is not None or lim is not None:
                extra = f"  value={val}  limit={lim}"
            print(f"    {name:25s} {res:10s}{extra}")

    if (m := data.get("is")):
        print(f"  IS:  sharpe={m.get('sharpe')} fitness={m.get('fitness')} "
              f"turnover={m.get('turnover')} returns={m.get('returns')} "
              f"drawdown={m.get('drawdown')} margin={m.get('margin')} "
              f"selfCorr={m.get('selfCorrelation')} prodCorr={m.get('prodCorrelation')}")
        _print_checks("IS", m.get("checks"))
    if (m := data.get("os")):
        print(f"  OS:  sharpe={m.get('osISSharpeRatio')} preCloseSharpe={m.get('preCloseSharpeRatio')}")
        _print_checks("OS", m.get("checks"))
    print()
    print("(use --raw for full JSON, or --field <name> for one field)")


if __name__ == "__main__":
    main()

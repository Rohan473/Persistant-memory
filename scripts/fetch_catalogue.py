"""
Refresh the WQ Brain operator + datafield catalogue cache.

Usage:
  python scripts/fetch_catalogue.py                       # all default settings
  python scripts/fetch_catalogue.py --operators-only
  python scripts/fetch_catalogue.py --datafields-only
  python scripts/fetch_catalogue.py --setting USA,1,TOP3000
  python scripts/fetch_catalogue.py --setting USA,1,TOP3000 --setting USA,1,TOP1000
  python scripts/fetch_catalogue.py --page-size 100 --sleep 0.1
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
brain_catalogue = _load("brain_catalogue", "memory_layer/brain_catalogue.py")


def parse_setting(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--setting expects 'region,delay,universe' (e.g. USA,1,TOP3000), got {s!r}"
        )
    return parts[0], int(parts[1]), parts[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operators-only", action="store_true")
    ap.add_argument("--datafields-only", action="store_true")
    ap.add_argument("--setting", action="append", type=parse_setting,
                    help="region,delay,universe (repeatable). "
                         "Default: USA,1,TOP3000 + TOP1000 + TOP500")
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to sleep between pages (rate-limit safety)")
    args = ap.parse_args()

    try:
        client = brain_api.BrainAPIClient.from_disk()
    except brain_api.BrainAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        sys.exit(1)

    snap = brain_catalogue.refresh(
        client,
        settings=args.setting,
        skip_operators=args.datafields_only,
        skip_datafields=args.operators_only,
        page_size=args.page_size,
        sleep_between_pages=args.sleep,
    )

    print()
    print(f"Saved to {brain_catalogue.CATALOGUE_PATH}")
    print(f"  Operators: {len(snap.operators)}")
    for key, rows in snap.datafields.items():
        print(f"  Datafields[{key}]: {len(rows)}")


if __name__ == "__main__":
    main()

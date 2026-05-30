"""
Recover a completed-but-unwritten WQ Brain alpha: fetch it by remote id,
build a SimulationResult, and run simulator.write_back so it lands in our KG.

Usage:
  python scripts/writeback_alpha.py <remote_alpha_id> <expression> --hypothesis "..." \
        --datafields fld1,fld2 --operators op1,op2 --concepts c1,c2
"""
import argparse
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


brain_api = _load("brain_api", "memory_layer/brain_api.py")
simulator = _load("simulator", "memory_layer/simulator.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("remote_alpha_id")
    ap.add_argument("expression")
    ap.add_argument("--sim-id", default="", help="Optional WQB simulation id")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--datafields", default="")
    ap.add_argument("--operators", default="")
    ap.add_argument("--concepts", default="")
    args = ap.parse_args()

    client = brain_api.BrainAPIClient.from_disk()
    alpha = client._request("GET", f"/alphas/{args.remote_alpha_id}").json()

    result = simulator._parse_response(
        sim_id=args.sim_id or args.remote_alpha_id,
        expression=args.expression,
        sim_payload={"status": "COMPLETE", "alpha": args.remote_alpha_id},
        alpha_payload=alpha,
    )

    df_list = [s for s in args.datafields.split(",") if s]
    op_list = [s for s in args.operators.split(",") if s]
    cn_list = [s for s in args.concepts.split(",") if s]

    out_path = simulator.write_back(
        result,
        hypothesis=args.hypothesis,
        concepts=cn_list,
        datafields=df_list,
        operators=op_list,
    )
    m = result.metrics
    print(f"wrote: {out_path}")
    print(f"  status={result.status} sharpe={m.get('sharpe')} fitness={m.get('fitness')} turnover={m.get('turnover')}%")
    if result.failure_modes:
        print(f"  failure_modes: {', '.join(result.failure_modes)}")


if __name__ == "__main__":
    main()

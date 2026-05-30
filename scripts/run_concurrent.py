"""
Fan out multiple sweep.py runs in parallel, capped at WQB's concurrent-sim limit
(default 3). Each template runs in its own sweep.py subprocess; each subprocess
posts sims sequentially internally, so total concurrent sims = number of active
subprocesses (one per template).

Two output modes:
  default        Capture per-template output; print combined report when each
                 subprocess completes. Optional --log-dir streams live to files.
  --tabs         Launch each template in its own Windows Terminal tab via
                 wt.exe. Output stays visible in each tab. Parent exits immediately
                 after launching — runs are truly independent. Requires Windows
                 Terminal (wt.exe). Each tab stays open after the run completes
                 so results remain visible.

Usage:
  python scripts/run_concurrent.py <template_id> [<template_id> ...] [--max-concurrent N]
  python scripts/run_concurrent.py phase_a phase_b phase_c --budget 100 --ignore-quiet-hours
  python scripts/run_concurrent.py phase_a phase_b phase_c --log-dir logs/concurrent
  python scripts/run_concurrent.py phase_a phase_b phase_c --tabs

Pass-through flags to sweep.py: --budget, --ignore-quiet-hours, --force, --filter, --limit.
"""

import argparse
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SWEEP_PATH = BASE / "scripts" / "sweep.py"


def launch_in_terminal_tab(template_id: str, common_args: list[str]) -> bool:
    """
    Spawn `sweep.py run <template_id>` in a new Windows Terminal tab.
    Returns True if launched, False if wt.exe not found.

    The tab uses `cmd /k` so the prompt remains after the run finishes —
    results stay visible until the user closes the tab.
    """
    wt = shutil.which("wt.exe") or shutil.which("wt")
    if not wt:
        return False
    py_cmd = (
        f'"{sys.executable}" "{SWEEP_PATH}" run {template_id} '
        + " ".join(common_args)
    )
    # `wt new-tab --title X cmd /k <full python invocation>`
    # cmd /k keeps the tab alive after the python process exits.
    subprocess.Popen(
        [wt, "new-tab", "--title", template_id,
         "cmd", "/k", py_cmd],
        cwd=str(BASE),
    )
    return True


def run_one(template_id: str, common_args: list[str], log_dir: Path | None):
    """
    Run sweep.py for one template. Returns (template_id, returncode, output).
    If log_dir is set, also stream output to a per-template log file as it arrives.
    """
    cmd = [sys.executable, str(SWEEP_PATH), "run", template_id] + common_args
    started_at = datetime.now().strftime("%H:%M:%S")

    if log_dir is None:
        # Simpler path — capture all and return.
        proc = subprocess.run(
            cmd, cwd=str(BASE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        return template_id, proc.returncode, started_at, proc.stdout

    # Streaming path — pipe to log file as it arrives.
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{template_id}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {template_id} started {started_at}\n# cmd: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(BASE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        chunks = []
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            chunks.append(line)
        proc.wait()
        return template_id, proc.returncode, started_at, "".join(chunks)


def main():
    ap = argparse.ArgumentParser(
        description="Fan out N sweep.py processes with max-concurrent cap (default 3 = WQB limit)"
    )
    ap.add_argument("templates", nargs="+", help="Template IDs to run in parallel")
    ap.add_argument("--max-concurrent", type=int, default=3,
                    help="Max concurrent sweep.py subprocesses (default 3 = WQB limit)")
    ap.add_argument("--log-dir", default=None,
                    help="Optional dir to stream each template's output to a separate log file")
    # Pass-through to sweep.py
    ap.add_argument("--budget", type=int, default=0,
                    help="Daily sim cap; 0 = unlimited (default). WQB-side cap + 3-concurrent still apply.")
    ap.add_argument("--ignore-quiet-hours", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--tabs", action="store_true",
                    help="Launch each template in its own Windows Terminal tab (wt.exe). "
                         "Parent exits immediately; each tab stays open after its run.")
    args = ap.parse_args()

    common = ["--budget", str(args.budget)]
    if args.ignore_quiet_hours:
        common.append("--ignore-quiet-hours")
    if args.force:
        common.append("--force")
    if args.filter:
        common += ["--filter", args.filter]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    if args.no_preflight:
        common.append("--no-preflight")

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir = log_dir if log_dir.is_absolute() else (BASE / log_dir)

    n = len(args.templates)

    # --tabs mode: launch each template in its own Windows Terminal tab and exit.
    if args.tabs:
        print(f"Launching {n} template(s) in Windows Terminal tabs")
        print(f"Pass-through args: {common}\n")
        launched = 0
        for t in args.templates:
            if launch_in_terminal_tab(t, common):
                print(f"  -> tab opened: {t}")
                launched += 1
            else:
                print(f"  !! wt.exe not found — cannot launch {t}", file=sys.stderr)
        if launched == 0:
            print("\nNo tabs launched. Install Windows Terminal (wt.exe) "
                  "or omit --tabs to use the default capture mode.", file=sys.stderr)
            sys.exit(1)
        print(f"\nLaunched {launched}/{n} tab(s). Each runs independently; "
              f"check each tab for live progress and results.")
        return

    cap = min(args.max_concurrent, n)
    print(f"Running {n} template(s) with max_concurrent={cap}")
    print(f"Pass-through args: {common}")
    if log_dir:
        print(f"Streaming per-template logs to: {log_dir}")
    print()

    t_start = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=cap) as executor:
        futures = {
            executor.submit(run_one, t, common, log_dir): t
            for t in args.templates
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                tid, rc, started, output = fut.result()
            except Exception as e:
                print(f"\n=== [{t}] CRASHED — {type(e).__name__}: {e} ===")
                continue
            completed += 1
            elapsed = time.time() - t_start
            print(f"\n=== [{tid}] returncode={rc} "
                  f"({completed}/{n} done, {elapsed:.1f}s elapsed) ===")
            print(output)

    print(f"\nAll {n} template(s) finished in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

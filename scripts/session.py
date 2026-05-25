"""
Research-session CLI.

Subcommands:
  start "title"          open a new session (one active at a time)
  end [--note "..."]     close the active session, write markdown summary
  status                 show the active session
  list                   list past sessions
  show <id>              display a session's full event log
  replay <id> [--run]    print (or execute) the commands from a session
                         (simulation commands are skipped on --run)
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


sessions = _load("sessions", "memory_layer/sessions.py")


def cmd_start(args):
    if not args.title:
        print("error: provide a session title", file=sys.stderr)
        sys.exit(1)
    try:
        s = sessions.start(args.title)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"session started: {s.id}")
    print(f"  title: {s.title}")
    print(f"  start: {s.start}")
    print(f"  file:  {s.path_json()}")


def cmd_end(args):
    s = sessions.end(note=args.note)
    if s is None:
        print("no active session.")
        sys.exit(1)
    print(f"session ended: {s.id}")
    print(f"  events: {len(s.events)}")
    print(f"  alphas touched: {', '.join(s.alphas_touched) or 'none'}")
    if s.conclusion:
        print(f"  conclusion: {s.conclusion}")
    print(f"  markdown: {s.path_md()}")


def cmd_status(args):
    s = sessions.load_active()
    if s is None:
        print("no active session")
        return
    print(f"active session: {s.id}")
    print(f"  title:  {s.title}")
    print(f"  start:  {s.start}")
    print(f"  events: {len(s.events)}")
    if s.alphas_touched:
        print(f"  alphas touched: {', '.join(s.alphas_touched)}")


def cmd_list(args):
    rows = sessions.list_sessions()
    if not rows:
        print("no sessions found.")
        return
    print(f"{len(rows)} session(s):")
    for s in rows:
        end = s.end[:19] if s.end else "(open)"
        n_alphas = len(s.alphas_touched)
        print(f"  {s.id}   start={s.start[:19]}  end={end}  "
              f"events={len(s.events):>3}  alphas={n_alphas}")
        print(f"      title: {s.title}")
        if s.conclusion:
            print(f"      note:  {s.conclusion}")


def cmd_show(args):
    s = sessions.load(args.id)
    if s is None:
        print(f"session not found: {args.id}")
        sys.exit(1)
    print(f"\n{s.id} — {s.title}")
    print(f"start: {s.start}")
    print(f"end:   {s.end or '(open)'}")
    print(f"alphas touched: {', '.join(s.alphas_touched) or 'none'}")
    if s.conclusion:
        print(f"conclusion: {s.conclusion}")
    print(f"\nEvent log ({len(s.events)}):")
    for ev in s.events:
        ts = ev.time[11:19]
        marker = {"command": "$", "simulation": "▶", "note": "·"}.get(ev.type, "?")
        print(f"  [{ts}] {marker} {ev.type}: {ev.summary}")


def cmd_replay(args):
    cmds = sessions.replay_commands(args.id, dry_run=not args.run)
    if not cmds:
        print(f"no replayable commands in session {args.id}")
        return
    if args.run:
        return  # already executed inside replay_commands
    print(f"# Commands in session {args.id} ({len(cmds)} total; simulation calls skipped on replay)")
    for c in cmds:
        print(c)


def main():
    ap = argparse.ArgumentParser(description="Research-session CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start"); sp.add_argument("title"); sp.set_defaults(fn=cmd_start)
    sp = sub.add_parser("end"); sp.add_argument("--note", default=None); sp.set_defaults(fn=cmd_end)
    sp = sub.add_parser("status"); sp.set_defaults(fn=cmd_status)
    sp = sub.add_parser("list"); sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("show"); sp.add_argument("id"); sp.set_defaults(fn=cmd_show)
    sp = sub.add_parser("replay"); sp.add_argument("id")
    sp.add_argument("--run", action="store_true", help="Actually execute commands (skips simulations)")
    sp.set_defaults(fn=cmd_replay)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

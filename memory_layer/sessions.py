"""
Research-session tracking for the WQ Brain knowledge graph.

A "session" is a named, time-bounded research thread. While active, any
invocation of scripts/query.py or scripts/run_simulation.py is logged into
the session. On end, a markdown summary is written to
private/nodes/sessions/ so it gets picked up by build_graph.py as a Session
node and linked to the alphas it produced.

State files:
  memory_layer/active_session.json       — pointer to the currently-open session
  private/nodes/sessions/<id>.json       — full event log (per session)
  private/nodes/sessions/<id>.md         — markdown summary (written on end)

The integration points in query.py and run_simulation.py are intentionally
tiny: they call `log_invocation()` at startup and `log_event()` for
significant events. If no session is active, both are no-ops.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE = Path(__file__).resolve().parent.parent
ACTIVE_FILE = BASE / "memory_layer" / "active_session.json"
SESSIONS_DIR = BASE / "private" / "nodes" / "sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _gen_id(title: Optional[str] = None) -> str:
    suffix = secrets.token_hex(2)
    return f"sess_{_today()}_{suffix}"


@dataclass
class SessionEvent:
    time: str
    type: str           # "command" | "simulation" | "note"
    summary: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    id: str
    title: str
    start: str
    end: Optional[str] = None
    events: List[SessionEvent] = field(default_factory=list)
    alphas_touched: List[str] = field(default_factory=list)
    conclusion: Optional[str] = None

    def path_json(self) -> Path:
        return SESSIONS_DIR / f"{self.id}.json"

    def path_md(self) -> Path:
        return SESSIONS_DIR / f"{self.id}.md"

    def save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # events come back as dicts already via dataclass asdict
        self.path_json().write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── active-session pointer ───────────────────────────────────────────────────

def get_active_id() -> Optional[str]:
    if not ACTIVE_FILE.exists():
        return None
    try:
        return json.loads(ACTIVE_FILE.read_text(encoding="utf-8")).get("id")
    except Exception:
        return None


def _set_active(session_id: Optional[str]) -> None:
    ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if session_id is None:
        if ACTIVE_FILE.exists():
            ACTIVE_FILE.unlink()
        return
    ACTIVE_FILE.write_text(
        json.dumps({"id": session_id, "since": _now_iso()}, indent=2),
        encoding="utf-8",
    )


# ── load/save ────────────────────────────────────────────────────────────────

def load(session_id: str) -> Optional[Session]:
    p = SESSIONS_DIR / f"{session_id}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    s = Session(
        id=raw["id"], title=raw["title"], start=raw["start"],
        end=raw.get("end"), conclusion=raw.get("conclusion"),
        alphas_touched=raw.get("alphas_touched", []),
    )
    s.events = [SessionEvent(**e) for e in raw.get("events", [])]
    return s


def load_active() -> Optional[Session]:
    sid = get_active_id()
    return load(sid) if sid else None


def list_sessions() -> List[Session]:
    out: List[Session] = []
    if not SESSIONS_DIR.exists():
        return out
    for p in sorted(SESSIONS_DIR.glob("sess_*.json")):
        s = load(p.stem)
        if s:
            out.append(s)
    return out


# ── lifecycle ────────────────────────────────────────────────────────────────

def start(title: str) -> Session:
    if get_active_id():
        raise RuntimeError(
            f"A session is already active: {get_active_id()}. "
            f"End it first with `python scripts/session.py end`."
        )
    sid = _gen_id(title)
    s = Session(id=sid, title=title, start=_now_iso())
    s.save()
    _set_active(sid)
    return s


def end(note: Optional[str] = None) -> Optional[Session]:
    s = load_active()
    if s is None:
        return None
    s.end = _now_iso()
    s.conclusion = note
    s.save()
    _write_markdown_summary(s)
    _set_active(None)
    return s


# ── logging hooks ────────────────────────────────────────────────────────────

def log_event(event_type: str, summary: str, **detail) -> None:
    """Called by query.py / run_simulation.py. No-op when no active session."""
    s = load_active()
    if s is None:
        return
    s.events.append(SessionEvent(time=_now_iso(), type=event_type,
                                 summary=summary, detail=detail))
    s.save()


def log_invocation(argv: List[str]) -> None:
    """Log a CLI invocation with its argv. Truncates argv to avoid log bloat."""
    if not argv:
        return
    s = load_active()
    if s is None:
        return
    summary = " ".join(argv[1:6]) if len(argv) > 1 else Path(argv[0]).name
    s.events.append(SessionEvent(
        time=_now_iso(), type="command",
        summary=summary,
        detail={"argv": argv[:10]},
    ))
    s.save()


def attach_alpha(alpha_id: str) -> None:
    s = load_active()
    if s is None:
        return
    if alpha_id and alpha_id not in s.alphas_touched:
        s.alphas_touched.append(alpha_id)
        s.save()


# ── markdown summary ─────────────────────────────────────────────────────────

def _write_markdown_summary(s: Session) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("---")
    lines.append(f"id: {s.id}")
    lines.append(f"title: {json.dumps(s.title)}")
    lines.append(f"date: {s.start[:10]}")
    lines.append(f"start: {s.start}")
    lines.append(f"end: {s.end or ''}")
    lines.append(f"turn_count: {len(s.events)}")
    if s.alphas_touched:
        lines.append("alphas_touched: [" + ", ".join(s.alphas_touched) + "]")
    if s.conclusion:
        lines.append(f"conclusion: {json.dumps(s.conclusion)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {s.title}")
    lines.append("")
    lines.append(f"**Window:** {s.start} → {s.end or 'open'}")
    if s.alphas_touched:
        lines.append(f"**Alphas touched:** {', '.join(s.alphas_touched)}")
    if s.conclusion:
        lines.append("")
        lines.append(f"**Conclusion:** {s.conclusion}")
    lines.append("")
    lines.append("## Event log")
    lines.append("")
    for ev in s.events:
        ts = ev.time[11:19]  # HH:MM:SS UTC
        marker = {"command": "$", "simulation": "▶", "note": "·"}.get(ev.type, "?")
        lines.append(f"- `{ts}` {marker} **{ev.type}** — {ev.summary}")
        if ev.detail:
            for k, v in ev.detail.items():
                if isinstance(v, (list, dict)):
                    v = json.dumps(v)[:140]
                else:
                    v = str(v)[:140]
                lines.append(f"    - {k}: {v}")
    s.path_md().write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── replay ───────────────────────────────────────────────────────────────────

def replay_commands(session_id: str, dry_run: bool = True) -> List[str]:
    """Return the list of command argv strings from a session. If not dry_run,
    actually execute each non-simulation command via subprocess."""
    import shlex, subprocess
    s = load(session_id)
    if s is None:
        return []
    cmds = []
    for ev in s.events:
        if ev.type != "command":
            continue
        argv = ev.detail.get("argv") if isinstance(ev.detail, dict) else None
        if not argv:
            continue
        cmd_line = " ".join(shlex.quote(a) for a in argv)
        cmds.append(cmd_line)
    if dry_run:
        return cmds
    # Skip simulation commands during replay — never re-burn the budget.
    for c in cmds:
        if "run_simulation" in c:
            print(f"  [skip] {c}")
            continue
        print(f"  $ {c}")
        try:
            subprocess.run(c, shell=True, cwd=str(BASE), check=False)
        except Exception as e:
            print(f"    FAILED: {e}")
    return cmds

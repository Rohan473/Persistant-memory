"""
Save each session's full conversational text to ./nodes/sessions/raw/ so
extraction sub-agents can read them without touching the large exports.json.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXPORTS_DIR = BASE / "exports"
RAW_DIR = BASE / "nodes" / "sessions" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPORTS_DIR / "exports.json", encoding="utf-8") as f:
    data = json.load(f)

for session in data:
    sid = session.get("uuid", "unknown")
    safe_id = re.sub(r'[^\w\-]', '_', sid)[:80]
    title = session.get("name", "Untitled")
    messages = session.get("chat_messages", [])

    lines = [f"SESSION: {title}", f"ID: {sid}", f"DATE: {session.get('created_at','')[:10]}", "=" * 60, ""]
    for m in messages:
        role = m.get("sender", "unknown").upper()
        text = m.get("text", "")
        if not text:
            parts = [c.get("text","") for c in m.get("content",[]) if c.get("type")=="text"]
            text = "\n".join(parts)
        lines.append(f"[{role}]")
        lines.append(text.strip())
        lines.append("")

    out = RAW_DIR / f"{safe_id}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {safe_id}.txt  ({len(messages)} messages, {out.stat().st_size//1024} KB)")

print(f"\nDone. {len(data)} session files written to {RAW_DIR}")

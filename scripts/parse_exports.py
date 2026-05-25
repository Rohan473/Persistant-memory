"""
Phase 2: Parse chat exports (Claude .json format) and filter WQ Brain sessions.
Usage: python scripts/parse_exports.py
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
EXPORTS_DIR = BASE / "exports"
SESSIONS_DIR = BASE / "nodes" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── WQ Brain filter keywords ───────────────────────────────────────────────────
WQ_KEYWORDS = [
    r"\balpha\b", r"\bsharpe\b", r"\bfitness\b", r"\bturnover\b",
    r"\bts_rank\b", r"\bgroup_neutralize\b", r"\bTOP3000\b", r"\bTOP1000\b",
    r"\bneutralization\b", r"\bdecay\b", r"\bdatafield\b", r"\bsimulation\b",
    r"\b\bIS\b", r"\bOS\b", r"\bsub-universe\b", r"\bsub_universe\b",
    r"\bworldquant\b", r"\bwqbrain\b", r"\bfastexpr\b", r"\brank\b.*\bdecay\b",
    r"\bts_corr\b", r"\bts_mean\b", r"\bts_std\b", r"\bts_skewness\b",
    r"\bvwap\b", r"\badv\b", r"\bcap\b", r"\bindustry\b.*\bneutral",
    r"\bsubindustry\b", r"\bregion\b.*\bUSA\b",
]
WQ_PATTERN = re.compile("|".join(WQ_KEYWORDS), re.IGNORECASE)


def detect_format(data):
    """Return 'claude' or 'chatgpt'."""
    if isinstance(data, list) and data and "chat_messages" in data[0]:
        return "claude"
    if isinstance(data, list) and data and "mapping" in data[0]:
        return "chatgpt"
    # single session object
    if isinstance(data, dict) and "chat_messages" in data:
        return "claude"
    if isinstance(data, dict) and "mapping" in data:
        return "chatgpt"
    return "unknown"


def normalize_claude(session):
    """Normalize Claude export session to common schema."""
    messages = session.get("chat_messages", [])
    turns = []
    for m in messages:
        role = m.get("sender", "unknown")
        # Prefer top-level text field (already concatenated)
        text = m.get("text", "")
        if not text:
            # Fallback: concat text parts from content array
            parts = [
                c.get("text", "") for c in m.get("content", [])
                if c.get("type") == "text"
            ]
            text = "\n".join(parts)
        turns.append({"role": role, "content": text})

    raw_date = session.get("created_at", "")
    try:
        date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        date = raw_date[:10] if raw_date else "unknown"

    return {
        "session_id": session.get("uuid", "unknown"),
        "title": session.get("name", "Untitled"),
        "date": date,
        "source": "claude",
        "turns": turns,
    }


def normalize_chatgpt(session):
    """Normalize ChatGPT export session to common schema."""
    mapping = session.get("mapping", {})
    turns = []
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "unknown")
        if role not in ("user", "assistant"):
            continue
        parts = msg.get("content", {}).get("parts", [])
        text = "\n".join(str(p) for p in parts if isinstance(p, str))
        if text.strip():
            turns.append({"role": role, "content": text})

    raw_date = session.get("create_time", 0)
    try:
        date = datetime.utcfromtimestamp(float(raw_date)).strftime("%Y-%m-%d")
    except Exception:
        date = "unknown"

    return {
        "session_id": session.get("id", "unknown"),
        "title": session.get("title", "Untitled"),
        "date": date,
        "source": "chatgpt",
        "turns": turns,
    }


def is_wq_session(session_norm):
    """Return True if any turn mentions WQ Brain keywords."""
    all_text = " ".join(t["content"] for t in session_norm["turns"])
    return bool(WQ_PATTERN.search(all_text))


def write_session_md(session_norm):
    """Write session to ./nodes/sessions/{session_id}.md with YAML frontmatter."""
    sid = session_norm["session_id"]
    # Sanitize filename
    safe_id = re.sub(r'[^\w\-]', '_', sid)[:80]
    path = SESSIONS_DIR / f"{safe_id}.md"

    turn_count = len(session_norm["turns"])
    title_escaped = session_norm["title"].replace('"', '\\"')

    # Build body: first 2 turns as preview (truncated)
    preview_lines = []
    for i, t in enumerate(session_norm["turns"][:4]):
        snippet = t["content"][:300].replace("\n", " ").strip()
        preview_lines.append(f"**[{t['role']}]** {snippet}...")

    body = "\n\n".join(preview_lines)

    content = f"""---
id: "{safe_id}"
title: "{title_escaped}"
date: "{session_norm['date']}"
source: "{session_norm['source']}"
turn_count: {turn_count}
---

{body}
"""
    path.write_text(content, encoding="utf-8")
    return safe_id


def main():
    json_files = list(EXPORTS_DIR.glob("*.json"))
    if not json_files:
        print("No .json files found in ./exports/")
        return

    total_sessions = 0
    kept = []
    discarded_count = 0

    for jf in json_files:
        print(f"\nReading {jf.name} ({jf.stat().st_size // 1024} KB)...")
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        fmt = detect_format(data)
        print(f"  Detected format: {fmt}")

        # Normalize to a list of raw sessions
        if isinstance(data, dict):
            data = [data]

        sessions_raw = data
        total_sessions += len(sessions_raw)

        for raw in sessions_raw:
            if fmt == "claude":
                norm = normalize_claude(raw)
            elif fmt == "chatgpt":
                norm = normalize_chatgpt(raw)
            else:
                # Try claude heuristic
                norm = normalize_claude(raw)

            if is_wq_session(norm):
                safe_id = write_session_md(norm)
                kept.append({
                    "session_id": norm["session_id"],
                    "safe_id": safe_id,
                    "title": norm["title"],
                    "date": norm["date"],
                    "turn_count": len(norm["turns"]),
                })
            else:
                discarded_count += 1

    print(f"\n{'='*60}")
    print(f"Total sessions found : {total_sessions}")
    print(f"Kept (WQ Brain)      : {len(kept)}")
    print(f"Discarded            : {discarded_count}")
    print(f"\nKept sessions:")
    for s in kept:
        print(f"  [{s['date']}] {s['title'][:60]:<60}  ({s['turn_count']} turns)  ->  {s['safe_id']}.md")

    # Save a manifest for Phase 3
    manifest_path = BASE / "nodes" / "sessions" / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()

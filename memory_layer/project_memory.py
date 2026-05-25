"""
Bridge to Claude Code's auto-memory system.

Reads MEMORY.md and the per-memory files in
  ~/.claude/projects/<slug>/memory/
and surfaces relevant entries inline in query outputs.

Matching is keyword-based for now (substring across title + description +
body). Cheap and good enough at this scale; can be upgraded to embedding
similarity later by reusing memory_layer/embed.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

# Auto-memory lives in C:\Users\<user>\.claude\projects\<project-slug>\memory\
DEFAULT_MEMORY_DIR = (
    Path.home() / ".claude" / "projects"
    / "E--New-folder-WQB-knowledge-graph" / "memory"
)


@dataclass
class MemoryEntry:
    name: str
    title: str
    description: str
    file_path: Path
    body: str = ""
    mem_type: str = "unknown"

    def text(self) -> str:
        return f"{self.title}\n{self.description}\n{self.body}"


def _resolve_memory_dir(override: Optional[Path] = None) -> Optional[Path]:
    if override and override.exists():
        return override
    if DEFAULT_MEMORY_DIR.exists():
        return DEFAULT_MEMORY_DIR
    env = os.environ.get("WQB_MEMORY_DIR")
    if env and Path(env).exists():
        return Path(env)
    return None


_INDEX_LINE_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]+)\)\s*(?:[—-]\s*(.*))?$")


def _parse_index(index_path: Path) -> List[MemoryEntry]:
    entries: List[MemoryEntry] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        m = _INDEX_LINE_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip()
        rel = m.group(2).strip()
        hook = (m.group(3) or "").strip()
        fp = index_path.parent / rel
        entries.append(MemoryEntry(
            name=Path(rel).stem,
            title=title,
            description=hook,
            file_path=fp,
        ))
    return entries


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


def _hydrate(entry: MemoryEntry) -> None:
    if not entry.file_path.exists():
        return
    raw = entry.file_path.read_text(encoding="utf-8", errors="replace")
    m = _FRONTMATTER_RE.match(raw)
    if m:
        # extract type if present, body is everything after the frontmatter
        fm = m.group(1)
        entry.body = m.group(2).strip()
        type_m = re.search(r"type:\s*([A-Za-z_]+)", fm)
        if type_m:
            entry.mem_type = type_m.group(1)
    else:
        entry.body = raw.strip()


def load_memories(memory_dir: Optional[Path] = None) -> List[MemoryEntry]:
    """Load all memory entries from the auto-memory index. Returns [] if absent."""
    mdir = _resolve_memory_dir(memory_dir)
    if mdir is None:
        return []
    idx = mdir / "MEMORY.md"
    if not idx.exists():
        return []
    entries = _parse_index(idx)
    for e in entries:
        _hydrate(e)
    return entries


def find_relevant(
    memories: List[MemoryEntry],
    *terms: str,
    types: Optional[Iterable[str]] = None,
    limit: int = 5,
) -> List[MemoryEntry]:
    """Substring-match terms against title + description + body. Returns top matches."""
    type_filter: Optional[Set[str]] = set(types) if types else None
    needles = [t.lower() for t in terms if t]
    scored = []
    for e in memories:
        if type_filter and e.mem_type not in type_filter:
            continue
        hay = e.text().lower()
        score = sum(hay.count(n) for n in needles) if needles else 0
        if needles and score == 0:
            continue
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


def format_memory_block(entries: List[MemoryEntry], header: str = "Project memory") -> str:
    if not entries:
        return ""
    lines = [f"\n{header}:"]
    for e in entries:
        lines.append(f"  • {e.title}")
        if e.description:
            lines.append(f"      {e.description}")
    return "\n".join(lines) + "\n"

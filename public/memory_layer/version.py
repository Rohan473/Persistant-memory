"""
Simple version snapshots for memory layer.
Stores timestamped snapshots of metadata for historical reference.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from .config import config

def get_version_path() -> Path:
    """Get the path to the versions directory."""
    return Path(config.metadata_output).parent / "versions"

def ensure_version_dir() -> None:
    """Ensure the versions directory exists."""
    get_version_path().mkdir(parents=True, exist_ok=True)

def create_snapshot(metadata: List[Dict], label: Optional[str] = None) -> str:
    """
    Create a timestamped snapshot of the current metadata.

    Args:
        metadata: List of metadata dictionaries
        label: Optional label for the snapshot

    Returns:
        Snapshot filename
    """
    ensure_version_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        filename = f"snapshot_{label}_{timestamp}.json"
    else:
        filename = f"snapshot_{timestamp}.json"

    filepath = get_version_path() / filename
    snapshot = {
        "created_at": datetime.now().isoformat(),
        "label": label,
        "node_count": len(metadata),
        "node_types": _count_by_type(metadata),
        "metadata": metadata
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    return filename

def _count_by_type(metadata: List[Dict]) -> Dict[str, int]:
    """Count nodes by type."""
    counts = {}
    for m in metadata:
        node_type = m.get("node_type", "unknown")
        counts[node_type] = counts.get(node_type, 0) + 1
    return counts

def list_snapshots() -> List[Dict]:
    """List all available snapshots."""
    ensure_version_dir()
    snapshots = []
    for f in get_version_path().glob("snapshot_*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            snapshots.append({
                "filename": f.name,
                "created_at": data.get("created_at"),
                "label": data.get("label"),
                "node_count": data.get("node_count"),
            })
    return sorted(snapshots, key=lambda x: x["created_at"], reverse=True)

def load_snapshot(filename: str) -> List[Dict]:
    """Load a specific snapshot."""
    filepath = get_version_path() / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("metadata", [])

def get_latest_snapshot() -> Optional[List[Dict]]:
    """Get the most recent snapshot if any exists."""
    snapshots = list_snapshots()
    if not snapshots:
        return None
    return load_snapshot(snapshots[0]["filename"])
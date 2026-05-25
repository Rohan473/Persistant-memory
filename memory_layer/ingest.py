"""
Ingestion pipeline: graph -> structured metadata -> embeddings -> Qdrant
"""

import json
from typing import List, Dict, Optional, Set
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .config import config, get_qdrant_url
from .embed import embed_texts
from .structure import extract_all_metadata, load_metadata, save_metadata
from .version import create_snapshot
from .salience import init_salience

_client = None

import hashlib

def qdrant_id(node_id: str) -> str:
    """Convert graph node ID to Qdrant-compatible UUID."""
    # Use MD5 hash truncated to create a valid UUID-like string
    hash_bytes = hashlib.md5(node_id.encode()).digest()
    # Format as UUID-like string (first 16 bytes as hex)
    return hash_bytes[:16].hex()

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=get_qdrant_url())
    return _client

def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if config.collection_name not in collection_names:
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.embedding_dim,
                distance=Distance.COSINE
            )
        )

def get_existing_ids() -> Set[str]:
    """Get all existing node IDs in Qdrant."""
    client = get_qdrant_client()
    try:
        # Scroll through all points to get IDs
        existing_ids = set()
        offset = None
        while True:
            results = client.scroll(
                collection_name=config.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True
            )
            points, offset = results
            for p in points:
                existing_ids.add(p.payload.get("id"))
            if offset is None:
                break
        return existing_ids
    except Exception:
        return set()

def ingest_metadata(metadata: List[Dict], force: bool = False) -> Dict:
    """
    Ingest metadata into Qdrant with embeddings.

    Args:
        metadata: List of structured metadata
        force: If True, re-ingest all. If False, skip existing.

    Returns:
        Dict with ingest statistics
    """
    ensure_collection()

    # Initialize salience
    metadata = init_salience(metadata, config.default_importance)

    # Get existing IDs for incremental ingest
    existing_ids = set()
    if not force:
        existing_ids = get_existing_ids()

    # Separate new and existing
    to_ingest = []
    skipped = 0

    for m in metadata:
        node_id = m.get("id", "")
        qid = qdrant_id(node_id)
        if qid in existing_ids and not force:
            skipped += 1
            continue

        structured = m.get("structured_summary", "")
        if not structured:
            continue
        to_ingest.append(m)

    if not to_ingest:
        return {
            "ingested": 0,
            "skipped": skipped,
            "total": len(metadata)
        }

    # Generate embeddings
    texts = [m.get("structured_summary", "") for m in to_ingest]
    embeddings = embed_texts(texts)

    # Create points
    points = []
    for i, m in enumerate(to_ingest):
        # Lightweight payload - no raw graph data
        payload = {
            "id": m.get("id"),
            "node_type": m.get("node_type"),
            "name": m.get("name"),
            "structured_summary": m.get("structured_summary"),
            # Only store lightweight refs, not raw graph neighborhoods
            "relationships": m.get("relationships", []),
            "salience": m.get("salience", {}),
        }

        # Add type-specific fields (lightweight)
        if m.get("node_type") == "Alpha":
            payload.update({
                "operators": m.get("operators", [])[:5],
                "concepts": m.get("concepts", [])[:5],
                "datafields": m.get("datafields", [])[:5],
                "failure_modes": m.get("failure_modes", [])[:3],
                "rating": m.get("rating"),
                "sharpe": m.get("sharpe"),
            })

        # Convert ID to Qdrant-compatible format using hash
        qid = qdrant_id(m.get("id", ""))

        points.append(PointStruct(
            id=qid,
            vector=embeddings[i].tolist(),
            payload=payload
        ))

    # Upsert to Qdrant
    client = get_qdrant_client()
    client.upsert(
        collection_name=config.collection_name,
        points=points
    )

    return {
        "ingested": len(points),
        "skipped": skipped,
        "total": len(metadata)
    }

def ingest_all(force: bool = False, create_version: bool = True) -> Dict:
    """
    Full ingestion pipeline: extract -> save metadata -> ingest to Qdrant.

    Args:
        force: Re-ingest all nodes even if they exist
        create_version: Create a snapshot after ingest

    Returns:
        Dict with ingest statistics
    """
    # Extract metadata from graph
    metadata = extract_all_metadata()

    # Save metadata to file (for inspection/debugging)
    save_metadata(metadata)

    # Create version snapshot
    if create_version:
        create_snapshot(metadata, label="ingest")

    # Ingest to Qdrant
    return ingest_metadata(metadata, force=force)

def ingest_node_type(node_type: str, force: bool = False) -> Dict:
    """Ingest only a specific node type."""
    metadata = extract_all_metadata()
    filtered = [m for m in metadata if m.get("node_type") == node_type]

    if not filtered:
        return {"ingested": 0, "skipped": 0, "total": 0, "message": f"No nodes of type {node_type}"}

    save_metadata(filtered)
    if create_version := True:
        create_snapshot(filtered, label=f"ingest_{node_type}")

    return ingest_metadata(filtered, force=force)

def get_stats() -> Dict:
    """Get Qdrant collection statistics."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(config.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
    except Exception as e:
        return {"error": str(e)}
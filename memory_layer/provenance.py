"""
Provenance Tracking
Every extracted concept knows its source with full metadata.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

class SourceType(Enum):
    INTERNAL_ALPHA = "internal_alpha"
    INTERNAL_CONCEPT = "internal_concept"
    EXTERNAL_PAPER = "external_paper"
    EXTERNAL_BLOG = "external_blog"
    EXTERNAL_FORUM = "external_forum"
    EXTERNAL_DOCS = "external_documentation"
    EXTERNAL_WEB = "external_web"

# Source authority weights - NOT all sources are equal
SOURCE_AUTHORITY = {
    "internal_alpha": 1.0,
    "internal_concept": 1.0,
    "external_paper": 0.9,       # peer-reviewed paper
    "external_docs": 0.8,        # official documentation
    "external_blog": 0.6,        # blog post
    "external_web": 0.5,        # general web
    "external_forum": 0.3,       # random forum
}

@dataclass
class ProvenanceRecord:
    """Complete provenance for any extracted entity."""
    # Entity identification (required, no defaults)
    entity_type: str
    entity_value: str

    # Source information (required, no defaults)
    source_type: str
    source_url: str
    source_title: str

    # Extraction details (required, no defaults)
    extraction_confidence: float
    extraction_method: str

    # Optional fields (with defaults)
    source_author: Optional[str] = None
    source_date: Optional[str] = None
    matched_patterns: List[str] = None
    source_text_snippet: Optional[str] = None
    extracted_at: Optional[str] = None
    last_verified: Optional[str] = None
    mapped_to: Optional[str] = None
    parent_concept: Optional[str] = None
    trust_score: float = 0.0
    citation_count: Optional[int] = None

    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.now().isoformat()

        # Compute trust score
        authority = SOURCE_AUTHORITY.get(self.source_type, 0.5)
        self.trust_score = round(authority * self.extraction_confidence, 3)

        if self.matched_patterns is None:
            self.matched_patterns = []


class ProvenanceStore:
    """Store and manage provenance records."""

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            store_path = Path(__file__).parent / "provenance_store.json"

        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, ProvenanceRecord] = {}
        self._load()

    def _load(self):
        """Load existing provenance records."""
        if self.store_path.exists():
            with open(self.store_path, "r") as f:
                data = json.load(f)
                for key, record in data.items():
                    self._records[key] = ProvenanceRecord(**record)

    def _save(self):
        """Save provenance records to disk."""
        data = {k: asdict(v) for k, v in self._records.items()}
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add(self, record: ProvenanceRecord) -> str:
        """Add a provenance record. Returns the record key."""
        key = f"{record.entity_type}::{record.entity_value}::{record.source_url}"
        self._records[key] = record
        self._save()
        return key

    def get(self, entity_type: str, entity_value: str) -> List[ProvenanceRecord]:
        """Get all provenance records for an entity."""
        results = []
        prefix = f"{entity_type}::{entity_value}::"
        for key, record in self._records.items():
            if key.startswith(prefix):
                results.append(record)
        return results

    def get_by_source(self, source_type: str) -> List[ProvenanceRecord]:
        """Get all records from a specific source type."""
        return [r for r in self._records.values() if r.source_type == source_type]

    def get_trust_scores(self, entity_type: str, entity_value: str) -> Dict:
        """Get aggregate trust metrics for an entity."""
        records = self.get(entity_type, entity_value)
        if not records:
            return {"trust_score": 0, "sources": 0, "avg_confidence": 0}

        trust_scores = [r.trust_score for r in records]
        confidences = [r.extraction_confidence for r in records]
        source_types = [r.source_type for r in records]

        # Count best source type
        best_source = max(set(source_types), key=source_types.count) if source_types else "unknown"

        return {
            "trust_score": round(sum(trust_scores) / len(trust_scores), 3),
            "max_trust": max(trust_scores),
            "sources": len(records),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "best_source": best_source,
            "authority_weight": SOURCE_AUTHORITY.get(best_source, 0.5)
        }

    def list_all(self, limit: int = 100) -> List[Dict]:
        """List all provenance records."""
        records = sorted(
            self._records.values(),
            key=lambda r: r.extracted_at,
            reverse=True
        )
        return [
            {
                "key": f"{r.entity_type}::{r.entity_value}",
                "source_type": r.source_type,
                "source_title": r.source_title[:50] if r.source_title else "N/A",
                "trust_score": r.trust_score,
                "extracted_at": r.extracted_at,
                "mapped_to": r.mapped_to
            }
            for r in records[:limit]
        ]


# Global store
provenance_store = ProvenanceStore()


def add_provenance(
    entity_type: str,
    entity_value: str,
    source_type: str,
    source_url: str,
    source_title: str,
    extraction_confidence: float,
    extraction_method: str,
    matched_patterns: List[str] = None,
    source_text_snippet: str = None,
    mapped_to: str = None,
    source_author: str = None,
    source_date: str = None
) -> ProvenanceRecord:
    """Helper to add a provenance record."""
    record = ProvenanceRecord(
        entity_type=entity_type,
        entity_value=entity_value,
        source_type=source_type,
        source_url=source_url,
        source_title=source_title,
        source_author=source_author,
        source_date=source_date,
        extraction_confidence=extraction_confidence,
        extraction_method=extraction_method,
        matched_patterns=matched_patterns or [],
        source_text_snippet=source_text_snippet,
        mapped_to=mapped_to
    )

    key = provenance_store.add(record)
    return record


def get_entity_provenance(entity_type: str, entity_value: str) -> Dict:
    """Get full provenance for an entity."""
    records = provenance_store.get(entity_type, entity_value)
    trust = provenance_store.get_trust_scores(entity_type, entity_value)

    return {
        "entity": f"{entity_type}::{entity_value}",
        "records": [asdict(r) for r in records],
        "trust_metrics": trust,
        "total_sources": len(records)
    }
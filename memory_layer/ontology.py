"""
Ontology Governance
Canonical Concept Registry - prevents concept explosion.
"""

import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

@dataclass
class CanonicalConcept:
    """A canonical concept with controlled aliases."""
    canonical_id: str
    name: str
    display_name: str
    description: str
    aliases: List[str]
    category: str
    parent: Optional[str] = None
    related_concepts: Optional[List[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    usage_count: int = 0

    # Metadata
    created_at: str = None
    updated_at: str = None
    usage_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.related_concepts is None:
            self.related_concepts = []


class OntologyRegistry:
    """Manages the canonical concept registry."""

    DEFAULT_CONCEPTS = {
        # Core factor concepts
        "Concept::momentum": {
            "name": "momentum",
            "display_name": "Momentum",
            "description": "Past returns predict future returns - trend following",
            "aliases": ["momentum", "trend", "trend following", "past performance", "relative strength"],
            "category": "factor"
        },
        "Concept::mean_reversion": {
            "name": "mean_reversion",
            "display_name": "Mean Reversion",
            "description": "Prices revert to mean - contrarian strategies",
            "aliases": ["reversal", "contrarian", "short-term reversal", "price reversal"],
            "parent": None,
            "category": "factor"
        },
        "Concept::value": {
            "name": "value",
            "display_name": "Value",
            "description": "Cheapest securities outperform - intrinsic value",
            "aliases": ["value", "valuation", "cheap", "undervalued", "book value"],
            "category": "factor"
        },
        "Concept::quality": {
            "name": "quality",
            "display_name": "Quality",
            "description": "High quality companies outperform",
            "aliases": ["profitability", "ROE", "financial health", "fundamentals"],
            "category": "factor"
        },
        "Concept::volatility": {
            "name": "volatility",
            "display_name": "Volatility",
            "description": "Volatility and risk factors",
            "aliases": ["risk", "variance", "standard deviation", "dispersion"],
            "category": "factor"
        },
        "Concept::liquidity": {
            "name": "liquidity",
            "display_name": "Liquidity",
            "description": "Liquidity provision and trading volume effects",
            "aliases": ["volume", "turnover", "bid-ask", "spread"],
            "category": "factor"
        },
        "Concept::sentiment": {
            "name": "sentiment",
            "display_name": "Sentiment",
            "description": "Investor sentiment and behavioral factors",
            "aliases": ["market sentiment", "investor mood", "opinion"],
            "category": "factor"
        },
        "Concept::technical": {
            "name": "technical",
            "display_name": "Technical",
            "description": "Price action and chart patterns",
            "aliases": ["price action", "chart patterns", "technical analysis"],
            "category": "factor"
        },
        "Concept::fundamental": {
            "name": "fundamental",
            "display_name": "Fundamental",
            "description": "Fundamental analysis factors",
            "aliases": ["financials", "earnings", "financial statements"],
            "category": "factor"
        },
        # Normalization methods
        "Concept::cross_sectional": {
            "name": "cross_sectional",
            "display_name": "Cross-Sectional",
            "description": "Rank-based normalization across universe",
            "aliases": ["rank", "percentile", "relative ranking"],
            "category": "method"
        },
        "Concept::time_series": {
            "name": "time_series",
            "display_name": "Time Series",
            "description": "Historical time-based calculations",
            "aliases": ["temporal", "historical", "rolling"],
            "category": "method"
        },
        "Concept::neutralization": {
            "name": "neutralization",
            "display_name": "Neutralization",
            "description": "Removing factor exposures",
            "aliases": ["neutral", "market neutral", "factor neutral"],
            "category": "method"
        },
        "Concept::normalization": {
            "name": "normalization",
            "display_name": "Normalization",
            "description": "Standardizing values",
            "aliases": ["standardize", "z-score", "scale"],
            "category": "method"
        },
    }

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path(__file__).parent / "ontology_registry.json"

        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.concepts: Dict[str, CanonicalConcept] = {}
        self._load_or_init()

    def _load_or_init(self):
        """Load existing registry or create default."""
        if self.registry_path.exists():
            self._load()
        else:
            self._init_defaults()
            self._save()

    def _load(self):
        """Load registry from disk."""
        with open(self.registry_path, "r") as f:
            data = json.load(f)
            for key, concept_data in data.items():
                self.concepts[key] = CanonicalConcept(**concept_data)

    def _save(self):
        """Save registry to disk."""
        data = {k: asdict(v) for k, v in self.concepts.items()}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _init_defaults(self):
        """Initialize default concepts."""
        for canonical_id, config in self.DEFAULT_CONCEPTS.items():
            self.concepts[canonical_id] = CanonicalConcept(
                canonical_id=canonical_id,
                **config
            )

    def resolve(self, alias: str) -> Optional[str]:
        """
        Resolve an alias to its canonical concept ID.

        Args:
            alias: Potential alias (e.g., "contrarian", "short-term reversal")

        Returns:
            Canonical concept ID if found, None otherwise
        """
        alias_lower = alias.lower().strip()

        for canonical_id, concept in self.concepts.items():
            # Check primary name
            if concept.name.lower() == alias_lower:
                return canonical_id

            # Check aliases
            for a in concept.aliases:
                if a.lower() == alias_lower:
                    return canonical_id

        return None

    def add_concept(
        self,
        canonical_id: str,
        name: str,
        display_name: str,
        description: str,
        aliases: List[str],
        category: str,
        parent: Optional[str] = None
    ) -> bool:
        """Add a new canonical concept."""
        if canonical_id in self.concepts:
            return False  # Already exists

        self.concepts[canonical_id] = CanonicalConcept(
            canonical_id=canonical_id,
            name=name,
            display_name=display_name,
            description=description,
            aliases=aliases,
            category=category,
            parent=parent
        )
        self._save()
        return True

    def add_alias(self, canonical_id: str, alias: str) -> bool:
        """Add an alias to an existing concept."""
        if canonical_id not in self.concepts:
            return False

        if alias.lower() not in [a.lower() for a in self.concepts[canonical_id].aliases]:
            self.concepts[canonical_id].aliases.append(alias)
            self.concepts[canonical_id].updated_at = datetime.now().isoformat()
            self._save()

        return True

    def merge_aliases(self, source_id: str, target_id: str) -> bool:
        """
        Merge one concept into another (for near-duplicate consolidation).
        Moves aliases from source to target and marks source as deprecated.
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            return False

        source = self.concepts[source_id]
        target = self.concepts[target_id]

        # Move aliases
        for alias in source.aliases:
            if alias.lower() not in [a.lower() for a in target.aliases]:
                target.aliases.append(alias)

        # Mark source as deprecated (in future, add deprecated field)
        target.updated_at = datetime.now().isoformat()
        target.usage_count += source.usage_count

        # Keep source but note it's merged
        source.description += f" [DEPRECATED - merged into {target_id}]"

        self._save()
        return True

    def get_concept(self, canonical_id: str) -> Optional[CanonicalConcept]:
        """Get a concept by canonical ID."""
        return self.concepts.get(canonical_id)

    def list_concepts(self, category: Optional[str] = None) -> List[Dict]:
        """List all concepts, optionally filtered by category."""
        concepts = list(self.concepts.values())

        if category:
            concepts = [c for c in concepts if c.category == category]

        return [
            {
                "canonical_id": c.canonical_id,
                "display_name": c.display_name,
                "description": c.description[:50] + "...",
                "aliases": c.aliases,
                "category": c.category,
                "usage_count": c.usage_count
            }
            for c in concepts
        ]

    def search(self, query: str) -> List[CanonicalConcept]:
        """Search concepts by name or alias."""
        query_lower = query.lower()
        results = []

        for concept in self.concepts.values():
            # Match name
            if query_lower in concept.name.lower():
                results.append(concept)
                continue

            # Match aliases
            for alias in concept.aliases:
                if query_lower in alias.lower():
                    results.append(concept)
                    break

        return results


# Global registry
ontology = OntologyRegistry()


def resolve_to_canonical(entity_value: str) -> Optional[str]:
    """Resolve any entity value to its canonical concept ID."""
    return ontology.resolve(entity_value)


def get_canonical_info(canonical_id: str) -> Optional[Dict]:
    """Get full info about a canonical concept."""
    concept = ontology.get_concept(canonical_id)
    if concept:
        return asdict(concept)
    return None
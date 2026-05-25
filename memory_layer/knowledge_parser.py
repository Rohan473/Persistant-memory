"""
Knowledge Parser
Extracts structured research entities from scraped content.
Converts raw text into concepts, operators, factors, claims.
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

# Research ontology mappings - this is what makes the system powerful
RESEARCH_ONTOLOGY = {
    "concepts": {
        "momentum": ["momentum", "trend following", "trend", "relative strength", "past returns"],
        "mean_reversion": ["mean reversion", "reversal", "contrarian", "short-term reversal", "price reversal"],
        "value": ["value", "valuation", "book value", "price to book", "intrinsic value", "cheap"],
        "quality": ["quality", "profitability", "return on equity", "ROE", "financial health", "fundamentals"],
        "volatility": ["volatility", "variance", "risk", "standard deviation", "dispersion"],
        "liquidity": ["liquidity", "volume", "turnover", "bid-ask", "spread"],
        "sentiment": ["sentiment", "investor sentiment", "market sentiment", "behavioral", "opinion"],
        "technical": ["technical", "price action", "chart", "patterns"],
        "fundamental": ["fundamental", "fundamentals", "earnings", "financial statements", "financials"],
        "cross_sectional": ["cross-sectional", "cross section", "rank", "percentile", "relative"],
        "time_series": ["time series", "temporal", "historical", "past", "ts_"],
        "neutralization": ["neutralize", "neutral", "adjust", "remove factor", "market neutral"],
        "normalization": ["normalize", "standardize", "z-score", "rank", "scale"],
    },
    "operators": {
        "ts_rank": ["ts_rank", "time series rank", "rank over time", "rolling rank"],
        "ts_decay_linear": ["decay linear", "exponential decay", "decay", "weighted average", "decay_"],
        "ts_mean": ["moving average", "ma", "rolling mean", "sma", "ema", "ewma"],
        "ts_delta": ["delta", "difference", "change", "differencing"],
        "ts_sum": ["sum", "rolling sum", "cumulative", "total"],
        "ts_std_dev": ["standard deviation", "volatility", "variance", "std"],
        "ts_corr": ["correlation", "corr", "correl"],
        "group_rank": ["group rank", "industry rank", "sector rank", "percentile rank"],
        "group_neutralize": ["neutralize", "group neutral", "industry neutral"],
        "rank": ["rank", "percentile", "decile", "quintile"],
        "zscore": ["z-score", "zscore", "standardize", "normalize"],
    },
    "datafield_patterns": {
        "returns": ["returns", "return", "performance"],
        "volume": ["volume", "trading volume", "ADV"],
        "price": ["price", "close", "open", "high", "low"],
        "earnings": ["earnings", "EPS", "revenue", "income", "profit"],
        "book_value": ["book value", "BV", "equity", "book"],
        "momentum": ["momentum", "past return", "performance"],
    },
    "factor_categories": {
        "style": ["value", "quality", "momentum", "size", "min volatility"],
        "size": ["size", "market cap", "small cap", "large cap", "SMB"],
        "volatility": ["low volatility", "min volatility", "volatility factor"],
        "carry": ["carry", "yield", "forward return"],
    }
}

# Common quant research sources for categorization
RESEARCH_SOURCES = {
    "arxiv": "Academic Paper",
    "ssrn": "Academic Paper",
    "blog": "Blog Post",
    "medium": "Blog Post",
    "quantopian": "Research Platform",
    "alphar": "Research Platform",
    "research": "Research Paper",
    "pdf": "Research Paper",
}


@dataclass
class ResearchEntity:
    """Structured research entity extracted from content."""
    entity_type: str  # concept, operator, factor, datafield, claim
    value: str
    confidence: float  # 0-1
    source_text: str  # where found
    metadata: Dict


def extract_research_entities(markdown: str, url: str, title: str = None) -> Dict:
    """
    Extract structured research entities from markdown content.

    This is selective extraction - NOT blind chunking.
    We look for specific patterns that map to our research ontology.
    """
    source_type = _infer_source_type(url)

    entities = {
        "concepts": [],
        "operators": [],
        "factors": [],
        "datafields": [],
        "claims": [],
        "urls": [],
        "formulas": [],
        "provenance": [],  # NEW: Provenance records
        "metadata": {
            "source_url": url,
            "source_title": title or url,
            "source_type": source_type,
            "extracted_at": datetime.now().isoformat(),
        }
    }

    # Extract concepts
    entities["concepts"] = _extract_concepts(markdown)

    # Extract operators
    entities["operators"] = _extract_operators(markdown)

    # Extract formulas (WQB-style expressions)
    entities["formulas"] = _extract_formulas(markdown)

    # Extract claims (quantitative statements)
    entities["claims"] = _extract_claims(markdown)

    # Extract any URLs for further crawling
    entities["urls"] = _extract_urls(markdown)

    # Build provenance records for each extracted entity
    entities["provenance"] = _build_provenance(
        entities["concepts"],
        entities["operators"],
        url,
        title or url,
        source_type
    )

    # Count statistics
    entities["metadata"]["concept_count"] = len(entities["concepts"])
    entities["metadata"]["operator_count"] = len(entities["operators"])
    entities["metadata"]["formula_count"] = len(entities["formulas"])
    entities["metadata"]["provenance_count"] = len(entities["provenance"])

    return entities


def _extract_concepts(markdown: str) -> List[Dict]:
    """Extract research concepts using ontology mapping."""
    found = []
    markdown_lower = markdown.lower()

    for concept, patterns in RESEARCH_ONTOLOGY["concepts"].items():
        for pattern in patterns:
            # Count occurrences
            count = markdown_lower.count(pattern.lower())
            if count > 0:
                confidence = min(1.0, count / 3)  # Cap at 1.0
                found.append({
                    "concept": concept,
                    "pattern_matched": pattern,
                    "occurrences": count,
                    "confidence": round(confidence, 2)
                })

    # Deduplicate and sort by confidence
    unique = {}
    for f in found:
        key = f["concept"]
        if key not in unique or unique[key]["confidence"] < f["confidence"]:
            unique[key] = f

    return sorted(unique.values(), key=lambda x: x["confidence"], reverse=True)


def _extract_operators(markdown: str) -> List[Dict]:
    """Extract WQB-style operators."""
    found = []
    markdown_lower = markdown.lower()

    for operator, patterns in RESEARCH_ONTOLOGY["operators"].items():
        for pattern in patterns:
            count = markdown_lower.count(pattern.lower())
            if count > 0:
                confidence = min(1.0, count / 2)
                found.append({
                    "operator": operator,
                    "pattern_matched": pattern,
                    "occurrences": count,
                    "confidence": round(confidence, 2)
                })

    unique = {}
    for f in found:
        key = f["operator"]
        if key not in unique or unique[key]["confidence"] < f["confidence"]:
            unique[key] = f

    return sorted(unique.values(), key=lambda x: x["confidence"], reverse=True)


def _extract_formulas(markdown: str) -> List[Dict]:
    """Extract WQB-style alpha expressions."""
    formulas = []

    # Pattern for WQB-style expressions
    patterns = [
        # Function calls like ts_rank(x, n)
        r'\b(ts_\w+|rank|zscore|group_rank|group_neutralize|decay_linear|decay_exp|if_else|sign|abs|log|power|max|min)\s*\([^)]+\)',
        # Simple expressions
        r'(?:rank|zscore)\s*\([^)]+\)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, markdown, re.IGNORECASE)
        for match in matches[:10]:  # Limit to 10 per document
            formulas.append({
                "expression": match,
                "type": "operator_expression"
            })

    return formulas


def _extract_claims(markdown: str) -> List[Dict]:
    """Extract quantitative claims/statements."""
    claims = []

    # Patterns for quantitative claims
    claim_patterns = [
        # Sharpe ratios, returns
        r'Sharpe[:\s]+(\d+\.?\d*)',
        r'return[:\s]+(\d+\.?\d*%)?',
        r'IC[:\s]+(-?\d+\.?\d*)',
        # Factor names with returns
        r'([a-zA-Z]+)\s+factor\s+generates?\s+(\d+\.?\d*%)?',
        r'([a-zA-Z]+)\s+signal\s+produces?\s+(\d+\.?\d*%)?',
    ]

    for pattern in claim_patterns:
        matches = re.finditer(pattern, markdown, re.IGNORECASE)
        for match in matches:
            claims.append({
                "claim": match.group(0),
                "pattern_type": pattern[:30],
                "confidence": 0.6
            })

    return claims[:20]  # Limit to 20 claims


def _extract_urls(markdown: str) -> List[str]:
    """Extract relevant URLs from markdown."""
    url_pattern = r'https?://[^\s\)\]]+'
    urls = re.findall(url_pattern, markdown)

    # Filter to likely research URLs
    research_keywords = ["arxiv", "ssrn", "pdf", "research", "blog", "medium", "quant"]
    relevant = [u for u in urls if any(k in u.lower() for k in research_keywords)]

    return relevant[:20]


def _build_provenance(concepts: List[Dict], operators: List[Dict], url: str, title: str, source_type: str) -> List[Dict]:
    """
    Build provenance records for extracted entities.
    This ensures every concept knows its source.
    """
    provenance_records = []

    # Map source type to provenance format
    source_type_map = {
        "Academic Paper": "external_paper",
        "Blog Post": "external_blog",
        "Research Platform": "external_docs",
        "Web Content": "external_web"
    }
    prov_source_type = source_type_map.get(source_type, "external_web")

    # Process concepts
    for concept in concepts:
        matched = concept.get("pattern_matched", "")
        confidence = concept.get("confidence", 0.5)

        # Try to resolve to canonical concept
        canonical_id = None
        try:
            from .ontology import resolve_to_canonical
            canonical_id = resolve_to_canonical(concept.get("concept", ""))
        except:
            pass

        provenance_records.append({
            "entity_type": "concept",
            "entity_value": concept.get("concept", ""),
            "source_type": prov_source_type,
            "source_url": url,
            "source_title": title,
            "extraction_confidence": confidence,
            "extraction_method": "ontology_match",
            "matched_patterns": [matched],
            "mapped_to": canonical_id
        })

    # Process operators
    for operator in operators:
        matched = operator.get("pattern_matched", "")
        confidence = operator.get("confidence", 0.5)

        provenance_records.append({
            "entity_type": "operator",
            "entity_value": operator.get("operator", ""),
            "source_type": prov_source_type,
            "source_url": url,
            "source_title": title,
            "extraction_confidence": confidence,
            "extraction_method": "ontology_match",
            "matched_patterns": [matched],
            "mapped_to": f"Operator::{operator.get('operator', '')}"
        })

    return provenance_records


def _infer_source_type(url: str) -> str:
    """Infer the type of research source from URL."""
    url_lower = url.lower()

    for keyword, source_type in RESEARCH_SOURCES.items():
        if keyword in url_lower:
            return source_type

    return "Web Content"


def merge_with_internal_memory(external_entities: Dict) -> Dict:
    """
    Merge external research entities with internal knowledge graph.

    This creates links between external research and internal concepts.
    """
    from .structure import load_metadata

    try:
        internal_metadata = load_metadata()
    except:
        internal_metadata = []

    # Index internal concepts
    internal_concepts = set()
    internal_operators = set()

    for m in internal_metadata:
        if m.get("node_type") == "Concept":
            internal_concepts.add(m.get("name", ""))
        elif m.get("node_type") == "Operator":
            internal_operators.add(m.get("name", ""))

    # Find matches
    matches = {
        "concepts_matched": [],
        "operators_matched": [],
        "new_concepts": [],
        "new_operators": [],
    }

    # Check external concepts against internal
    for ec in external_entities.get("concepts", []):
        concept = ec.get("concept", "")
        if concept in internal_concepts:
            matches["concepts_matched"].append(concept)
        else:
            matches["new_concepts"].append(concept)

    # Check external operators against internal
    for eo in external_entities.get("operators", []):
        op = eo.get("operator", "")
        if op in internal_operators:
            matches["operators_matched"].append(op)
        else:
            matches["new_operators"].append(op)

    external_entities["internal_links"] = matches

    return external_entities
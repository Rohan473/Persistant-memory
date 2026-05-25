"""
Natural Language Research Query System
Process queries like "Find recovery-quality alphas" into graph/vector searches.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class QueryIntent(Enum):
    FIND_ALPHAS = "find_alphas"
    FIND_OPERATORS = "find_operators"
    FIND_CONCEPTS = "find_concepts"
    EXPLAIN_FAILURE = "explain_failure"
    FIND_REGIME = "find_regime"
    FIND_ORTHOGONAL = "find_orthogonal"
    FIND_RELATED = "find_related"
    GET_STATS = "get_stats"
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    """Parsed natural language query."""
    intent: QueryIntent
    entities: Dict[str, List[str]]
    filters: Dict[str, Any]
    raw_query: str


class NLQueryParser:
    """Parse natural language queries into structured requests."""

    REGIME_KEYWORDS = {
        "recovery": ["recovery", "rebound", "post-crisis", "bounce back"],
        "crisis": ["crisis", "recession", "crash", "downturn", "stress"],
        "inflation": ["inflation", "inflationary", "rising rates", "rate hike"],
        "deflation": ["deflation", "falling prices", "disinflation"],
        "growth_leadership": ["growth", "growth regime", "tech rally"],
        "value_rotation": ["value", "value rotation", "value rally"],
        "volatility_spike": ["volatile", "volatility", "VIX", "turbulent"],
        "risk_on": ["risk-on", "risk on", "bullish", "risk appetite"],
        "risk_off": ["risk-off", "risk off", "defensive", "flight to safety"]
    }

    FACTOR_KEYWORDS = {
        "quality": ["quality", "profitability", "ROE", "high margin"],
        "value": ["value", "cheap", "undervalued", "valuation"],
        "momentum": ["momentum", "trend", "relative strength"],
        "reversal": ["reversal", "mean reversion", "contrarian"],
        "liquidity": ["liquidity", "volume", "trading activity"],
        "volatility": ["volatility", "risk", "dispersion", "low vol"],
        "growth": ["growth", "earnings growth", "revenue growth"],
        "defensive": ["defensive", "low beta", "stable earnings"]
    }

    FAILURE_KEYWORDS = {
        "high_turnover": ["turnover", "too much trading", "high turnover"],
        "low_sharpe": ["low sharpe", "poor sharpe", "below threshold"],
        "low_fitness": ["fitness", "fitness failure"],
        "correlated": ["correlated", "self-correlation", "correlation too high"],
        "sector_bias": ["sector bias", "sector exposure", "industry"],
        "regime_collapse": ["regime", "crisis collapse", "regime failure"]
    }

    def parse(self, query: str) -> ParsedQuery:
        """Parse natural language query."""
        query_lower = query.lower()

        entities = {
            "factors": [],
            "operators": [],
            "concepts": [],
            "regimes": [],
            "failure_modes": [],
            "datafields": []
        }

        filters = {}

        if any(kw in query_lower for kw in ["find", "search", "show", "list", "get", "which", "what"]):
            if "recovery" in query_lower and any(kw in query_lower for kw in ["quality", "value", "momentum"]):
                entities["regimes"].append("recovery")
                if "quality" in query_lower:
                    entities["factors"].append("quality")
                if "value" in query_lower:
                    entities["factors"].append("value")
                if "momentum" in query_lower:
                    entities["factors"].append("momentum")
                intent = QueryIntent.FIND_ALPHAS

            elif any(r in query_lower for r in self.REGIME_KEYWORDS.keys()):
                for regime, keywords in self.REGIME_KEYWORDS.items():
                    if any(kw in query_lower for kw in keywords):
                        entities["regimes"].append(regime)
                intent = QueryIntent.FIND_REGIME

            elif any(f in query_lower for f in self.FACTOR_KEYWORDS.keys()):
                for factor, keywords in self.FACTOR_KEYWORDS.items():
                    if any(kw in query_lower for kw in keywords):
                        entities["factors"].append(factor)
                if entities["regimes"]:
                    intent = QueryIntent.FIND_ALPHAS
                else:
                    intent = QueryIntent.FIND_ALPHAS

            elif any(fm in query_lower for fm in self.FAILURE_KEYWORDS.keys()):
                for fm, keywords in self.FAILURE_KEYWORDS.items():
                    if any(kw in query_lower for kw in keywords):
                        entities["failure_modes"].append(fm)
                intent = QueryIntent.FIND_ALPHAS

            elif "operators" in query_lower or "operator" in query_lower:
                for op in ["ts_rank", "ts_mean", "rank", "group_neutralize", "ts_corr", "ts_decay_linear"]:
                    if op in query_lower:
                        entities["operators"].append(op)
                intent = QueryIntent.FIND_OPERATORS

            elif "orthogonal" in query_lower or "uncorrelated" in query_lower:
                intent = QueryIntent.FIND_ORTHOGONAL

            else:
                intent = QueryIntent.FIND_ALPHAS

        elif "why" in query_lower and ("fail" in query_lower or "collapsed" in query_lower):
            intent = QueryIntent.EXPLAIN_FAILURE

        elif "stats" in query_lower or "statistics" in query_lower:
            intent = QueryIntent.GET_STATS

        else:
            intent = QueryIntent.UNKNOWN

        if "sharpe" in query_lower:
            match = re.search(r"sharpe\s*>?\s*(\d+\.?\d*)", query_lower)
            if match:
                filters["min_sharpe"] = float(match.group(1))

        if "turnover" in query_lower:
            match = re.search(r"turnover\s*<?\s*(\d+\.?\d*)", query_lower)
            if match:
                filters["max_turnover"] = float(match.group(1))

        if "top" in query_lower:
            match = re.search(r"top\s*(\d+)", query_lower)
            if match:
                filters["limit"] = int(match.group(1))
            else:
                filters["limit"] = 10
        else:
            filters["limit"] = 5

        return ParsedQuery(
            intent=intent,
            entities=entities,
            filters=filters,
            raw_query=query
        )


class QueryExecutor:
    """Execute parsed queries against the knowledge graph."""

    def __init__(self):
        self.parser = NLQueryParser()
        self._load_sources()

    def _load_sources(self):
        """Load graph and vector sources."""
        self.graph = None
        self.metadata = []
        self.alpha_index = {}

        try:
            from memory_layer.structure import load_metadata, load_graph
            self.metadata = load_metadata()
            self.graph = load_graph()
            self._build_alpha_index()
        except Exception as e:
            print(f"Warning: Could not load graph sources: {e}")

    def _build_alpha_index(self):
        """Build alpha lookup index."""
        for m in self.metadata:
            if m.get("node_type") == "Alpha":
                self.alpha_index[m.get("name", "")] = m

    def execute(self, query: str) -> Dict:
        """Execute natural language query."""
        parsed = self.parser.parse(query)

        if parsed.intent == QueryIntent.FIND_ALPHAS:
            return self._find_alphas(parsed)
        elif parsed.intent == QueryIntent.FIND_REGIME:
            return self._find_regime_alphas(parsed)
        elif parsed.intent == QueryIntent.FIND_ORTHOGONAL:
            return self._find_orthogonal(parsed)
        elif parsed.intent == QueryIntent.EXPLAIN_FAILURE:
            return self._explain_failure(parsed)
        elif parsed.intent == QueryIntent.GET_STATS:
            return self._get_stats()
        else:
            return {"error": "Could not understand query", "query": query}

    def _find_alphas(self, parsed: ParsedQuery) -> Dict:
        """Find alphas matching criteria."""
        results = []

        for alpha_id, alpha in self.alpha_index.items():
            match = True

            if parsed.entities.get("factors"):
                alpha_factors = set(alpha.get("concepts", []))
                if not any(f in alpha_factors for f in parsed.entities["factors"]):
                    match = False

            if parsed.entities.get("failure_modes"):
                alpha_failures = set(alpha.get("failure_modes", []))
                if parsed.filters.get("exclude_failures"):
                    if any(f in alpha_failures for f in parsed.entities["failure_modes"]):
                        match = False
                else:
                    if not any(f in alpha_failures for f in parsed.entities["failure_modes"]):
                        match = False

            if parsed.filters.get("min_sharpe"):
                if (alpha.get("sharpe") or 0) < parsed.filters["min_sharpe"]:
                    match = False

            if parsed.filters.get("max_turnover"):
                if (alpha.get("turnover") or 999) > parsed.filters["max_turnover"]:
                    match = False

            if match:
                results.append({
                    "alpha_id": alpha_id,
                    "sharpe": alpha.get("sharpe"),
                    "turnover": alpha.get("turnover"),
                    "fitness": alpha.get("fitness"),
                    "expression": alpha.get("expression", "")[:60],
                    "concepts": alpha.get("concepts", [])[:3],
                    "failure_modes": alpha.get("failure_modes", [])[:2]
                })

        results.sort(key=lambda x: x.get("sharpe") or 0, reverse=True)
        limit = parsed.filters.get("limit", 10)
        results = results[:limit]

        return {
            "query": parsed.raw_query,
            "intent": parsed.intent.value,
            "results": results,
            "count": len(results),
            "explanation": self._generate_find_explanation(parsed, len(results))
        }

    def _find_regime_alphas(self, parsed: ParsedQuery) -> Dict:
        """Find alphas for specific regimes."""
        from memory_layer.regime_analysis import analyzer

        regimes = parsed.entities.get("regimes", [])
        results = []

        for regime in regimes:
            alphas = analyzer.find_alphas_by_regime(regime, min_sharpe=0.5)
            for a in alphas[:parsed.filters.get("limit", 5)]:
                results.append({
                    "alpha_id": a["alpha_id"],
                    "regime": regime,
                    "sharpe": a.get("avg_sharpe"),
                    "sample_years": a.get("sample_years")
                })

        return {
            "query": parsed.raw_query,
            "intent": parsed.intent.value,
            "regimes": regimes,
            "results": results,
            "count": len(results),
            "explanation": f"Found {len(results)} alphas performing well in {', '.join(regimes)} regimes"
        }

    def _find_orthogonal(self, parsed: ParsedQuery) -> Dict:
        """Find orthogonal alphas."""
        target = parsed.entities.get("target_alpha", [])
        candidates = list(self.alpha_index.keys())

        if not target:
            return {
                "query": parsed.raw_query,
                "results": [],
                "explanation": "Please specify which alpha to find orthogonal alternatives for"
            }

        from memory_layer.correlation_engine import correlation_engine

        orthogonal = correlation_engine.find_orthogonal_alphas(target[0], max_correlation=0.3)

        results = []
        for o in orthogonal[:5]:
            if o["alpha_id"] in self.alpha_index:
                a = self.alpha_index[o["alpha_id"]]
                results.append({
                    "alpha_id": o["alpha_id"],
                    "correlation": o["correlation"],
                    "factor_overlap": o.get("factor_overlap", 0),
                    "sharpe": a.get("sharpe")
                })

        return {
            "query": parsed.raw_query,
            "target": target[0],
            "results": results,
            "count": len(results),
            "explanation": f"Found {len(results)} orthogonal alternatives to {target[0]}"
        }

    def _explain_failure(self, parsed: ParsedQuery) -> Dict:
        """Explain alpha failures."""
        alpha_id = parsed.entities.get("target_alpha", [""])[0]

        if not alpha_id:
            return {
                "query": parsed.raw_query,
                "error": "Please specify which alpha to explain"
            }

        if alpha_id not in self.alpha_index:
            return {
                "query": parsed.raw_query,
                "error": f"Alpha {alpha_id} not found"
            }

        alpha = self.alpha_index[alpha_id]
        from memory_layer.research_copilot import explain_alpha_failure

        explanation = explain_alpha_failure(
            alpha_id=alpha_id,
            failure_modes=alpha.get("failure_modes", []),
            sharpe=alpha.get("sharpe"),
            turnover=alpha.get("turnover"),
            fitness=alpha.get("fitness"),
            expression=alpha.get("expression", ""),
            operators=alpha.get("operators", []),
            datafields=alpha.get("datafields", [])
        )

        return {
            "query": parsed.raw_query,
            "alpha_id": alpha_id,
            "explanation": explanation,
            "sharpe": alpha.get("sharpe"),
            "turnover": alpha.get("turnover"),
            "fitness": alpha.get("fitness"),
            "failure_modes": alpha.get("failure_modes", [])
        }

    def _get_stats(self) -> Dict:
        """Get graph statistics."""
        counts = defaultdict(int)
        for m in self.metadata:
            counts[m.get("node_type", "unknown")] += 1

        return {
            "query": parsed.raw_query if 'parsed' in locals() else "stats",
            "node_counts": dict(counts),
            "total_nodes": len(self.metadata)
        }

    def _generate_find_explanation(self, parsed: ParsedQuery, count: int) -> str:
        """Generate human-readable explanation."""
        parts = [f"Found {count} alphas"]

        if parsed.entities.get("factors"):
            parts.append(f"with factors: {', '.join(parsed.entities['factors'])}")

        if parsed.entities.get("regimes"):
            parts.append(f"in regimes: {', '.join(parsed.entities['regimes'])}")

        if parsed.entities.get("failure_modes"):
            parts.append(f"having failures: {', '.join(parsed.entities['failure_modes'])}")

        return " ".join(parts)


query_executor = QueryExecutor()


def process_query(query: str) -> Dict:
    """Process a natural language query."""
    return query_executor.execute(query)


def parse_query(query: str) -> ParsedQuery:
    """Parse query without executing."""
    return query_executor.parser.parse(query)
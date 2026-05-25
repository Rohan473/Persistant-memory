"""
MCP (Model Context Protocol) Server for Memory Layer
Lightweight JSON-RPC 2.0 compatible interface for external agents.
"""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from .config import config
from .retrieve import retrieve_hybrid, format_context
from .budget import compile_context, estimate_tokens_precise
from .ingest import get_stats

@dataclass
class MCPRequest:
    jsonrpc: str
    method: str
    params: Optional[Dict] = None
    id: Optional[Union[str, int]] = None

@dataclass  
class MCPResponse:
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict] = None
    id: Optional[Union[str, int]] = None

class MCPToolRegistry:
    """Registry of available MCP tools."""

    TOOLS = {
        "search_research_memory": {
            "description": "Semantic search across research knowledge graph",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "k": {"type": "integer", "default": 5, "description": "Number of results"},
                    "node_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by node types"},
                    "operators": {"type": "array", "items": {"type": "string"}, "description": "Filter by operators"},
                    "concepts": {"type": "array", "items": {"type": "string"}, "description": "Filter by concepts"},
                },
                "required": ["query"]
            }
        },
        "get_research_context": {
            "description": "Get LLM-ready context from retrieved memories within token budget",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "default": 3, "description": "Number of memories"},
                    "budget_tokens": {"type": "integer", "default": 2000, "description": "Max tokens"},
                    "format": {"type": "string", "enum": ["full", "compact"], "default": "full"}
                },
                "required": ["query"]
            }
        },
        "search_by_symbolic": {
            "description": "Pure symbolic retrieval - filter by operators, concepts, settings, rating without vector search",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operators": {"type": "array", "items": {"type": "string"}, "description": "Filter by operators (e.g., ts_rank, decay_linear)"},
                    "concepts": {"type": "array", "items": {"type": "string"}, "description": "Filter by concepts (e.g., mean_reversion, momentum)"},
                    "failure_modes": {"type": "array", "items": {"type": "string"}, "description": "Filter by failure modes (e.g., low_sharpe, high_turnover)"},
                    "universes": {"type": "array", "items": {"type": "string"}, "description": "Filter by universe settings (e.g., TOP3000_subindustry, TOP500_industry)"},
                    "rating": {"type": "string", "enum": ["Good", "Average", "Needs Improvement"], "description": "Filter by alpha rating"},
                    "datafields": {"type": "array", "items": {"type": "string"}, "description": "Filter by datafields used"},
                    "has_expression": {"type": "boolean", "description": "Only return alphas with expressions"}
                }
            }
        },
        "get_alpha_lineage": {
            "description": "Get derivation chain for an alpha",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_name": {"type": "string", "description": "Alpha name (e.g., alpha_0020)"}
                },
                "required": ["alpha_name"]
            }
        },
        "get_memory_stats": {
            "description": "Get statistics about the research memory system",
            "input_schema": {"type": "object", "properties": {}}
        },
        "compile_research_state": {
            "description": "Compile structured research state from query - produces machine-assisted research cognition with concepts, failures, lineage, and recommendations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                    "k": {"type": "integer", "default": 5, "description": "Number of results to consider"},
                    "budget_tokens": {"type": "integer", "default": 2000, "description": "Token budget for context"},
                    "rerank": {"type": "boolean", "default": True, "description": "Apply hybrid reranking"}
                },
                "required": ["query"]
            }
        },
        "list_tools": {
            "description": "List all available MCP tools",
            "input_schema": {"type": "object", "properties": {}}
        },
        "classify_alpha_factors": {
            "description": "Classify alpha into canonical factor families",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"},
                    "expression": {"type": "string"},
                    "datafields": {"type": "array", "items": {"type": "string"}},
                    "operators": {"type": "array", "items": {"type": "string"}},
                    "concepts": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["alpha_id", "expression", "datafields", "operators"]
            }
        },
        "get_regime_performance": {
            "description": "Get alpha performance breakdown by market regime",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"}
                },
                "required": ["alpha_id"]
            }
        },
        "find_regime_alphas": {
            "description": "Find alphas that perform well in a specific regime",
            "input_schema": {
                "type": "object",
                "properties": {
                    "regime": {"type": "string"},
                    "min_sharpe": {"type": "number", "default": 0.5}
                },
                "required": ["regime"]
            }
        },
        "explain_alpha_failure": {
            "description": "Explain why an alpha failed with institutional-grade reasoning",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"},
                    "failure_modes": {"type": "array", "items": {"type": "string"}},
                    "sharpe": {"type": "number"},
                    "turnover": {"type": "number"},
                    "fitness": {"type": "number"},
                    "expression": {"type": "string"},
                    "operators": {"type": "array", "items": {"type": "string"}},
                    "datafields": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["alpha_id", "failure_modes"]
            }
        },
        "find_orthogonal_alphas": {
            "description": "Find orthogonal (uncorrelated) alpha alternatives",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"},
                    "max_correlation": {"type": "number", "default": 0.3}
                },
                "required": ["alpha_id"]
            }
        },
        "detect_hidden_exposures": {
            "description": "Detect hidden factor exposures in alpha",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "datafields": {"type": "array", "items": {"type": "string"}},
                    "operators": {"type": "array", "items": {"type": "string"}},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                    "neutralization": {"type": "string"}
                },
                "required": ["expression", "datafields", "operators"]
            }
        },
        "natural_language_query": {
            "description": "Process natural language research queries",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        },
        "get_exploration_coverage": {
            "description": "Get research exploration coverage statistics",
            "input_schema": {"type": "object", "properties": {}}
        },
        "suggest_factor_combinations": {
            "description": "Suggest unexplored factor combinations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "used_factors": {"type": "array", "items": {"type": "string"}},
                    "used_datafields": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["used_factors"]
            }
        },
        "register_alpha_lineage": {
            "description": "Register a new alpha with lineage tracking",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "branch_id": {"type": "string"}
                },
                "required": ["alpha_id"]
            }
        },
        "get_alpha_lineage": {
            "description": "Get complete lineage for an alpha",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"}
                },
                "required": ["alpha_id"]
            }
        },
        "compare_alpha_lineage": {
            "description": "Compare two alphas in the same lineage",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_1": {"type": "string"},
                    "alpha_2": {"type": "string"}
                },
                "required": ["alpha_1", "alpha_2"]
            }
        },
        "get_experiment_tree": {
            "description": "Get full experiment tree starting from a root alpha",
            "input_schema": {
                "type": "object",
                "properties": {
                    "root_alpha_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 5}
                },
                "required": ["root_alpha_id"]
            }
        },
        "get_lineage_stats": {
            "description": "Get lineage tracking statistics",
            "input_schema": {"type": "object", "properties": {}}
        },
        "generate_hypothesis": {
            "description": "Generate a new research hypothesis",
            "input_schema": {
                "type": "object",
                "properties": {
                    "focus_area": {"type": "string"},
                    "based_on_alpha": {"type": "string"}
                }
            }
        },
        "analyze_failed_alpha_research": {
            "description": "Analyze failed alpha and propose improvements",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"}
                },
                "required": ["alpha_id"]
            }
        },
        "find_orthogonal_sleeve": {
            "description": "Find orthogonal sleeve for an alpha",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha_id": {"type": "string"},
                    "max_candidates": {"type": "integer", "default": 5}
                },
                "required": ["alpha_id"]
            }
        },
        "detect_research_gaps": {
            "description": "Detect unexplored research areas",
            "input_schema": {"type": "object", "properties": {}}
        },
        "maintain_research_diversity": {
            "description": "Analyze and maintain research diversity",
            "input_schema": {
                "type": "object",
                "properties": {
                    "current_alphas": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "get_agent_stats": {
            "description": "Get research agent statistics",
            "input_schema": {"type": "object", "properties": {}}
        }
    }

    @classmethod
    def list_tools(cls) -> List[Dict]:
        """Return list of tool definitions."""
        return [
            {"name": name, **tool}
            for name, tool in cls.TOOLS.items()
        ]

    @classmethod
    def get_tool(cls, name: str) -> Optional[Dict]:
        """Get tool definition by name."""
        return cls.TOOLS.get(name)


class MCPServer:
    """MCP Server handling JSON-RPC 2.0 requests."""

    def __init__(self):
        self.registry = MCPToolRegistry()

    def handle_request(self, request: Dict) -> MCPResponse:
        """Handle incoming MCP request."""
        try:
            # Parse request
            method = request.get("method")
            params = request.get("params", {})
            req_id = request.get("id")

            if not method:
                return MCPResponse(
                    error={"code": -32600, "message": "Invalid Request: method required"},
                    id=req_id
                )

            # Handle method
            if method == "tools/list":
                result = {"tools": self.registry.list_tools()}
            elif method == "tools/call":
                result = self._call_tool(params)
            elif method == "resources/list":
                result = {"resources": []}  # Not implemented yet
            else:
                # Try to call as direct tool
                result = self._call_tool({"name": method, **params})
            
            return MCPResponse(result=result, id=req_id)

        except Exception as e:
            return MCPResponse(
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
                id=request.get("id")
            )

    def _call_tool(self, params: Dict) -> Any:
        """Call a specific tool with parameters."""
        import time
        start_time = time.time()

        tool_name = params.get("name")
        tool_params = {k: v for k, v in params.items() if k != "name"}

        tool_def = self.registry.get_tool(tool_name)
        if not tool_def:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Route to handler
        result = None
        if tool_name == "search_research_memory":
            result = self._tool_search(tool_params)
        elif tool_name == "get_research_context":
            result = self._tool_context(tool_params)
        elif tool_name == "search_by_symbolic":
            result = self._tool_symbolic(tool_params)
        elif tool_name == "get_alpha_lineage":
            result = self._tool_lineage(tool_params)
        elif tool_name == "get_memory_stats":
            result = self._tool_stats()
        elif tool_name == "compile_research_state":
            result = self._tool_research_state(tool_params)
        elif tool_name == "classify_alpha_factors":
            result = self._tool_classify_factors(tool_params)
        elif tool_name == "get_regime_performance":
            result = self._tool_regime_performance(tool_params)
        elif tool_name == "find_regime_alphas":
            result = self._tool_find_regime_alphas(tool_params)
        elif tool_name == "explain_alpha_failure":
            result = self._tool_explain_failure(tool_params)
        elif tool_name == "find_orthogonal_alphas":
            result = self._tool_orthogonal_alphas(tool_params)
        elif tool_name == "detect_hidden_exposures":
            result = self._tool_hidden_exposures(tool_params)
        elif tool_name == "natural_language_query":
            result = self._tool_nl_query(tool_params)
        elif tool_name == "get_exploration_coverage":
            result = self._tool_exploration_coverage(tool_params)
        elif tool_name == "suggest_factor_combinations":
            result = self._tool_factor_suggestions(tool_params)
        elif tool_name == "register_alpha_lineage":
            result = self._tool_register_lineage(tool_params)
        elif tool_name == "get_alpha_lineage":
            result = self._tool_get_lineage(tool_params)
        elif tool_name == "compare_alpha_lineage":
            result = self._tool_compare_lineage(tool_params)
        elif tool_name == "get_experiment_tree":
            result = self._tool_experiment_tree(tool_params)
        elif tool_name == "get_lineage_stats":
            result = self._tool_lineage_stats(tool_params)
        elif tool_name == "generate_hypothesis":
            result = self._tool_generate_hypothesis(tool_params)
        elif tool_name == "analyze_failed_alpha_research":
            result = self._tool_analyze_failed_alpha(tool_params)
        elif tool_name == "find_orthogonal_sleeve":
            result = self._tool_find_orthogonal_sleeve(tool_params)
        elif tool_name == "detect_research_gaps":
            result = self._tool_detect_gaps(tool_params)
        elif tool_name == "maintain_research_diversity":
            result = self._tool_maintain_diversity(tool_params)
        elif tool_name == "get_agent_stats":
            result = self._tool_agent_stats(tool_params)
        else:
            raise ValueError(f"Tool handler not implemented: {tool_name}")

        # Log retrieval event
        latency_ms = round((time.time() - start_time) * 1000, 1)
        self._log_trace(tool_name, tool_params, result, latency_ms)

        return result

    def _log_trace(self, tool_name: str, params: Dict, result: Dict, latency_ms: float) -> None:
        """Log retrieval event for training data."""
        try:
            from .trace import RetrievalEvent
            from datetime import datetime

            retrieved = result.get("results", []) if isinstance(result, dict) else []

            event = RetrievalEvent(
                timestamp=datetime.now().isoformat(),
                query=params.get("query", params.get("alpha_name", "")),
                tool_used=tool_name,
                retrieved_ids=[r.get("id", "") for r in retrieved],
                retrieved_names=[r.get("name", "") for r in retrieved],
                scores=[round(r.get("score", 0), 3) for r in retrieved],
                token_budget=params.get("budget_tokens", 2000),
                tokens_used=result.get("tokens", 0) if isinstance(result, dict) else 0,
                compression_ratio=0,
                latency_ms=latency_ms,
                filters={k: v for k, v in params.items() if k not in ["query", "k", "budget_tokens", "format"]}
            )

            from .trace import logger
            logger.log(event)
        except Exception:
            pass  # Don't fail if tracing fails

    def _tool_search(self, params: Dict) -> Dict:
        """Handle search_research_memory tool."""
        from .retrieve import retrieve_hybrid
        
        results = retrieve_hybrid(
            query=params.get("query", ""),
            k=params.get("k", 5),
            operators=params.get("operators"),
            concepts=params.get("concepts"),
            node_types=params.get("node_types")
        )
        
        # Format for external agents
        return {
            "query": params.get("query"),
            "count": len(results),
            "results": [
                {
                    "name": r.get("name"),
                    "type": r.get("node_type"),
                    "summary": r.get("structured_summary"),
                    "score": round(r.get("score", 0), 3),
                    "retrieval_score": round(r.get("retrieval_score", 0), 3),
                    "operators": r.get("operators", [])[:3],
                    "concepts": r.get("concepts", [])[:3],
                    "failure_modes": r.get("failure_modes", [])[:2],
                }
                for r in results
            ]
        }

    def _tool_context(self, params: Dict) -> Dict:
        """Handle get_research_context tool."""
        from .retrieve import retrieve_hybrid, format_context
        from .budget import compile_context
        
        results = retrieve_hybrid(
            query=params.get("query", ""),
            k=params.get("k", 3)
        )
        
        format_type = params.get("format", "full")
        budget = params.get("budget_tokens", 2000)
        
        if format_type == "compact":
            from .budget import compile_context_compact
            context, tokens = compile_context_compact(results, budget_tokens=budget)
            return {
                "query": params.get("query"),
                "context": context,
                "tokens": tokens,
                "memories_used": len(results)
            }
        else:
            context = format_context(results, max_memories=params.get("k", 3))
            compiled, tokens, used = compile_context(results, budget_tokens=budget)
            return {
                "query": params.get("query"),
                "context": context,
                "compiled": compiled,
                "tokens": tokens,
                "memories_used": len(used)
            }

    def _tool_symbolic(self, params: Dict) -> Dict:
        """Handle search_by_symbolic tool - pure metadata filtering."""
        from .retrieve import retrieve_hybrid
        
        # Build query from symbolic filters
        parts = []
        if params.get("operators"):
            parts.extend(params["operators"])
        if params.get("concepts"):
            parts.extend(params["concepts"])
        
        query = " ".join(parts) if parts else "*"
        
        results = retrieve_hybrid(
            query=query,
            k=10,
            operators=params.get("operators"),
            concepts=params.get("concepts")
        )
        
        # Additional filtering
        if params.get("failure_modes"):
            results = [r for r in results if r.get("failure_modes")]
            results = [r for r in results if set(r.get("failure_modes", [])).intersection(set(params["failure_modes"]))]
        
        if params.get("rating"):
            results = [r for r in results if r.get("rating") == params["rating"]]
        
        return {
            "filter": params,
            "count": len(results),
            "results": [
                {
                    "name": r.get("name"),
                    "type": r.get("node_type"),
                    "operators": r.get("operators", []),
                    "concepts": r.get("concepts", []),
                    "failure_modes": r.get("failure_modes", []),
                    "rating": r.get("rating"),
                }
                for r in results
            ]
        }

    def _tool_lineage(self, params: Dict) -> Dict:
        """Handle get_alpha_lineage tool."""
        from .structure import load_metadata
        
        alpha_name = params.get("alpha_name", "")
        metadata = load_metadata()
        
        # Find the alpha
        alpha = None
        for m in metadata:
            if m.get("name") == alpha_name and m.get("node_type") == "Alpha":
                alpha = m
                break
        
        if not alpha:
            return {"error": f"Alpha not found: {alpha_name}"}
        
        # Build lineage
        lineage = {
            "alpha": alpha_name,
            "derived_from": alpha.get("derived_from", []),
            "correlated_with": alpha.get("correlated_with", []),
            "uses_operators": alpha.get("operators", []),
            "implements_concepts": alpha.get("concepts", []),
            "uses_datafields": alpha.get("datafields", [])[:10],
            "failed_by": alpha.get("failure_modes", []),
            "tested_under": alpha.get("settings", []),
        }
        
        return lineage

    def _tool_stats(self) -> Dict:
        """Handle get_memory_stats tool."""
        stats = get_stats()
        
        from .structure import load_metadata
        metadata = load_metadata()
        
        by_type = {}
        for m in metadata:
            t = m.get("node_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "qdrant": stats,
            "node_counts": by_type,
            "total_nodes": len(metadata),
            "embedding_model": config.embedding_model_name,
            "collection": config.collection_name
        }

    def _tool_research_state(self, params: Dict) -> Dict:
        """Handle compile_research_state tool - Research State Compiler."""
        from memory_layer.retrieve import retrieve_hybrid
        from memory_layer.rerank import compile_research_state, rerank_results

        query = params.get("query", "")
        k = params.get("k", 5)
        budget = params.get("budget_tokens", 2000)
        rerank = params.get("rerank", True)

        # Retrieve results
        results = retrieve_hybrid(query=query, k=k * 2)  # Get more for reranking

        # Apply reranking if requested
        if rerank:
            results = rerank_results(results)

        # Compile research state
        state = compile_research_state(query=query, results=results, token_budget=budget)

        return state


def handle_jsonrpc(request: Union[Dict, str]) -> Dict:
    """Main entry point for MCP requests."""
    if isinstance(request, str):
        request = json.loads(request)
    
    server = MCPServer()
    response = server.handle_request(request)
    
    result = {"jsonrpc": response.jsonrpc, "id": response.id}
    if response.error:
        result["error"] = response.error
    else:
        result["result"] = response.result
    
    return result

    def _tool_classify_factors(self, params: Dict) -> Dict:
        """Handle classify_alpha_factors tool."""
        from .factor_ontology import classify_alpha
        result = classify_alpha(
            params.get("expression", ""),
            params.get("datafields", []),
            params.get("operators", []),
            params.get("concepts", [])
        )
        return {"alpha_id": params.get("alpha_id"), "factor_exposures": result}

    def _tool_regime_performance(self, params: Dict) -> Dict:
        """Handle get_regime_performance tool."""
        from .regime_analysis import get_regime_performance
        return {"alpha_id": params.get("alpha_id"), "regime_performance": get_regime_performance(params.get("alpha_id"))}

    def _tool_find_regime_alphas(self, params: Dict) -> Dict:
        """Handle find_regime_alphas tool."""
        from .regime_analysis import find_regime_alphas
        return {"regime": params.get("regime"), "alphas": find_regime_alphas(params.get("regime"), params.get("min_sharpe", 0.5))}

    def _tool_explain_failure(self, params: Dict) -> Dict:
        """Handle explain_alpha_failure tool."""
        from .research_copilot import explain_alpha_failure
        return explain_alpha_failure(
            params.get("alpha_id", ""),
            params.get("failure_modes", []),
            params.get("sharpe"),
            params.get("turnover"),
            params.get("fitness"),
            params.get("expression", ""),
            params.get("operators", []),
            params.get("datafields", [])
        )

    def _tool_orthogonal_alphas(self, params: Dict) -> Dict:
        """Handle find_orthogonal_alphas tool."""
        from .correlation_engine import find_orthogonal_sleeves
        return {"target": params.get("alpha_id"), "orthogonal": find_orthogonal_sleeves(params.get("alpha_id"))}

    def _tool_hidden_exposures(self, params: Dict) -> Dict:
        """Handle detect_hidden_exposures tool."""
        from .research_copilot import detect_hidden_exposures
        return detect_hidden_exposures(
            params.get("expression", ""),
            params.get("datafields", []),
            params.get("operators", []),
            params.get("concepts", []),
            params.get("neutralization", "market")
        )

    def _tool_nl_query(self, params: Dict) -> Dict:
        """Handle natural_language_query tool."""
        from .nl_query import process_query
        return process_query(params.get("query", ""))

    def _tool_exploration_coverage(self, params: Dict) -> Dict:
        """Handle get_exploration_coverage tool."""
        from .recommendation_engine import get_exploration_coverage
        return get_exploration_coverage()

    def _tool_factor_suggestions(self, params: Dict) -> Dict:
        """Handle suggest_factor_combinations tool."""
        from .recommendation_engine import suggest_factors
        return {"suggestions": suggest_factors(params.get("used_factors", []), params.get("used_datafields", []))}

    def _tool_register_lineage(self, params: Dict) -> Dict:
        """Handle register_alpha_lineage tool."""
        from .alpha_lineage import register_alpha
        return register_alpha(
            params.get("alpha_id", ""),
            params.get("parent_id"),
            branch_id=params.get("branch_id")
        )

    def _tool_get_lineage(self, params: Dict) -> Dict:
        """Handle get_alpha_lineage tool."""
        from .alpha_lineage import get_lineage
        result = get_lineage(params.get("alpha_id", ""))
        if not result:
            return {"error": "Alpha not found in lineage"}
        return result

    def _tool_compare_lineage(self, params: Dict) -> Dict:
        """Handle compare_alpha_lineage tool."""
        from .alpha_lineage import compare_lineage
        return compare_lineage(params.get("alpha_1", ""), params.get("alpha_2", ""))

    def _tool_experiment_tree(self, params: Dict) -> Dict:
        """Handle get_experiment_tree tool."""
        from .alpha_lineage import get_tree
        return get_tree(params.get("root_alpha_id", ""), params.get("max_depth", 5))

    def _tool_lineage_stats(self, params: Dict) -> Dict:
        """Handle get_lineage_stats tool."""
        from .alpha_lineage import get_lineage_stats
        return get_lineage_stats()

    def _tool_generate_hypothesis(self, params: Dict) -> Dict:
        """Handle generate_hypothesis tool."""
        from .research_agent import generate_research_hypothesis
        return generate_research_hypothesis(
            params.get("focus_area"),
            params.get("based_on_alpha")
        )

    def _tool_analyze_failed_alpha(self, params: Dict) -> Dict:
        """Handle analyze_failed_alpha_research tool."""
        from .research_agent import analyze_failed_alpha_research
        return analyze_failed_alpha_research(params.get("alpha_id", ""))

    def _tool_find_orthogonal_sleeve(self, params: Dict) -> Dict:
        """Handle find_orthogonal_sleeve tool."""
        from .research_agent import find_orthogonal_sleeve_research
        return {
            "alpha_id": params.get("alpha_id"),
            "sleeves": find_orthogonal_sleeve_research(
                params.get("alpha_id", ""),
                params.get("max_candidates", 5)
            )
        }

    def _tool_detect_gaps(self, params: Dict) -> Dict:
        """Handle detect_research_gaps tool."""
        from .research_agent import detect_research_gaps
        return detect_research_gaps()

    def _tool_maintain_diversity(self, params: Dict) -> Dict:
        """Handle maintain_research_diversity tool."""
        from .research_agent import maintain_research_diversity
        return maintain_research_diversity(params.get("current_alphas", []))

    def _tool_agent_stats(self, params: Dict) -> Dict:
        """Handle get_agent_stats tool."""
        from .research_agent import get_agent_stats
        return get_agent_stats()


# Standalone MCP server with HTTP
def run_mcp_server(host: str = "0.0.0.0", port: int = 8000):
    """Run MCP server as HTTP endpoint."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="WQ Brain MCP Server")

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        body = await request.json()
        response = handle_jsonrpc(body)
        return JSONResponse(response)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "wq-brain-mcp"}

    @app.get("/")
    async def root():
        return {
            "service": "WQ Brain Memory MCP Server",
            "version": "0.1.0",
            "endpoints": {
                "mcp": "/mcp (POST, JSON-RPC 2.0)",
                "tools": "Use tools/list method to see available tools"
            }
        }

    print(f"Starting MCP server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_mcp_server()
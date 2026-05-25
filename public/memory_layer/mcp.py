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
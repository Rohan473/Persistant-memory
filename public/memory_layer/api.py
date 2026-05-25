"""
FastAPI server for memory layer.
Provides endpoints for agent integration.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from memory_layer import (
    config,
    ingest_all,
    retrieve_hybrid,
    format_context,
    compile_context,
    get_stats
)

app = FastAPI(title="WQ Brain Memory API", version="0.1.0")


# Request/Response models

class IngestRequest(BaseModel):
    force: bool = False
    node_type: Optional[str] = None


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    total: int


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    operators: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    node_types: Optional[List[str]] = None


class ContextRequest(BaseModel):
    query: str
    k: int = 5
    operators: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    budget: int = 2000
    format: str = "full"  # "full" or "compact"


class SearchResult(BaseModel):
    id: str
    node_type: str
    name: str
    score: float
    retrieval_score: float
    structured_summary: str
    operators: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    failure_modes: Optional[List[str]] = None
    salience: Optional[Dict] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str


# Endpoints

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest graph nodes into memory."""
    if req.node_type:
        from memory_layer.ingest import ingest_node_type
        result = ingest_node_type(req.node_type, force=req.force)
    else:
        result = ingest_all(force=req.force)
    return result


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Search memories with hybrid retrieval."""
    results = retrieve_hybrid(
        query=req.query,
        k=req.k,
        operators=req.operators,
        concepts=req.concepts,
        node_types=req.node_types,
    )

    search_results = []
    for r in results:
        sr = SearchResult(
            id=r.get("id", ""),
            node_type=r.get("node_type", ""),
            name=r.get("name", ""),
            score=r.get("score", 0),
            retrieval_score=r.get("retrieval_score", 0),
            structured_summary=r.get("structured_summary", ""),
            operators=r.get("operators"),
            concepts=r.get("concepts"),
            failure_modes=r.get("failure_modes"),
            salience=r.get("salience"),
        )
        search_results.append(sr)

    return SearchResponse(
        results=search_results,
        total=len(search_results),
        query=req.query
    )


@app.post("/context")
async def get_context(req: ContextRequest):
    """Get LLM-ready context from memories."""
    results = retrieve_hybrid(
        query=req.query,
        k=req.k,
        operators=req.operators,
        concepts=req.concepts,
    )

    if req.format == "compact":
        from memory_layer.budget import compile_context_compact
        context, tokens = compile_context_compact(results, budget_tokens=req.budget)
        return {
            "context": context,
            "tokens": tokens,
            "memories_used": len(results)
        }
    else:
        context = format_context(results, max_memories=req.k)
        from memory_layer.budget import compile_context, estimate_tokens_precise
        compiled, tokens, used = compile_context(results, budget_tokens=req.budget, max_memories=req.k)

        return {
            "context": context,
            "compiled_context": compiled,
            "tokens": tokens,
            "memories_used": len(used),
            "results": results[:req.k]
        }


@app.get("/stats")
async def stats():
    """Get memory statistics."""
    return get_stats()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


# MCP JSON-RPC 2.0 Endpoint
@app.post("/mcp")
async def mcp_endpoint(request: Dict):
    """
    MCP (Model Context Protocol) JSON-RPC 2.0 endpoint.
    Compatible with Claude, GPT, and other AI agents.
    """
    from .mcp import handle_jsonrpc
    return handle_jsonrpc(request)


@app.get("/mcp")
async def mcp_info():
    """MCP server info and available tools."""
    from .mcp import MCPToolRegistry
    return {
        "service": "WQ Brain Memory MCP Server",
        "version": "0.1.0",
        "protocol": "JSON-RPC 2.0",
        "tools": MCPToolRegistry.list_tools()
    }


@app.get("/trace/stats")
async def trace_stats():
    """Get retrieval logging statistics."""
    from .trace import logger
    return logger.get_stats()


@app.get("/trace/recent")
async def trace_recent(n: int = 20):
    """Get recent retrieval events."""
    from .trace import logger
    events = logger.get_recent(n)
    return {
        "events": [
            {
                "timestamp": e.timestamp,
                "query": e.query[:50] + "..." if len(e.query) > 50 else e.query,
                "tool": e.tool_used,
                "retrieved": e.retrieved_names[:3],
                "latency_ms": e.latency_ms,
            }
            for e in events
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
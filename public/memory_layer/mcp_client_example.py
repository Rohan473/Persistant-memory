"""
Example MCP Client - How external agents interact with WQ Brain Memory
"""

import requests
import json

MCP_BASE_URL = "http://localhost:8000"


def call_mcp(method: str, params: dict = None, req_id: int = 1) -> dict:
    """Make a JSON-RPC 2.0 call to MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id
    }
    if params:
        payload["params"] = params
    
    response = requests.post(f"{MCP_BASE_URL}/mcp", json=payload)
    return response.json()


# ============================================================
# EXAMPLE 1: List available tools
# ============================================================
def example_list_tools():
    """Show all available MCP tools."""
    print("=" * 60)
    print("EXAMPLE 1: List Available Tools")
    print("=" * 60)
    
    response = call_mcp("tools/list")
    print(json.dumps(response, indent=2))


# ============================================================
# EXAMPLE 2: Semantic search (like vector retrieval)
# ============================================================
def example_semantic_search():
    """Search using natural language + vector similarity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Semantic Search")
    print("=" * 60)
    
    response = call_mcp("tools/call", {
        "name": "search_research_memory",
        "query": "fundamental value alphas with low sharpe",
        "k": 5
    })
    print(json.dumps(response, indent=2))


# ============================================================
# EXAMPLE 3: Get LLM-ready context (with budget)
# ============================================================
def example_get_context():
    """Get token-budgeted context for LLM injection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Get LLM Context with Budget")
    print("=" * 60)
    
    response = call_mcp("tools/call", {
        "name": "get_research_context",
        "query": "mean reversion alphas",
        "k": 3,
        "budget_tokens": 1500,
        "format": "compact"
    })
    print(json.dumps(response, indent=2))


# ============================================================
# EXAMPLE 4: Pure symbolic retrieval (no vector)
# ============================================================
def example_symbolic_search():
    """Filter by operators, concepts without vector search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Symbolic Retrieval (No Vectors)")
    print("=" * 60)
    
    response = call_mcp("tools/call", {
        "name": "search_by_symbolic",
        "operators": ["ts_rank", "ts_decay_linear"],
        "concepts": ["mean_reversion"],
        "rating": "Needs Improvement"
    })
    print(json.dumps(response, indent=2))


# ============================================================
# EXAMPLE 5: Get alpha lineage (graph traversal)
# ============================================================
def example_alpha_lineage():
    """Get derivation chain for an alpha."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Alpha Lineage (Graph Traversal)")
    print("=" * 60)
    
    response = call_mcp("tools/call", {
        "name": "get_alpha_lineage",
        "alpha_name": "alpha_0020"
    })
    print(json.dumps(response, indent=2))


# ============================================================
# EXAMPLE 6: Get system stats
# ============================================================
def example_stats():
    """Get memory system statistics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: System Statistics")
    print("=" * 60)
    
    response = call_mcp("tools/call", {
        "name": "get_memory_stats"
    })
    print(json.dumps(response, indent=2))


# ============================================================
# EXAMPLE 7: Integration with Claude/Coding Agent
# ============================================================
def example_agent_integration():
    """
    This is how an AI agent (like Claude, GPT) would use the system.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Agent Integration Pattern")
    print("=" * 60)
    
    # Step 1: Agent has a research question
    research_question = "What fundamental value alphas derived from alpha_0017 failed due to low sharpe?"
    
    # Step 2: Get structured context
    context_response = call_mcp("tools/call", {
        "name": "get_research_context",
        "query": research_question,
        "k": 3,
        "budget_tokens": 1000,
        "format": "compact"
    })
    
    # Step 3: Agent uses context in its reasoning
    context = context_response.get("result", {}).get("context", "")
    
    print(f"Research Question: {research_question}")
    print(f"\nRetrieved Context ({context_response.get('result', {}).get('tokens', 0)} tokens):")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    print("WQ Brain Memory MCP Client Examples")
    print("=" * 60)
    print("Make sure the server is running: python -m memory_layer.api")
    print("=" * 60)
    
    try:
        example_list_tools()
        example_semantic_search()
        example_get_context()
        example_symbolic_search()
        example_alpha_lineage()
        example_stats()
        example_agent_integration()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Server not running. Start with: python -m memory_layer.api")
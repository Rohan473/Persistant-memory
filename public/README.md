# WQ Brain Memory Layer

A persistent symbolic research-memory system with hybrid retrieval and MCP interoperability for quantitative research workflows.

---

## What This Is

A knowledge graph infrastructure for tracking and retrieving quantitative research experiments. Built to work with WorldQuant Brain alpha development sessions but designed as a generic framework.

### Core Components

| Component | Description |
|-----------|-------------|
| `memory_layer/` | Core retrieval, MCP server, Qdrant integration, symbolic indexing |
| `memory_layer/api.py` | FastAPI REST interface |
| `memory_layer/mcp.py` | Model Context Protocol server |
| `memory_layer/retrieve.py` | Hybrid retrieval (semantic + symbolic) |
| `memory_layer/salience.py` | Memory importance scoring |
| `memory_layer/budget.py` | Token budget management for LLM context |
| `graph/` | Interactive D3.js graph visualization |

---

## Quick Start

```bash
# Install dependencies
pip install -r memory_layer/requirements.txt

# Start Qdrant (required for semantic search)
docker run -p 6333:6333 qdrant/qdrant

# Start the API
python -m memory_layer.api

# Or use the MCP server
python -m memory_layer.mcp
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  MCP Server  │────▶│  Retrieval  │
└─────────────┘     └──────────────┘     └─────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
             ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
             │  Qdrant     │         │   Graph     │         │  Salience   │
             │ (semantic)  │         │ (symbolic)  │         │  Scoring    │
             └─────────────┘         └─────────────┘         └─────────────┘
```

---

## Key Features

- **Hybrid Retrieval**: Combines semantic (vector) search with symbolic (graph) traversal
- **MCP Integration**: Works with Claude, ChatGPT, and other LLM assistants
- **Token Budgeting**: Compiles memories within LLM context limits
- **Salience Scoring**: Prioritizes high-importance memories
- **Version Tracking**: Maintains history of memory changes

---

## Example Usage

```python
from memory_layer import retrieve

# Retrieve relevant memories
results = retrieve.hybrid_retrieve(
    query="momentum strategy",
    k=5,
    token_budget=2000
)

# Get LLM-ready context
context = retrieve.compile_context(results, budget_tokens=1500)
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /retrieve` | Hybrid retrieval |
| `POST /retrieve/compact` | Compact context retrieval |
| `POST /research/state` | Full research state compilation |
| `GET /health` | Health check |

---

## Graph Visualization

Open `graph/graph.html` in a browser to see an interactive demo of the knowledge graph structure.

> **Note**: The demo includes synthetic sample data. The system can handle real research graphs when properly configured.

---

## Requirements

- Python 3.10+
- Qdrant (for vector storage)
- sentence-transformers (for embeddings)
- FastAPI + Uvicorn (for API)

Full list in `memory_layer/requirements.txt`

---

## License

MIT License - feel free to use this framework for your own projects.
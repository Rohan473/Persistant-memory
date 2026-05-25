"""
Memory Layer Configuration
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Paths
    graph_path: Path = Path(__file__).parent.parent / "graph" / "graph.json"
    metadata_output: Path = Path(__file__).parent / "graph_metadata.json"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "wq_memory"

    # Embedding
    embedding_model: str = "sentence-transformers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Context budget (for LLM tokens)
    context_budget_tokens: int = 2000

    # Salience
    default_importance: float = 0.5

config = Config()

def get_qdrant_url() -> str:
    return f"http://{config.qdrant_host}:{config.qdrant_port}"
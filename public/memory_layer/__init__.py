"""
WQ Brain Memory Layer

Multi-resolution hybrid memory for quant research knowledge graph.
"""

__version__ = "0.1.0"

from .config import config, Config
from .embed import embed_texts, get_embedding_model
from .structure import extract_all_metadata, generate_structured_summary, save_metadata, load_metadata
from .ingest import ingest_all, ingest_node_type, get_stats
from .retrieve import retrieve, retrieve_hybrid, format_context
from .budget import compile_context, estimate_tokens
from .rerank import rerank_results, compile_research_state
from . import version, trace

__all__ = [
    "config",
    "Config",
    "embed_texts",
    "get_embedding_model",
    "extract_all_metadata",
    "generate_structured_summary",
    "save_metadata",
    "load_metadata",
    "ingest_all",
    "ingest_node_type",
    "get_stats",
    "retrieve",
    "retrieve_hybrid",
    "format_context",
    "compile_context",
    "estimate_tokens",
    "rerank_results",
    "compile_research_state",
    "version",
    "trace",
]
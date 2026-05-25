"""
WQ Brain Memory Layer

Multi-resolution hybrid memory for quant research knowledge graph.
Institutional-grade quant research intelligence platform.
"""

__version__ = "0.2.0"

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
    "factor_ontology",
    "regime_analysis",
    "failure_learning",
    "vector_memory",
    "correlation_engine",
    "research_copilot",
    "nl_query",
    "recommendation_engine",
]

import memory_layer.factor_ontology as factor_ontology
import memory_layer.regime_analysis as regime_analysis
import memory_layer.failure_learning as failure_learning
import memory_layer.vector_memory as vector_memory
import memory_layer.correlation_engine as correlation_engine
import memory_layer.research_copilot as research_copilot
import memory_layer.nl_query as nl_query
import memory_layer.recommendation_engine as recommendation_engine
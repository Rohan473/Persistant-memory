"""
Embedding module using sentence-transformers.
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .config import config

_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.embedding_model_name)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using sentence-transformers.

    Args:
        texts: List of strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

def embed_text(text: str) -> np.ndarray:
    """Embed a single text."""
    return embed_texts([text])[0]

def get_embedding_dimension() -> int:
    """Return the embedding dimension for the configured model."""
    return config.embedding_dim
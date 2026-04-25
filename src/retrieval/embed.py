"""
Embed text passages and queries using sentence-transformers.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
        _model = SentenceTransformer(model_name)
    return _model


def embed_passages(passages: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of document passages. Adds 'passage: ' prefix for e5 models."""
    model = get_model()
    prefixed = [f"passage: {p}" for p in passages]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Adds 'query: ' prefix for e5 models."""
    model = get_model()
    return model.encode(
        f"query: {query}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

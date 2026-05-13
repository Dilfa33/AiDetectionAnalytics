"""
src/embeddings/embedder.py
Lab 11: Load sentence-transformer model once and generate text embeddings.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils.logger import logging

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (downloaded once from HuggingFace)."""
    global _model
    if _model is None:
        logging.info("[Embedder] Loading model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logging.info("[Embedder] Model loaded — embedding dim: %d", _model.get_sentence_embedding_dimension())
    return _model


def make_paper_text(row: pd.Series) -> str:
    """Combine title, abstract, and primary_category into one string for embedding."""
    title    = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    category = str(row.get("primary_category", "")).strip()
    return f"{title}. {category}. {abstract}"


def generate_embeddings(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Encode a list of strings into a 2-D NumPy array of shape (N, 384)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logging.info("[Embedder] Generated embeddings: shape=%s", embeddings.shape)
    return embeddings


def embed_dataframe(df: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
    """Generate one embedding per row by combining title + category + abstract."""
    texts = df.apply(make_paper_text, axis=1).tolist()
    return generate_embeddings(texts, show_progress=show_progress)

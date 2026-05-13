"""
src/embeddings/chroma_store.py
Lab 11: Manage a persistent ChromaDB collection for ArXiv paper embeddings.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import chromadb
from chromadb.config import Settings
from pathlib import Path
from src.utils.logger import logging
from src.embeddings.embedder import make_paper_text, generate_embeddings

CHROMA_PATH = Path("data/embeddings/chroma_db")
COLLECTION_NAME = "papers"


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at data/embeddings/chroma_db/."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_or_create_collection(client: chromadb.PersistentClient, reset: bool = False):
    """Get (or create) the papers collection using cosine similarity."""
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logging.info("[ChromaStore] Collection '%s' deleted for reset.", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logging.info("[ChromaStore] Collection '%s' ready — %d documents.", COLLECTION_NAME, collection.count())
    return collection


def add_papers(collection, df: pd.DataFrame, batch_size: int = 100) -> int:
    """
    Embed and upsert papers from df into the ChromaDB collection.
    Skips papers already present. Returns number of new papers added.
    """
    existing_ids = set(collection.get(include=[])["ids"])
    new_df = df[~df["paper_id"].astype(str).isin(existing_ids)].copy()

    if new_df.empty:
        logging.info("[ChromaStore] All %d papers already in collection.", len(df))
        return 0

    logging.info("[ChromaStore] Embedding %d new papers ...", len(new_df))
    texts = new_df.apply(make_paper_text, axis=1).tolist()
    embeddings = generate_embeddings(texts, show_progress=True)

    ids       = new_df["paper_id"].astype(str).tolist()
    documents = texts
    metadatas = []
    for _, row in new_df.iterrows():
        metadatas.append({
            "title":            str(row.get("title", "")),
            "primary_category": str(row.get("primary_category", "")),
            "published_year":   int(row.get("published_year", 0)),
            "language":         str(row.get("language", "en")),
            "citation_count":   int(row.get("citation_count", 0)),
            "relevance_score":  float(row.get("relevance_score", 0.0)),
        })

    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            metadatas=metadatas[i:i+batch_size],
        )

    logging.info("[ChromaStore] Upserted %d papers.", len(ids))
    return len(ids)


def query_collection(collection, query_texts: list[str], n_results: int = 5,
                     where: dict | None = None) -> list[dict]:
    """
    Query the collection with one or more text strings.
    Returns a list of result dicts (one per query), each with keys:
    ids, documents, metadatas, distances.
    """
    from src.embeddings.embedder import generate_embeddings
    query_embeddings = generate_embeddings(query_texts, show_progress=False).tolist()

    kwargs = dict(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results = []
    for i, qtext in enumerate(query_texts):
        results.append({
            "query":     qtext,
            "ids":       raw["ids"][i],
            "documents": raw["documents"][i],
            "metadatas": raw["metadatas"][i],
            "distances": raw["distances"][i],
        })
    return results

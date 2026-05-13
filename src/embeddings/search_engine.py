"""
src/embeddings/search_engine.py
Lab 11: High-level semantic, keyword, and comparison search functions.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.utils.logger import logging
from src.embeddings.chroma_store import query_collection


def semantic_search(collection, query: str, n_results: int = 5,
                    where: dict | None = None) -> pd.DataFrame:
    """Search papers by meaning using ChromaDB cosine similarity."""
    results = query_collection(collection, [query], n_results=n_results, where=where)
    hits = results[0]

    rows = []
    for id_, doc, meta, dist in zip(
        hits["ids"], hits["documents"], hits["metadatas"], hits["distances"]
    ):
        rows.append({
            "paper_id":         id_,
            "title":            meta.get("title", ""),
            "primary_category": meta.get("primary_category", ""),
            "published_year":   meta.get("published_year", ""),
            "citation_count":   meta.get("citation_count", ""),
            "similarity":       round(1 - dist, 4),  # cosine distance → similarity
        })

    df = pd.DataFrame(rows).sort_values("similarity", ascending=False)
    logging.info("[SearchEngine] Semantic search '%s' → %d results", query, len(df))
    return df


def keyword_search(df: pd.DataFrame, query: str, n_results: int = 5,
                   columns: list[str] | None = None) -> pd.DataFrame:
    """Search papers by exact keyword match across title and abstract columns."""
    if columns is None:
        columns = ["title", "abstract"]

    query_lower = query.lower()
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False)

    hits = df[mask].copy()
    hits["keyword_match"] = True
    logging.info("[SearchEngine] Keyword search '%s' → %d results", query, len(hits))
    return hits.head(n_results)[["paper_id", "title", "primary_category", "published_year", "citation_count"]]


def compare_search(collection, df: pd.DataFrame, query: str,
                   n_results: int = 5) -> None:
    """Run both search methods on the same query and print results side by side."""
    print(f"\n{'='*70}")
    print(f"Query: \"{query}\"")
    print(f"{'='*70}")

    print(f"\n--- Keyword Search ({n_results} results) ---")
    kw = keyword_search(df, query, n_results=n_results)
    if kw.empty:
        print("  (no exact keyword matches)")
    else:
        print(kw.to_string(index=False))

    print(f"\n--- Semantic Search ({n_results} results) ---")
    sem = semantic_search(collection, query, n_results=n_results)
    print(sem.to_string(index=False))
    print()

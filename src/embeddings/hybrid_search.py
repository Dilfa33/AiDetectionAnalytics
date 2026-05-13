"""
src/embeddings/hybrid_search.py
Lab 11: Reciprocal Rank Fusion (RRF) to combine keyword and semantic search results.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.utils.logger import logging
from src.embeddings.search_engine import keyword_search, semantic_search

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Combine multiple ranked lists of paper IDs using RRF.
    score = 1 / (k + rank)   (rank is 1-based)
    Returns a dict mapping paper_id → combined RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, paper_id in enumerate(ranked, start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(collection, df: pd.DataFrame, query: str,
                  n_results: int = 5, k: int = RRF_K) -> pd.DataFrame:
    """
    Perform hybrid search by fusing keyword and semantic ranked lists with RRF.
    Returns a DataFrame sorted by combined RRF score descending.
    """
    fetch_n = max(n_results * 3, 20)

    kw_df  = keyword_search(df, query, n_results=fetch_n)
    sem_df = semantic_search(collection, query, n_results=fetch_n)

    kw_ids  = kw_df["paper_id"].astype(str).tolist()
    sem_ids = sem_df["paper_id"].astype(str).tolist()

    rrf_scores = reciprocal_rank_fusion([kw_ids, sem_ids], k=k)

    rows = []
    all_ids = list(rrf_scores.keys())
    for pid in all_ids:
        meta_row = df[df["paper_id"].astype(str) == pid]
        if not meta_row.empty:
            r = meta_row.iloc[0]
            rows.append({
                "paper_id":         pid,
                "title":            r.get("title", ""),
                "primary_category": r.get("primary_category", ""),
                "published_year":   r.get("published_year", ""),
                "citation_count":   r.get("citation_count", ""),
                "rrf_score":        round(rrf_scores[pid], 6),
            })
        else:
            # Paper came from semantic search — get title from ChromaDB results
            sem_hit = sem_df[sem_df["paper_id"].astype(str) == pid]
            if not sem_hit.empty:
                r = sem_hit.iloc[0]
                rows.append({
                    "paper_id":         pid,
                    "title":            r.get("title", ""),
                    "primary_category": r.get("primary_category", ""),
                    "published_year":   r.get("published_year", ""),
                    "citation_count":   r.get("citation_count", ""),
                    "rrf_score":        round(rrf_scores[pid], 6),
                })

    result = pd.DataFrame(rows).sort_values("rrf_score", ascending=False).head(n_results)
    logging.info("[HybridSearch] '%s' → %d hybrid results (RRF k=%d)", query, len(result), k)
    return result

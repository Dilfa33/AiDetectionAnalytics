"""
src/analytics/mongo_pipeline.py
Lab 10: MongoDB aggregation pipeline with $match, $group, $sort, $project.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.utils.logger import logging

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "arxiv_pipeline"
COLLECTION = "papers"


def get_mongo_collection(uri: str = MONGO_URI, db: str = DB_NAME, col: str = COLLECTION):
    """Return a pymongo Collection object."""
    from pymongo import MongoClient
    client = MongoClient(uri)
    return client[db][col], client


def build_aggregation_pipeline(min_relevance: float = 0.5) -> list[dict]:
    """
    Four-stage aggregation pipeline:
      $match  → relevance_score >= min_relevance
      $group  → avg citations and count per primary_category
      $sort   → descending by avg_citations
      $project → clean output field names
    """
    return [
        {"$match": {"relevance_score": {"$gte": min_relevance}}},
        {
            "$group": {
                "_id":           "$primary_category",
                "avg_citations": {"$avg": "$citation_count"},
                "total_papers":  {"$sum": 1},
                "max_citations": {"$max": "$citation_count"},
            }
        },
        {"$sort": {"avg_citations": -1}},
        {
            "$project": {
                "_id":             0,
                "category":        "$_id",
                "avg_citations":   {"$round": ["$avg_citations", 2]},
                "total_papers":    1,
                "max_citations":   1,
            }
        },
    ]


def run_mongo_pipeline(min_relevance: float = 0.5,
                       df_fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Run the aggregation pipeline against MongoDB.
    Falls back to computing the same aggregation from df_fallback if MongoDB
    is unavailable, so the notebook always produces a result.
    """
    try:
        collection, client = get_mongo_collection()
        pipeline = build_aggregation_pipeline(min_relevance)
        results  = list(collection.aggregate(pipeline))
        client.close()

        if not results:
            logging.warning("[MongoPipeline] No results — collection may be empty")
            raise ValueError("empty collection")

        df = pd.DataFrame(results)
        logging.info("[MongoPipeline] Pipeline returned %d categories", len(df))
        return df

    except Exception as e:
        logging.warning("[MongoPipeline] MongoDB unavailable (%s) — using CSV fallback", e)

        if df_fallback is None:
            return pd.DataFrame()

        # Mirror the pipeline logic in pandas
        filtered = df_fallback[df_fallback["relevance_score"] >= min_relevance]
        result = (
            filtered.groupby("primary_category")
            .agg(
                avg_citations=("citation_count", "mean"),
                total_papers =("citation_count", "count"),
                max_citations=("citation_count", "max"),
            )
            .reset_index()
            .rename(columns={"primary_category": "category"})
            .sort_values("avg_citations", ascending=False)
            .round({"avg_citations": 2})
            .reset_index(drop=True)
        )
        logging.info("[MongoPipeline] Fallback pipeline: %d categories", len(result))
        return result


def seed_mongo_from_df(df: pd.DataFrame,
                        uri: str = MONGO_URI,
                        db: str = DB_NAME,
                        col: str = COLLECTION) -> int:
    """Insert cleaned DataFrame records into MongoDB (idempotent via paper_id)."""
    try:
        from pymongo import MongoClient, UpdateOne
        client     = MongoClient(uri)
        collection = client[db][col]
        ops = [
            UpdateOne(
                {"paper_id": str(row["paper_id"])},
                {"$set": row.to_dict()},
                upsert=True,
            )
            for _, row in df.iterrows()
        ]
        result = collection.bulk_write(ops)
        client.close()
        logging.info("[MongoPipeline] Seeded %d docs into '%s'",
                     result.upserted_count + result.modified_count, col)
        return result.upserted_count + result.modified_count
    except Exception as e:
        logging.warning("[MongoPipeline] Seed failed: %s", e)
        return 0

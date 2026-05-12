"""
src/analytics/db_connector.py
Lab 10: Connect to MySQL, populate and query paper_metrics table.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from pathlib import Path
from src.utils.logger import logging

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "papers_db",
}

CLEANED_CSV = Path("data/processed/cleaned/cleaned_data.csv")


def get_connection():
    """Open and return a PyMySQL connection."""
    try:
        import pymysql
        conn = pymysql.connect(**DB_CONFIG)
        logging.info("[DBConnector] MySQL connection established")
        return conn
    except ImportError:
        logging.error("[DBConnector] pymysql not installed — run: pip install pymysql")
        raise
    except Exception as e:
        logging.error("[DBConnector] Connection failed: %s", e)
        raise


def create_table(conn) -> None:
    """Create paper_metrics table if it does not exist."""
    ddl = """
        CREATE TABLE IF NOT EXISTS paper_metrics (
            paper_id         VARCHAR(50) PRIMARY KEY,
            title            TEXT,
            citation_count   INT,
            relevance_score  FLOAT,
            published_year   INT,
            primary_category VARCHAR(50)
        )
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    logging.info("[DBConnector] Table paper_metrics ready")


def populate_metrics(conn, df: pd.DataFrame) -> int:
    """Insert rows from df into paper_metrics, skipping invalid rows."""
    required = {"paper_id", "title", "citation_count", "relevance_score",
                "published_year", "primary_category"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logging.error("[DBConnector] Missing columns: %s", missing)
        return 0

    sql = """
        INSERT IGNORE INTO paper_metrics
            (paper_id, title, citation_count, relevance_score, published_year, primary_category)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    inserted = 0
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            try:
                pid = str(row["paper_id"])
                if not pid or pid == "nan":
                    continue
                cur.execute(sql, (
                    pid,
                    str(row["title"])[:500],
                    int(row["citation_count"]),
                    float(row["relevance_score"]),
                    int(row["published_year"]),
                    str(row["primary_category"]),
                ))
                inserted += 1
            except Exception as e:
                logging.warning("[DBConnector] Skipped row %s: %s", row.get("paper_id"), e)
    conn.commit()
    logging.info("[DBConnector] Inserted %d rows into paper_metrics", inserted)
    return inserted


def query_metrics(conn) -> pd.DataFrame:
    """Query all rows from paper_metrics back into a DataFrame."""
    sql = "SELECT * FROM paper_metrics WHERE citation_count IS NOT NULL"
    df = pd.read_sql(sql, conn)
    logging.info("[DBConnector] Queried %d rows from paper_metrics", len(df))
    return df


def run_db_pipeline(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full round-trip: connect → create table → populate → query → return."""
    if df is None:
        df = pd.read_csv(CLEANED_CSV)
    conn = get_connection()
    try:
        create_table(conn)
        populate_metrics(conn, df)
        result = query_metrics(conn)
    finally:
        conn.close()
        logging.info("[DBConnector] Connection closed")
    return result

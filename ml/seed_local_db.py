"""
Seed the local SQLite database with business_scores and grid_aggregates
derived from the processed reviews Parquet.

This script is Person B's local dev shortcut: run it once after clean_job.py
has written data/processed/reviews_enriched/ so the Streamlit dashboard has
real data to display without needing a live RDS connection.

Usage:
    python ml/seed_local_db.py --input data/processed/

Output:
    data/citybite_local.db  (SQLite, tables: business_scores, grid_aggregates)
"""

import argparse
import math
import os
import sqlite3

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

load_dotenv()

# Path to the SQLite file relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOCAL_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------

def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("CityBite-SeedLocalDB")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")   # keep low for sample data
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Aggregation — mirrors aggregate_job.py SQL logic
# ---------------------------------------------------------------------------

def compute_business_scores(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Compute per-business popularity scores from enriched reviews.

    popularity_score formula (from CLAUDE.md):
        0.4 * avg_stars + 0.4 * log(review_count + 1) + 0.2 * avg_recency
    """
    path = input_path.rstrip("/") + "/reviews_enriched"
    df = spark.read.parquet(path)

    scores = (
        df.groupBy(
            "business_id", "name", "metro_area", "city",
            "latitude", "longitude", "grid_cell", "categories",
        )
        .agg(
            F.avg("stars").alias("avg_rating"),
            F.count("*").alias("review_count"),
            (F.sum("recency_weight") / F.count("*")).alias("recency_score"),
        )
        .withColumn(
            "popularity_score",
            F.col("avg_rating") * 0.4
            + F.log(F.col("review_count") + 1) * 0.4
            + F.col("recency_score") * 0.2,
        )
    )

    return scores


def compute_grid_aggregates(business_scores: DataFrame) -> DataFrame:
    """
    Compute per-grid-cell aggregate statistics from business_scores.

    top_cuisine is approximated with first() — on EMR we'd use a window
    function for the true modal cuisine, but first() is fine for local dev.
    """
    return (
        business_scores
        .groupBy("grid_cell", "metro_area")
        .agg(
            F.avg("latitude").alias("center_lat"),
            F.avg("longitude").alias("center_lng"),
            F.avg("popularity_score").alias("avg_popularity"),
            F.count("*").alias("restaurant_count"),
            F.first("categories").alias("top_cuisine"),
        )
    )


# ---------------------------------------------------------------------------
# SQLite I/O
# ---------------------------------------------------------------------------

def _ensure_sqlite_schema(db_path: str) -> None:
    """
    Create SQLite tables that mirror infra/schema.sql.

    SQLite doesn't support DEFAULT NOW() or FLOAT the same as Postgres,
    but the column names and semantics are identical so the dashboard
    queries work unchanged.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS business_scores (
        business_id      TEXT PRIMARY KEY,
        name             TEXT,
        city             TEXT,
        metro_area       TEXT,
        latitude         REAL,
        longitude        REAL,
        grid_cell        TEXT,
        categories       TEXT,
        avg_rating       REAL,
        review_count     INTEGER,
        popularity_score REAL,
        last_updated     TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS grid_aggregates (
        grid_cell         TEXT PRIMARY KEY,
        metro_area        TEXT,
        center_lat        REAL,
        center_lng        REAL,
        avg_popularity    REAL,
        restaurant_count  INTEGER,
        top_cuisine       TEXT
    );

    CREATE TABLE IF NOT EXISTS als_recommendations (
        user_id          TEXT,
        business_id      TEXT,
        predicted_rating REAL,
        rank             INTEGER,
        PRIMARY KEY (user_id, business_id)
    );

    CREATE TABLE IF NOT EXISTS grid_sentiment (
        grid_cell       TEXT PRIMARY KEY,
        sentiment_score REAL,
        positive_count  INTEGER,
        negative_count  INTEGER,
        last_updated    TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_business_city   ON business_scores(city);
    CREATE INDEX IF NOT EXISTS idx_business_metro  ON business_scores(metro_area);
    CREATE INDEX IF NOT EXISTS idx_grid_metro      ON grid_aggregates(metro_area);
    CREATE INDEX IF NOT EXISTS idx_als_user        ON als_recommendations(user_id);
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(ddl)
    conn.commit()
    conn.close()


def write_spark_df_to_sqlite(df: DataFrame, db_path: str, table: str) -> None:
    """
    Collect a Spark DataFrame to pandas and write it to SQLite via to_sql.

    Using replace mode so re-running the seed script is idempotent.
    """
    from sqlalchemy import create_engine

    pandas_df = df.toPandas()
    engine = create_engine(f"sqlite:///{db_path}")
    pandas_df.to_sql(
        name=table,
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"  Wrote {len(pandas_df):,} rows → {table}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed local SQLite DB for Streamlit dev"
    )
    parser.add_argument(
        "--input",
        default="data/processed/",
        help="Path to processed/ directory containing reviews_enriched/",
    )
    args = parser.parse_args()

    print(f"SQLite target: {LOCAL_DB_PATH}")
    _ensure_sqlite_schema(LOCAL_DB_PATH)

    spark = build_spark()

    # ── Business scores ────────────────────────────────────────────────────
    print("Computing business_scores ...")
    biz_scores = compute_business_scores(spark, args.input)
    write_spark_df_to_sqlite(biz_scores, LOCAL_DB_PATH, "business_scores")

    # ── Grid aggregates ────────────────────────────────────────────────────
    print("Computing grid_aggregates ...")
    grid_agg = compute_grid_aggregates(biz_scores)
    write_spark_df_to_sqlite(grid_agg, LOCAL_DB_PATH, "grid_aggregates")

    spark.stop()
    print(f"\nDone. Database seeded at: {LOCAL_DB_PATH}")
    print("Run `streamlit run dashboard/app.py` to view the map.")


if __name__ == "__main__":
    main()

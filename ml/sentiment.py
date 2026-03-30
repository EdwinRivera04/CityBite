"""
Compute grid-level sentiment scores and write to local SQLite or RDS.

This local-friendly implementation derives sentiment labels from star ratings:
- positive: stars >= 4
- negative: stars <= 2
- neutral:  stars == 3 (excluded from score denominator)

Usage (local):
    spark-submit ml/sentiment.py --input data/processed/ --mode local

Usage (EMR / S3):
    spark-submit ml/sentiment.py --input s3://citybite/processed/ --mode emr
"""

import argparse
import os
import sqlite3

from dotenv import load_dotenv
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sqlalchemy import create_engine

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_LOCAL_DB = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")


def build_spark(mode: str) -> SparkSession:
    builder = SparkSession.builder.appName("CityBite-Sentiment")
    if mode == "local":
        builder = builder.master("local[*]")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_reviews_spark(spark: SparkSession, input_path: str) -> DataFrame:
    path = input_path.rstrip("/") + "/reviews_enriched"
    return spark.read.parquet(path).select("grid_cell", "stars")


def load_reviews_pandas(input_path: str) -> pd.DataFrame:
    path = input_path.rstrip("/") + "/reviews_enriched"
    df = pd.read_parquet(path, columns=["grid_cell", "stars"])
    return df


def compute_grid_sentiment_spark(reviews: DataFrame) -> DataFrame:
    labeled = (
        reviews
        .withColumn(
            "label",
            F.when(F.col("stars") >= 4, F.lit("positive"))
             .when(F.col("stars") <= 2, F.lit("negative"))
             .otherwise(F.lit("neutral")),
        )
    )

    agg = (
        labeled.groupBy("grid_cell")
        .agg(
            F.sum(F.when(F.col("label") == "positive", F.lit(1)).otherwise(F.lit(0))).alias("positive_count"),
            F.sum(F.when(F.col("label") == "negative", F.lit(1)).otherwise(F.lit(0))).alias("negative_count"),
        )
        .withColumn("denom", F.col("positive_count") + F.col("negative_count"))
        .withColumn(
            "sentiment_score",
            F.when(F.col("denom") > 0, F.col("positive_count") / F.col("denom")).otherwise(F.lit(None)),
        )
        .filter(F.col("denom") > 0)
        .drop("denom")
    )

    return agg.select("grid_cell", "sentiment_score", "positive_count", "negative_count")


def compute_grid_sentiment_pandas(reviews: pd.DataFrame) -> pd.DataFrame:
    data = reviews.copy()
    data["label"] = "neutral"
    data.loc[data["stars"] >= 4, "label"] = "positive"
    data.loc[data["stars"] <= 2, "label"] = "negative"

    grouped = (
        data.groupby("grid_cell")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "positive" not in grouped.columns:
        grouped["positive"] = 0
    if "negative" not in grouped.columns:
        grouped["negative"] = 0

    grouped = grouped.rename(
        columns={
            "positive": "positive_count",
            "negative": "negative_count",
        }
    )
    denom = grouped["positive_count"] + grouped["negative_count"]
    grouped = grouped[denom > 0].copy()
    grouped["sentiment_score"] = grouped["positive_count"] / (grouped["positive_count"] + grouped["negative_count"])

    return grouped[["grid_cell", "sentiment_score", "positive_count", "negative_count"]]


def get_db_url(mode: str) -> str:
    if mode == "local":
        return f"sqlite:///{_LOCAL_DB}"

    host = os.environ["RDS_HOST"]
    port = os.environ.get("RDS_PORT", "5432")
    db = os.environ["RDS_DB"]
    user = os.environ["RDS_USER"]
    pw = os.environ["RDS_PASSWORD"]
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


def ensure_local_table() -> None:
    conn = sqlite3.connect(_LOCAL_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS grid_sentiment (
            grid_cell       TEXT PRIMARY KEY,
            sentiment_score REAL,
            positive_count  INTEGER,
            negative_count  INTEGER,
            last_updated    TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()


def write_sentiment(sentiment_pd: pd.DataFrame, db_url: str, mode: str) -> None:
    if mode == "local":
        ensure_local_table()

    engine = create_engine(db_url)
    sentiment_pd.to_sql(
        name="grid_sentiment",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=2000,
    )
    print(f"  Wrote {len(sentiment_pd):,} rows to grid_sentiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="CityBite sentiment aggregation job")
    parser.add_argument("--input", required=True, help="Path to processed/ directory")
    parser.add_argument("--mode", choices=["local", "emr"], default="local")
    args = parser.parse_args()

    if args.mode == "local":
        print("Loading reviews (pandas local mode) ...")
        reviews_pd = load_reviews_pandas(args.input)

        print("Computing grid sentiment ...")
        sentiment_pd = compute_grid_sentiment_pandas(reviews_pd)
        print(f"  Grid cells with sentiment: {len(sentiment_pd):,}")
    else:
        spark = build_spark(args.mode)

        print("Loading reviews (Spark mode) ...")
        reviews = load_reviews_spark(spark, args.input)

        print("Computing grid sentiment ...")
        sentiment_df = compute_grid_sentiment_spark(reviews)
        sentiment_pd = sentiment_df.toPandas()
        print(f"  Grid cells with sentiment: {len(sentiment_pd):,}")

    db_url = get_db_url(args.mode)
    print(f"Writing to {db_url} ...")
    write_sentiment(sentiment_pd, db_url, args.mode)

    if args.mode != "local":
        spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()

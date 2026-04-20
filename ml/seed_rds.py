"""
Seed RDS PostgreSQL with business_scores and grid_aggregates from local processed Parquet.

Use this when you want to populate RDS without running the full EMR pipeline —
e.g. after running clean_job.py locally on sample data.

Usage:
    python ml/seed_rds.py --input data/processed/

Requires RDS_HOST, RDS_DB, RDS_USER, RDS_PASSWORD in .env (or environment).
"""

import argparse
import os

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sqlalchemy import create_engine

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("CityBite-SeedRDS")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def compute_business_scores(spark: SparkSession, input_path: str) -> DataFrame:
    path = input_path.rstrip("/") + "/reviews_enriched"
    df = spark.read.parquet(path)

    return (
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


def compute_grid_aggregates(business_scores: DataFrame) -> DataFrame:
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


def get_rds_engine():
    host = os.environ["RDS_HOST"]
    port = os.environ.get("RDS_PORT", "5432")
    db   = os.environ["RDS_DB"]
    user = os.environ["RDS_USER"]
    pw   = os.environ["RDS_PASSWORD"]
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")


def write_to_rds(df: DataFrame, table: str, engine) -> None:
    pandas_df = df.toPandas()
    pandas_df.to_sql(
        name=table,
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )
    print(f"  Wrote {len(pandas_df):,} rows → {table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed RDS from local processed Parquet")
    parser.add_argument("--input", default="data/processed/",
                        help="Path to processed/ directory containing reviews_enriched/")
    args = parser.parse_args()

    for var in ("RDS_HOST", "RDS_DB", "RDS_USER", "RDS_PASSWORD"):
        if not os.environ.get(var):
            raise SystemExit(f"Missing required env var: {var}. Check your .env file.")

    engine = get_rds_engine()
    spark  = build_spark()

    print("Computing business_scores ...")
    biz_scores = compute_business_scores(spark, args.input)
    biz_scores.cache()

    print("Computing grid_aggregates ...")
    grid_agg = compute_grid_aggregates(biz_scores)

    print("Writing to RDS ...")
    write_to_rds(biz_scores, "business_scores", engine)
    write_to_rds(grid_agg,   "grid_aggregates", engine)

    spark.stop()
    print("\nDone. RDS seeded — run `streamlit run dashboard/app.py` to view the map.")


if __name__ == "__main__":
    main()

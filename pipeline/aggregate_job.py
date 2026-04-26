"""
PySpark aggregation job: enriched reviews → business scores + grid aggregates.

Steps:
  1. Read reviews_enriched Parquet (partitioned by city)
  2. Compute popularity_score per business
  3. Compute grid_aggregates per grid cell
  4. Write both to S3 as Parquet
  5. Write both to RDS PostgreSQL via JDBC

Usage (local):
    spark-submit pipeline/aggregate_job.py \
        --input data/processed/ --output data/gold/ --mode local

Usage (EMR):
    spark-submit --packages org.postgresql:postgresql:42.6.0 \
        pipeline/aggregate_job.py \
        --input s3://citybite-560580021963/processed/reviews_enriched/ \
        --output s3://citybite-560580021963/processed/ \
        --mode emr
"""

import argparse
import os

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # not available on EMR — env vars set via cluster config


def build_spark(mode: str) -> SparkSession:
    builder = SparkSession.builder.appName("CityBite-Aggregate")
    if mode == "local":
        builder = builder.master("local[*]").config("spark.sql.shuffle.partitions", "4")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def compute_business_scores(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Compute per-business popularity scores.

    popularity_score = 0.4 * avg_rating
                     + 0.4 * log(review_count + 1)
                     + 0.2 * recency_score
    """
    enriched_path = input_path.rstrip("/")
    df = spark.read.parquet(enriched_path)

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
    # Determine the most common category in each grid cell deterministically
    # by ranking category counts and picking rank 1 (ties broken alphabetically).
    cat_counts = business_scores.groupBy("grid_cell", "metro_area", "categories").count()
    w = Window.partitionBy("grid_cell", "metro_area").orderBy(F.desc("count"), "categories")
    top_cats = (
        cat_counts
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select("grid_cell", "metro_area", F.col("categories").alias("top_cuisine"))
    )

    grid_agg = (
        business_scores
        .groupBy("grid_cell", "metro_area")
        .agg(
            F.avg("latitude").alias("center_lat"),
            F.avg("longitude").alias("center_lng"),
            F.avg("popularity_score").alias("avg_popularity"),
            F.count("*").alias("restaurant_count"),
        )
        .join(top_cats, on=["grid_cell", "metro_area"], how="left")
    )
    return grid_agg


def write_parquet(df: DataFrame, path: str, partition_col: str = None) -> None:
    writer = df.write.mode("overwrite")
    if partition_col:
        writer = writer.partitionBy(partition_col)
    writer.parquet(path)
    print(f"  Wrote Parquet → {path}")


def write_jdbc(df: DataFrame, table: str) -> None:
    """Write a DataFrame to RDS PostgreSQL via JDBC."""
    host = os.environ["RDS_HOST"]
    port = os.environ.get("RDS_PORT", "5432")
    db   = os.environ["RDS_DB"]
    user = os.environ["RDS_USER"]
    pw   = os.environ["RDS_PASSWORD"]

    jdbc_url = f"jdbc:postgresql://{host}:{port}/{db}"
    props = {
        "user":   user,
        "password": pw,
        "driver": "org.postgresql.Driver",
    }

    df.write.jdbc(url=jdbc_url, table=table, mode="overwrite", properties=props)
    print(f"  Wrote JDBC → {table}")


_RDS_ENV_KEYS = ("RDS_HOST", "RDS_PORT", "RDS_DB", "RDS_USER", "RDS_PASSWORD")


def _require_rds_env() -> None:
    missing = [k for k in _RDS_ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars for JDBC write: {missing}. "
            "Set them in .env or pass --skip-jdbc for local testing."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CityBite PySpark aggregation job")
    parser.add_argument("--input",  required=True, help="Path to reviews_enriched Parquet")
    parser.add_argument("--output", required=True, help="Base output path for S3/local Parquet")
    parser.add_argument("--mode",   choices=["local", "emr"], default="emr")
    parser.add_argument("--skip-jdbc", action="store_true",
                        help="Skip JDBC write (useful for local testing without RDS)")
    args = parser.parse_args()

    if not args.skip_jdbc:
        _require_rds_env()

    spark = build_spark(args.mode)

    print("Computing business_scores ...")
    biz_scores = compute_business_scores(spark, args.input)
    biz_scores.cache()
    print(f"  Businesses: {biz_scores.count():,}")

    print("Computing grid_aggregates ...")
    grid_agg = compute_grid_aggregates(biz_scores)
    grid_agg.cache()
    print(f"  Grid cells: {grid_agg.count():,}")

    # S3 / local Parquet output
    out = args.output.rstrip("/")
    write_parquet(biz_scores, f"{out}/business_scores", partition_col="metro_area")
    write_parquet(grid_agg,   f"{out}/grid_aggregates", partition_col="metro_area")

    # RDS JDBC write
    if not args.skip_jdbc:
        print("Writing to RDS ...")
        write_jdbc(biz_scores, "business_scores")
        write_jdbc(grid_agg,   "grid_aggregates")
    else:
        print("Skipping JDBC write (--skip-jdbc set).")

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()

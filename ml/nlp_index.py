"""
Build natural-language business profiles for NLP-based search.

For each business, concatenates its Yelp category tags (repeated 3× for
extra weight) with a sample of up to 20 review text snippets (first 120
chars each) into a single profile_text string.  The resulting table can be
loaded into scikit-learn TF-IDF at query time so users can find restaurants
by describing what they want in plain English.

Output table: business_profiles
    business_id, name, metro_area, city, latitude, longitude,
    avg_rating, review_count, popularity_score, profile_text

Usage (local):
    spark-submit ml/nlp_index.py --input data/processed/ --mode local

Usage (EMR):
    python pipeline/submit_emr.py nlp
"""

import argparse
import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_LOCAL_DB = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")

# Number of review snippets to include per business in the profile
_REVIEW_SAMPLE = 20
# Characters to keep from each review
_SNIPPET_LEN = 120


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("CityBite-NLPIndex")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def build_profiles(spark: SparkSession, input_path: str):
    """
    Read reviews_enriched Parquet and produce one row per business with a
    concatenated profile_text (categories × 3 + review snippets).

    Handles both old schema (city only) and new schema (metro_area + city).
    """
    path = input_path.rstrip("/") + "/reviews_enriched"
    df = spark.read.parquet(path)

    # Support both old (city-only) and new (metro_area) schemas
    has_metro = "metro_area" in df.columns
    if not has_metro:
        df = df.withColumn("metro_area", F.col("city"))

    # Truncate each review to _SNIPPET_LEN chars
    df = df.withColumn("snippet", F.substring(F.col("text"), 1, _SNIPPET_LEN))

    # Collect up to _REVIEW_SAMPLE snippets per business as a list
    snippets = (
        df.groupBy("business_id")
        .agg(
            F.first("name").alias("name"),
            F.first("metro_area").alias("metro_area"),
            F.first("city").alias("city"),
            F.first("latitude").alias("latitude"),
            F.first("longitude").alias("longitude"),
            F.first("categories").alias("categories"),
            F.avg("stars").alias("avg_rating"),
            F.count("*").alias("review_count"),
            (
                F.avg("stars") * 0.4
                + F.log(F.count("*") + 1) * 0.4
                + (F.sum("recency_weight") / F.count("*")) * 0.2
            ).alias("popularity_score"),
            F.collect_list("snippet").alias("snippets"),
        )
    )

    # Build profile_text: "categories categories categories review1 review2 ..."
    @F.udf(StringType())
    def make_profile(categories, snippets):
        cats = (categories or "").strip()
        cat_block = f"{cats} {cats} {cats}"
        sample = snippets[:_REVIEW_SAMPLE] if snippets else []
        return (cat_block + " " + " ".join(sample)).strip()

    return (
        snippets
        .withColumn("profile_text", make_profile(F.col("categories"), F.col("snippets")))
        .drop("snippets", "categories")
    )


def write_to_sqlite(df, db_path: str) -> None:
    from sqlalchemy import create_engine

    pandas_df = df.toPandas()
    engine = create_engine(f"sqlite:///{db_path}")
    pandas_df.to_sql(
        "business_profiles",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"  Wrote {len(pandas_df):,} rows → business_profiles (SQLite)")


def write_to_rds(df) -> None:
    from sqlalchemy import create_engine

    host = os.environ["RDS_HOST"]
    port = os.environ.get("RDS_PORT", "5432")
    db   = os.environ["RDS_DB"]
    user = os.environ["RDS_USER"]
    pw   = os.environ["RDS_PASSWORD"]
    engine = create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")

    pandas_df = df.toPandas()
    pandas_df.to_sql(
        "business_profiles",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )
    print(f"  Wrote {len(pandas_df):,} rows → business_profiles (RDS)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NLP business profiles")
    parser.add_argument("--input", default="data/processed/",
                        help="Path to processed/ directory containing reviews_enriched/")
    parser.add_argument("--mode", choices=["local", "emr"], default="local")
    args = parser.parse_args()

    spark = build_spark()

    print("Building business profiles ...")
    profiles = build_profiles(spark, args.input)
    profiles.cache()
    count = profiles.count()
    print(f"  {count:,} business profiles built")

    if args.mode == "local":
        write_to_sqlite(profiles, _LOCAL_DB)
    else:
        write_to_rds(profiles)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()

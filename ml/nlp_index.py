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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # not installed on EMR
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_LOCAL_DB = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")

# Number of review snippets to include per business in the profile
_REVIEW_SAMPLE = 20
# Characters to keep from each review
_SNIPPET_LEN = 120


def build_spark(mode: str) -> SparkSession:
    builder = SparkSession.builder.appName("CityBite-NLPIndex")
    if mode == "local":
        builder = builder.master("local[*]").config("spark.sql.shuffle.partitions", "4")
    spark = builder.getOrCreate()
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
    import io
    import psycopg2

    pandas_df = df.toPandas()

    conn = psycopg2.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", "5432")),
        dbname=os.environ["RDS_DB"],
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        connect_timeout=30,
    )
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS business_profiles")
    cur.execute("""
        CREATE TABLE business_profiles (
            business_id      VARCHAR(50) PRIMARY KEY,
            name             TEXT,
            metro_area       VARCHAR(100),
            city             VARCHAR(100),
            latitude         FLOAT,
            longitude        FLOAT,
            avg_rating       FLOAT,
            review_count     INT,
            popularity_score FLOAT,
            profile_text     TEXT
        )
    """)

    cols = ["business_id", "name", "metro_area", "city",
            "latitude", "longitude", "avg_rating", "review_count",
            "popularity_score", "profile_text"]
    out = pandas_df[cols].dropna(subset=["business_id"]).copy()
    out["review_count"] = out["review_count"].astype(int)

    # Strip tabs/newlines from text fields so psycopg2 COPY doesn't break
    for col in ("name", "metro_area", "city", "profile_text"):
        out[col] = out[col].fillna("").str.replace("\t", " ", regex=False) \
                                      .str.replace("\n", " ", regex=False) \
                                      .str.replace("\r", " ", regex=False)

    buf = io.StringIO()
    out.to_csv(buf, sep="\t", index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_from(buf, "business_profiles", sep="\t", null="\\N", columns=cols)

    cur.execute("CREATE INDEX idx_profiles_metro ON business_profiles(metro_area)")
    conn.commit()
    cur.close()
    conn.close()
    print(f"  Wrote {len(out):,} rows → business_profiles via psycopg2 COPY")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NLP business profiles")
    parser.add_argument("--input", default="data/processed/",
                        help="Path to processed/ directory containing reviews_enriched/")
    parser.add_argument("--mode", choices=["local", "emr"], default="local")
    args = parser.parse_args()

    spark = build_spark(args.mode)

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

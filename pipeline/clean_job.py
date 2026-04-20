"""
PySpark cleaning job: raw Yelp JSON → enriched reviews Parquet.

Steps:
  1. Read business.json + review.json
  2. Drop nulls on key columns; filter is_open == 1; keep only restaurants
  3. Normalize city name (title-case) and state (upper-case)
  4. Assign metro_area via haversine-based greedy clustering (50 km radius)
  5. Join reviews ← businesses on business_id
  6. Add recency_weight = 1 / (1 + days_since_review / 365)
  7. Add grid_cell = "<lat_bucket>_<lng_bucket>" (0.1-degree grid)
  8. Write Parquet partitioned by metro_area

Usage (local):
    spark-submit pipeline/clean_job.py \
        --input data/sample/ --output data/processed/ --mode local

Usage (EMR / S3):
    spark-submit pipeline/clean_job.py \
        --input s3://citybite/raw/ --output s3://citybite/processed/reviews_enriched/
"""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, StringType

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # not available on EMR — paths come from CLI args

METRO_RADIUS_KM = 50.0

# Yelp top-level and common subcategory tags that indicate a food/drink business.
# A business passes if ANY of its comma-split category tags (lowercased) is in this set.
# This excludes Active Life, Auto, Beauty & Spas, Arts & Entertainment, etc.
FOOD_DRINK_ANCHORS: frozenset = frozenset([
    # Yelp "Restaurants" parent + all cuisine subcategories fall under this tag
    "restaurants",
    # Yelp "Food" parent — covers bakeries, delis, coffee, grocery, specialty food, etc.
    "food",
    # Drink establishments not always tagged "Restaurants"
    "bars", "pubs", "sports bars", "dive bars", "cocktail bars", "wine bars",
    "breweries", "brewpubs", "wineries", "distilleries",
    # Coffee / tea / juice — sometimes standalone without "Food" parent
    "coffee & tea", "cafes", "juice bars & smoothies", "bubble tea",
    # Dessert / sweet spots sometimes standalone
    "desserts", "ice cream & frozen yogurt", "bakeries", "patisserie/cake shop",
    # Other common standalone food tags
    "food trucks", "food stands", "caterers", "personal chefs",
    "diners", "fast food", "pizza", "burgers", "sandwiches", "sushi bars",
    "steakhouses", "seafood", "buffets", "creperies", "waffles",
])

REQUIRED_REVIEW_COLS = ["review_id", "business_id", "user_id", "stars", "date", "text"]
REQUIRED_BUSINESS_COLS = [
    "business_id", "is_open", "name", "city", "state",
    "latitude", "longitude", "categories", "review_count",
]


def validate_windows_local_hadoop(mode: str) -> None:
    if mode != "local" or os.name != "nt":
        return

    hadoop_home = os.getenv("HADOOP_HOME") or os.getenv("hadoop.home.dir")
    if not hadoop_home:
        raise RuntimeError(
            "Windows local Spark requires HADOOP_HOME (or hadoop.home.dir). "
            "Set it to a folder containing bin/winutils.exe and bin/hadoop.dll."
        )

    bin_dir = Path(hadoop_home) / "bin"
    missing = []
    for filename in ("winutils.exe", "hadoop.dll"):
        candidate = bin_dir / filename
        if not candidate.exists():
            missing.append(str(candidate))

    if missing:
        raise RuntimeError(
            "Windows Hadoop binaries are incomplete. Missing required file(s): "
            f"{', '.join(missing)}. Ensure both winutils.exe and hadoop.dll are present "
            "under HADOOP_HOME/bin and come from a compatible Hadoop 3.x build."
        )


def build_spark(mode: str) -> SparkSession:
    master = "local[*]" if mode == "local" else None
    builder = SparkSession.builder.appName("CityBite-Clean")
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_json(spark: SparkSession, path: str, filename: str):
    full_path = path.rstrip("/") + "/" + filename
    return spark.read.json(full_path)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km — pure Python, safe on EMR without external libs."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_metro_map(city_anchors_pd, radius_km: float = METRO_RADIUS_KM) -> dict:
    """
    Greedy single-linkage metro clustering.

    Cities are sorted descending by total_reviews so the highest-volume city
    in each geographic cluster becomes the anchor (and its name labels the metro).
    Returns {city_key -> metro_name}, e.g.:
        {"Nashville_TN": "Nashville", "Brentwood_TN": "Nashville",
         "Portland_OR": "Portland", "Portland_ME": "Portland, ME"}
    """
    df = city_anchors_pd.sort_values("total_reviews", ascending=False).reset_index(drop=True)
    metro_map: dict = {}
    anchors: list = []
    used_names: set = set()

    for _, row in df.iterrows():
        ck = row["city_key"]
        assigned = False
        for anchor in anchors:
            if haversine_km(
                row["center_lat"], row["center_lng"],
                anchor["center_lat"], anchor["center_lng"],
            ) <= radius_km:
                metro_map[ck] = anchor["metro_name"]
                assigned = True
                break
        if not assigned:
            city = row["city"]
            state = row["state"]
            metro_name = city if city not in used_names else f"{city}, {state}"
            used_names.add(metro_name)
            metro_map[ck] = metro_name
            anchors.append({
                "center_lat": row["center_lat"],
                "center_lng": row["center_lng"],
                "metro_name": metro_name,
            })

    return metro_map


def add_recency_weight(df, date_col: str = "date"):
    today = datetime.now()
    today_ts = F.lit(today.strftime("%Y-%m-%d"))
    days_since = F.datediff(today_ts, F.to_date(F.col(date_col), "yyyy-MM-dd HH:mm:ss"))
    return df.withColumn("recency_weight", F.lit(1.0) / (F.lit(1.0) + days_since.cast(DoubleType()) / F.lit(365.0)))


def add_grid_cell(df, lat_col: str = "latitude", lng_col: str = "longitude"):
    lat_bucket = F.floor(F.col(lat_col) / F.lit(0.1)) * F.lit(0.1)
    lng_bucket = F.floor(F.col(lng_col) / F.lit(0.1)) * F.lit(0.1)
    cell = F.concat(F.format_number(lat_bucket, 1), F.lit("_"), F.format_number(lng_bucket, 1))
    return df.withColumn("grid_cell", cell)


_is_food_udf = F.udf(
    lambda cats: bool(
        cats and {t.strip().lower() for t in cats.split(",")} & FOOD_DRINK_ANCHORS
    ),
    BooleanType(),
)


def clean_businesses(df, spark, radius_km: float = METRO_RADIUS_KM):
    # Step 1: select required columns, drop bad rows, keep only food/drink businesses
    cleaned = (
        df.select(*REQUIRED_BUSINESS_COLS)
          .dropna(subset=["business_id", "latitude", "longitude"])
          .filter(F.col("is_open") == 1)
          .filter(_is_food_udf(F.col("categories")))
    )
    # Step 2: normalize capitalization so "las vegas" == "Las Vegas"
    cleaned = (
        cleaned
        .withColumn("city",  F.initcap(F.trim(F.col("city"))))
        .withColumn("state", F.upper(F.trim(F.col("state"))))
    )
    # Step 3: disambiguation key prevents Portland OR colliding with Portland ME
    cleaned = cleaned.withColumn(
        "city_key", F.concat(F.col("city"), F.lit("_"), F.col("state"))
    )
    # Step 4: compute per-city center coords + total review volume on driver
    anchors_pd = (
        cleaned
        .groupBy("city_key", "city", "state")
        .agg(
            F.avg("latitude").alias("center_lat"),
            F.avg("longitude").alias("center_lng"),
            F.sum(F.col("review_count").cast("long")).alias("total_reviews"),
        )
        .toPandas()
    )
    # Step 5: greedy metro clustering (driver-side), broadcast result to workers
    metro_map = build_metro_map(anchors_pd, radius_km=radius_km)
    bc_metro = spark.sparkContext.broadcast(metro_map)
    lookup_metro = F.udf(lambda ck: bc_metro.value.get(ck, ck), StringType())

    return cleaned.withColumn("metro_area", lookup_metro(F.col("city_key"))).drop("city_key")


def clean_reviews(df):
    return (
        df.select(*REQUIRED_REVIEW_COLS)
          .dropna(subset=REQUIRED_REVIEW_COLS)
    )


def build_enriched(reviews, businesses):
    joined = reviews.join(businesses, on="business_id", how="inner")
    with_recency = add_recency_weight(joined)
    return add_grid_cell(with_recency)


def write_output(df, output_path: str, mode: str) -> None:
    dest = output_path if mode != "local" else output_path.rstrip("/") + "/reviews_enriched"
    print(f"Writing enriched reviews to {dest}...")
    try:
        (
            df.write
              .mode("overwrite")
              .partitionBy("metro_area")
              .parquet(dest)
        )
    except Exception as e:
        error_text = str(e)
        if os.name == "nt" and "HADOOP_HOME and hadoop.home.dir are unset" in error_text:
            raise RuntimeError(
                "Spark local write failed on Windows because winutils.exe is not configured. "
                "Install a Hadoop winutils binary and set HADOOP_HOME to its parent folder "
                "(which must contain bin/winutils.exe), then rerun this job."
            ) from e
        if os.name == "nt" and "NativeIO$Windows.access0" in error_text:
            raise RuntimeError(
                "Spark local write failed on Windows due to Hadoop native library mismatch. "
                "Make sure HADOOP_HOME/bin contains BOTH winutils.exe and hadoop.dll from "
                "the same Hadoop 3.x build, then reopen terminal and rerun."
            ) from e
        raise
    print("Write complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CityBite PySpark cleaning job")
    parser.add_argument("--input", required=True, help="Path to raw JSON files")
    parser.add_argument("--output", required=True, help="Output path for enriched Parquet")
    parser.add_argument("--mode", choices=["local", "emr"], default="local")
    args = parser.parse_args()

    validate_windows_local_hadoop(args.mode)

    spark = build_spark(args.mode)

    print("Reading business data ...")
    raw_businesses = read_json(spark, args.input, "yelp_academic_dataset_business.json")
    businesses = clean_businesses(raw_businesses, spark)
    print(f"  Open restaurants: {businesses.count():,}")

    print("Reading review data ...")
    raw_reviews = read_json(spark, args.input, "yelp_academic_dataset_review.json")
    reviews = clean_reviews(raw_reviews)
    print(f"  Clean reviews: {reviews.count():,}")

    enriched = build_enriched(reviews, businesses)
    print(f"  Enriched rows: {enriched.count():,}")

    write_output(enriched, args.output, args.mode)
    spark.stop()


if __name__ == "__main__":
    main()

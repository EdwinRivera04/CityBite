"""
PySpark MLlib ALS collaborative filtering recommender.

Pipeline:
  1. Read enriched reviews Parquet (output of clean_job.py)
  2. Build user-item matrix: (user_id, business_id, stars)
  3. Encode string IDs → integer indices via StringIndexer
  4. 80/20 train/test split
  5. Train ALS model (optionally with CrossValidator grid search)
  6. Evaluate with RMSE on test set
  7. Generate top-N recommendations per user
  8. Write recommendations to local SQLite (--mode local) or RDS (--mode emr)

Usage (local):
    spark-submit ml/als_train.py --input data/processed/ --mode local

Usage (EMR / S3):
    spark-submit ml/als_train.py \
        --input s3://citybite/processed/reviews_enriched/ --mode emr
"""

import argparse
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # not available on EMR — env vars set via cluster Configurations

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ALS default hyperparameters (override via --rank / --reg-param CLI args)
DEFAULT_RANK = 20
DEFAULT_MAX_ITER = 10
DEFAULT_REG_PARAM = 0.1

# Top-N recommendations to generate per user
TOP_N_RECS = 10


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def build_spark(mode: str) -> SparkSession:
    """Create a SparkSession; use local[*] master when running locally."""
    builder = SparkSession.builder.appName("CityBite-ALS")
    if mode == "local":
        builder = builder.master("local[*]")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reviews(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Load enriched reviews Parquet.

    Expects columns produced by clean_job.py:
        user_id, business_id, stars, city, grid_cell, ...
    """
    path = input_path.rstrip("/") + "/reviews_enriched"
    df = spark.read.parquet(path)
    # Keep only the three columns needed for the rating matrix
    return df.select("user_id", "business_id", F.col("stars").cast("float").alias("stars"))


# ---------------------------------------------------------------------------
# User-item matrix construction
# ---------------------------------------------------------------------------

def build_user_item_matrix(df: DataFrame):
    """
    Encode string user/business IDs as integer indices required by ALS.

    ALS in Spark MLlib requires numeric (integer) user and item columns.
    StringIndexer assigns a dense integer index to each unique string value,
    sorted by descending frequency so the most-reviewed businesses get index 0.

    Returns:
        matrix_df  - DataFrame with columns: user_idx, business_idx, stars
        user_map   - {int_index: user_id}   for decoding predictions
        biz_map    - {int_index: business_id}
    """
    # Fit indexers on the full dataset so train + test share the same vocab
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx",
                                 handleInvalid="skip")
    biz_indexer = StringIndexer(inputCol="business_id", outputCol="business_idx",
                                handleInvalid="skip")
    pipeline = Pipeline(stages=[user_indexer, biz_indexer])
    model = pipeline.fit(df)
    indexed = model.transform(df)

    # Cast indices to int (ALS requires IntegerType, not DoubleType)
    matrix_df = (
        indexed
        .withColumn("user_idx", F.col("user_idx").cast("int"))
        .withColumn("business_idx", F.col("business_idx").cast("int"))
        .select("user_idx", "business_idx", "stars")
    )

    # Build reverse-lookup maps for decoding top-N recommendations
    user_labels = model.stages[0].labels       # list: index → user_id string
    biz_labels = model.stages[1].labels        # list: index → business_id string
    user_map = {i: uid for i, uid in enumerate(user_labels)}
    biz_map = {i: bid for i, bid in enumerate(biz_labels)}

    return matrix_df, user_map, biz_map


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_als(
    train_df: DataFrame,
    rank: int = DEFAULT_RANK,
    max_iter: int = DEFAULT_MAX_ITER,
    reg_param: float = DEFAULT_REG_PARAM,
    use_cv: bool = False,
):
    """
    Train ALS model.

    When use_cv=True, runs a 2-fold CrossValidator over a small param grid
    (rank × regParam) — useful when tuning on the full dataset on EMR.
    For local/sample dev, use_cv=False to keep runtime reasonable.

    coldStartStrategy="drop" prevents NaN RMSE on users/items not in training.
    """
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="user_idx",
        itemCol="business_idx",
        ratingCol="stars",
        coldStartStrategy="drop",  # drop test rows for unseen users/items
        nonnegative=True,          # ratings are non-negative stars
    )

    if not use_cv:
        return als.fit(train_df)

    # Grid search — only used when explicitly requested (e.g., EMR full dataset)
    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10, 20])
        .addGrid(als.regParam, [0.01, 0.1])
        .build()
    )
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="stars",
        predictionCol="prediction",
    )
    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=2,
        parallelism=2,
    )
    return cv.fit(train_df).bestModel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rmse(model, test_df: DataFrame) -> float:
    """Return RMSE of the ALS model on the held-out test set."""
    predictions = model.transform(test_df)
    # Drop rows where prediction is NaN (cold-start users/items)
    predictions = predictions.filter(F.col("prediction").isNotNull())
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="stars",
        predictionCol="prediction",
    )
    return evaluator.evaluate(predictions)


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

def generate_recommendations(
    model,
    biz_map: dict,
    user_map: dict,
    top_n: int = TOP_N_RECS,
) -> "pandas.DataFrame":  # noqa: F821
    """
    Generate top-N business recommendations for every user.

    ALS.recommendForAllUsers returns a DataFrame:
        user_idx | recommendations: [{business_idx, rating}, ...]

    We explode that, convert integer indices back to original string IDs,
    and return a tidy pandas DataFrame ready to write to the database.
    """
    import pandas as pd

    recs_df = model.recommendForAllUsers(top_n)

    # Explode the array of (business_idx, rating) structs into individual rows
    recs_exploded = (
        recs_df
        .select(
            F.col("user_idx"),
            F.posexplode(F.col("recommendations")).alias("rank_0based", "rec"),
        )
        .select(
            F.col("user_idx"),
            (F.col("rank_0based") + 1).alias("rank"),       # 1-based rank
            F.col("rec.business_idx").alias("business_idx"),
            F.col("rec.rating").alias("predicted_rating"),
        )
    )

    # Use toPandas() + vectorized map instead of collect() + Python loop.
    # collect() on 1.8M users × 10 recs = 18M rows iterates row-by-row in
    # Python and hangs. toPandas() uses Arrow-based transfer and pandas .map()
    # is vectorized — orders of magnitude faster on large datasets.
    recs_pd = recs_exploded.toPandas()
    recs_pd["user_id"] = recs_pd["user_idx"].map(user_map)
    recs_pd["business_id"] = recs_pd["business_idx"].map(biz_map)
    recs_pd = recs_pd.dropna(subset=["user_id", "business_id"])
    recs_pd["predicted_rating"] = recs_pd["predicted_rating"].astype(float)
    recs_pd["rank"] = recs_pd["rank"].astype(int)

    return recs_pd[["user_id", "business_id", "predicted_rating", "rank"]]


# ---------------------------------------------------------------------------
# Database I/O
# ---------------------------------------------------------------------------

def get_db_url(mode: str) -> str:
    """
    Build a SQLAlchemy connection URL.
    - local → SQLite file at data/citybite_local.db
    - emr   → PostgreSQL RDS (reads env vars set in .env)
    """
    if mode == "local":
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "citybite_local.db"
        )
        return f"sqlite:///{db_path}"

    host = os.environ["RDS_HOST"]
    port = os.environ.get("RDS_PORT", "5432")
    db = os.environ["RDS_DB"]
    user = os.environ["RDS_USER"]
    pw = os.environ["RDS_PASSWORD"]
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


def write_recommendations(recs_pd, db_url: str) -> None:
    """
    Write ALS recommendations into the als_recommendations table.

    Uses psycopg2 COPY for PostgreSQL (EMR) to avoid pandas/SQLAlchemy version
    conflicts on EMR Python 3.7.  Falls back to SQLAlchemy to_sql for SQLite
    (local dev).
    """
    if db_url.startswith("sqlite"):
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        with engine.begin() as conn:
            recs_pd.to_sql(
                name="als_recommendations",
                con=conn,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=10000,
            )
        print(f"  Wrote {len(recs_pd):,} recommendation rows to {db_url}")
        return

    # PostgreSQL via direct psycopg2 — avoids EMR's old pandas/SQLAlchemy clash
    import io
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", "5432")),
        dbname=os.environ["RDS_DB"],
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        connect_timeout=30,
    )
    cur = conn.cursor()

    # Full refresh: drop + recreate so schema stays consistent
    cur.execute("DROP TABLE IF EXISTS als_recommendations")
    cur.execute("""
        CREATE TABLE als_recommendations (
            user_id          VARCHAR(50),
            business_id      VARCHAR(50),
            predicted_rating FLOAT,
            rank             INT,
            PRIMARY KEY (user_id, business_id)
        )
    """)

    # COPY is orders of magnitude faster than INSERT for 11M rows.
    # Tab-separated avoids comma-in-data issues; user/business IDs are alphanumeric.
    # na_rep/null="\\N" ensures PostgreSQL COPY can handle any NaN values.
    out = recs_pd[["user_id", "business_id", "predicted_rating", "rank"]].dropna().copy()
    out["rank"] = out["rank"].astype(int)
    out["predicted_rating"] = out["predicted_rating"].astype(float)

    buf = io.StringIO()
    out.to_csv(buf, sep="\t", index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_from(buf, "als_recommendations",
                  sep="\t", null="\\N",
                  columns=["user_id", "business_id", "predicted_rating", "rank"])

    cur.execute("CREATE INDEX idx_als_user     ON als_recommendations(user_id)")
    cur.execute("CREATE INDEX idx_als_business ON als_recommendations(business_id)")

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Wrote {len(recs_pd):,} recommendation rows via psycopg2 COPY")
    print("  Created indices on als_recommendations")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CityBite ALS recommender training")
    parser.add_argument("--input",     required=True,  help="Path to processed/ directory")
    parser.add_argument("--mode",      choices=["local", "emr"], default="local")
    parser.add_argument("--rank",      type=int,   default=DEFAULT_RANK)
    parser.add_argument("--max-iter",  type=int,   default=DEFAULT_MAX_ITER)
    parser.add_argument("--reg-param", type=float, default=DEFAULT_REG_PARAM)
    parser.add_argument("--cv",        action="store_true",
                        help="Enable CrossValidator hyperparameter search")
    args = parser.parse_args()

    spark = build_spark(args.mode)

    # ── Step 1: Load enriched reviews ──────────────────────────────────────
    print("Loading reviews ...")
    reviews = load_reviews(spark, args.input)
    total = reviews.count()
    print(f"  Total rating rows: {total:,}")

    # ── Step 2: Build integer-indexed user-item matrix ─────────────────────
    print("Building user-item matrix ...")
    matrix_df, user_map, biz_map = build_user_item_matrix(reviews)
    print(f"  Unique users:      {len(user_map):,}")
    print(f"  Unique businesses: {len(biz_map):,}")

    # ── Step 3: Train / test split ─────────────────────────────────────────
    train_df, test_df = matrix_df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count():,}  |  Test: {test_df.count():,}")

    # ── Step 4: Train ALS ──────────────────────────────────────────────────
    print(f"Training ALS (rank={args.rank}, regParam={args.reg_param}, cv={args.cv}) ...")
    model = train_als(
        train_df,
        rank=args.rank,
        max_iter=args.max_iter,
        reg_param=args.reg_param,
        use_cv=args.cv,
    )

    # ── Step 5: Evaluate ───────────────────────────────────────────────────
    rmse = evaluate_rmse(model, test_df)
    print(f"  RMSE on test set: {rmse:.4f}")
    if rmse >= 1.5:
        print("  WARNING: RMSE >= 1.5 — consider tuning rank/regParam or using --cv")

    # ── Step 6: Generate top-N recommendations ─────────────────────────────
    print(f"Generating top-{TOP_N_RECS} recommendations per user ...")
    recs_pd = generate_recommendations(model, biz_map, user_map, top_n=TOP_N_RECS)
    print(f"  Generated {len(recs_pd):,} recommendation rows")

    # ── Step 7: Write to database ──────────────────────────────────────────
    db_url = get_db_url(args.mode)
    print(f"Writing to {db_url} ...")
    write_recommendations(recs_pd, db_url)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()

"""
Train a scikit-learn sentiment classifier on review text, evaluate F1, then
compute and persist grid-level sentiment scores.

Steps (local mode):
  1. Load review text + stars from processed Parquet (pandas)
  2. Label: positive (stars >= 4), negative (stars <= 2), drop neutral (3)
  3. Train TF-IDF (10K terms, unigrams+bigrams) + LogisticRegression classifier
  4. Evaluate: accuracy, weighted F1, confusion matrix on 20% held-out test set
  5. Compute per-grid-cell sentiment score (% positive reviews)
  6. Write grid_sentiment table to SQLite (local) or RDS PostgreSQL (emr)

Usage (local):
    spark-submit ml/sentiment.py --input data/processed/ --mode local

Usage (EMR / S3):
    spark-submit ml/sentiment.py --input s3://citybite/processed/ --mode emr
"""

import argparse
import os
import sqlite3

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # not available on EMR — env vars set via cluster Configurations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sqlalchemy import create_engine

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
    """Load grid_cell + stars for grid-level sentiment aggregation."""
    path = input_path.rstrip("/") + "/reviews_enriched"
    df = pd.read_parquet(path, columns=["grid_cell", "stars"])
    return df


def load_reviews_for_classifier(input_path: str) -> pd.DataFrame:
    """Load text + stars for sklearn classifier training (local mode only)."""
    path = input_path.rstrip("/") + "/reviews_enriched"
    return pd.read_parquet(path, columns=["text", "stars"])


# ---------------------------------------------------------------------------
# Scikit-learn sentiment classifier (local mode)
# ---------------------------------------------------------------------------

def train_and_evaluate_classifier(reviews: pd.DataFrame):
    """
    Train a TF-IDF + LogisticRegression classifier on review text.

    Labels are derived from star ratings (same convention as the grid scorer):
        positive → stars >= 4
        negative → stars <= 2
        neutral  → stars == 3 (dropped — ambiguous signal)

    Limits training corpus to 200K rows for speed on local hardware.

    Returns:
        (accuracy, weighted_f1) — both in [0, 1]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    # Drop neutral reviews; assign binary label
    labeled = reviews[reviews["stars"] != 3].copy()
    labeled["label"] = (labeled["stars"] >= 4).astype(int)

    # Cap at 200K for local dev speed
    if len(labeled) > 200_000:
        labeled = labeled.sample(200_000, random_state=42)

    X = labeled["text"].fillna("").values
    y = labeled["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000)),
    ])

    print(f"  Training on {len(X_train):,} reviews, testing on {len(X_test):,} ...")
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 score : {f1:.4f}  (target >= 0.80)")
    print(f"  Confusion matrix (rows=actual, cols=predicted):\n{cm}")

    if f1 < 0.80:
        print("  WARNING: F1 < 0.80 — consider adding more training data or tuning TF-IDF params")

    return acc, f1


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
        with engine.begin() as conn:
            sentiment_pd.to_sql(
                name="grid_sentiment",
                con=conn,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=2000,
            )
        print(f"  Wrote {len(sentiment_pd):,} rows to grid_sentiment")
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
    cur.execute("DROP TABLE IF EXISTS grid_sentiment")
    cur.execute("""
        CREATE TABLE grid_sentiment (
            grid_cell       VARCHAR(20) PRIMARY KEY,
            sentiment_score FLOAT,
            positive_count  INT,
            negative_count  INT,
            last_updated    TIMESTAMP DEFAULT NOW()
        )
    """)
    # Ensure clean types and drop any rows with nulls before COPY.
    # copy_from needs \N for nulls (not empty string) — na_rep handles this.
    out = (
        sentiment_pd[["grid_cell", "sentiment_score", "positive_count", "negative_count"]]
        .dropna(subset=["grid_cell", "sentiment_score"])
        .copy()
    )
    out["positive_count"] = out["positive_count"].astype(int)
    out["negative_count"] = out["negative_count"].astype(int)

    buf = io.StringIO()
    out.to_csv(buf, sep="\t", index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_from(buf, "grid_sentiment", sep="\t", null="\\N",
                  columns=["grid_cell", "sentiment_score", "positive_count", "negative_count"])
    conn.commit()
    cur.close()
    conn.close()
    print(f"  Wrote {len(sentiment_pd):,} rows to grid_sentiment via psycopg2 COPY")


def main() -> None:
    parser = argparse.ArgumentParser(description="CityBite sentiment aggregation job")
    parser.add_argument("--input", required=True, help="Path to processed/ directory")
    parser.add_argument("--mode", choices=["local", "emr"], default="local")
    args = parser.parse_args()

    if args.mode == "local":
        # ── Step 1: Train + evaluate sklearn classifier ────────────────────
        print("Loading reviews for sklearn classifier ...")
        clf_reviews = load_reviews_for_classifier(args.input)
        print(f"  Total reviews loaded: {len(clf_reviews):,}")

        print("Training TF-IDF + LogisticRegression sentiment classifier ...")
        acc, f1 = train_and_evaluate_classifier(clf_reviews)

        # ── Step 2: Compute grid-level sentiment scores ────────────────────
        print("Loading reviews for grid sentiment aggregation ...")
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

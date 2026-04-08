"""
Unit tests for ml/sentiment.py.

Covers:
  - compute_grid_sentiment_pandas  (pure pandas, no Spark needed)
  - compute_grid_sentiment_spark   (PySpark local mode)
  - train_and_evaluate_classifier  (sklearn, tiny synthetic corpus)
  - get_db_url                     (URL construction logic)
  - ensure_local_table             (SQLite schema creation)
  - write_sentiment                (round-trip to in-memory SQLite)

No AWS credentials or real dataset required.
"""

import os
import sqlite3
import tempfile

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------

def _make_reviews_pd(n_per_sentiment: int = 50) -> pd.DataFrame:
    """
    Create a tiny pandas DataFrame with grid_cell, stars, and text columns.

    Deliberately polarised corpus so the classifier can train without a huge
    dataset (positive text is clearly positive, negative text clearly negative).
    """
    positive = pd.DataFrame({
        "grid_cell": ["33.6_-112.0"] * n_per_sentiment,
        "stars": [5] * n_per_sentiment,
        "text": ["amazing excellent wonderful fantastic great food"] * n_per_sentiment,
    })
    negative = pd.DataFrame({
        "grid_cell": ["33.6_-112.0"] * n_per_sentiment,
        "stars": [1] * n_per_sentiment,
        "text": ["terrible awful disgusting horrible worst ever"] * n_per_sentiment,
    })
    neutral = pd.DataFrame({
        "grid_cell": ["33.7_-112.1"] * (n_per_sentiment // 2),
        "stars": [3] * (n_per_sentiment // 2),
        "text": ["it was okay nothing special"] * (n_per_sentiment // 2),
    })
    return pd.concat([positive, negative, neutral], ignore_index=True)


# ---------------------------------------------------------------------------
# Session-scoped SparkSession
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .appName("CityBite-Sentiment-Test")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# compute_grid_sentiment_pandas
# ---------------------------------------------------------------------------

class TestComputeGridSentimentPandas:
    def test_output_columns(self):
        from ml.sentiment import compute_grid_sentiment_pandas

        df = _make_reviews_pd()
        result = compute_grid_sentiment_pandas(df)
        assert set(result.columns) == {
            "grid_cell", "sentiment_score", "positive_count", "negative_count"
        }

    def test_score_in_range(self):
        from ml.sentiment import compute_grid_sentiment_pandas

        df = _make_reviews_pd()
        result = compute_grid_sentiment_pandas(df)
        assert result["sentiment_score"].between(0.0, 1.0).all()

    def test_positive_count_correct(self):
        from ml.sentiment import compute_grid_sentiment_pandas

        df = _make_reviews_pd(n_per_sentiment=10)
        # Only the 33.6_-112.0 cell has both positive (10) and negative (10)
        result = compute_grid_sentiment_pandas(df)
        cell_row = result[result["grid_cell"] == "33.6_-112.0"].iloc[0]
        assert cell_row["positive_count"] == 10
        assert cell_row["negative_count"] == 10

    def test_neutral_only_cell_excluded(self):
        """Grid cells with only neutral reviews should not appear (denom == 0)."""
        from ml.sentiment import compute_grid_sentiment_pandas

        df = pd.DataFrame({
            "grid_cell": ["neutral_cell"] * 5,
            "stars": [3] * 5,
        })
        result = compute_grid_sentiment_pandas(df)
        assert "neutral_cell" not in result["grid_cell"].values

    def test_all_positive_yields_score_one(self):
        from ml.sentiment import compute_grid_sentiment_pandas

        df = pd.DataFrame({
            "grid_cell": ["pos_only"] * 5,
            "stars": [5] * 5,
        })
        result = compute_grid_sentiment_pandas(df)
        assert result.iloc[0]["sentiment_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_grid_sentiment_spark
# ---------------------------------------------------------------------------

class TestComputeGridSentimentSpark:
    def test_output_columns(self, spark):
        from ml.sentiment import compute_grid_sentiment_spark

        pdf = _make_reviews_pd()
        reviews = spark.createDataFrame(pdf[["grid_cell", "stars"]])
        result = compute_grid_sentiment_spark(reviews)
        assert set(result.columns) == {
            "grid_cell", "sentiment_score", "positive_count", "negative_count"
        }

    def test_row_count_nonzero(self, spark):
        from ml.sentiment import compute_grid_sentiment_spark

        pdf = _make_reviews_pd()
        reviews = spark.createDataFrame(pdf[["grid_cell", "stars"]])
        result = compute_grid_sentiment_spark(reviews)
        assert result.count() > 0

    def test_score_in_range(self, spark):
        from ml.sentiment import compute_grid_sentiment_spark
        from pyspark.sql import functions as F

        pdf = _make_reviews_pd()
        reviews = spark.createDataFrame(pdf[["grid_cell", "stars"]])
        result = compute_grid_sentiment_spark(reviews)
        out_of_range = result.filter(
            (F.col("sentiment_score") < 0) | (F.col("sentiment_score") > 1)
        )
        assert out_of_range.count() == 0

    def test_neutral_only_cell_excluded(self, spark):
        from ml.sentiment import compute_grid_sentiment_spark

        pdf = pd.DataFrame({
            "grid_cell": ["neutral_only"] * 4,
            "stars": [3] * 4,
        })
        reviews = spark.createDataFrame(pdf)
        result = compute_grid_sentiment_spark(reviews)
        assert result.count() == 0


# ---------------------------------------------------------------------------
# train_and_evaluate_classifier
# ---------------------------------------------------------------------------

class TestTrainAndEvaluateClassifier:
    def test_returns_two_floats(self):
        from ml.sentiment import train_and_evaluate_classifier

        df = _make_reviews_pd(n_per_sentiment=60)
        acc, f1 = train_and_evaluate_classifier(df)
        assert isinstance(acc, float)
        assert isinstance(f1, float)

    def test_accuracy_in_range(self):
        from ml.sentiment import train_and_evaluate_classifier

        df = _make_reviews_pd(n_per_sentiment=60)
        acc, _ = train_and_evaluate_classifier(df)
        assert 0.0 <= acc <= 1.0

    def test_f1_in_range(self):
        from ml.sentiment import train_and_evaluate_classifier

        df = _make_reviews_pd(n_per_sentiment=60)
        _, f1 = train_and_evaluate_classifier(df)
        assert 0.0 <= f1 <= 1.0

    def test_polarised_corpus_high_accuracy(self):
        """
        A clearly polarised corpus of 200 reviews should yield accuracy > 0.85
        even with the default TF-IDF + LR setup.
        """
        from ml.sentiment import train_and_evaluate_classifier

        df = _make_reviews_pd(n_per_sentiment=100)
        acc, _ = train_and_evaluate_classifier(df)
        assert acc > 0.85

    def test_neutral_stars_dropped(self):
        """Neutral reviews (stars == 3) must not affect training labels."""
        from ml.sentiment import train_and_evaluate_classifier

        # All neutral — after dropping, the dataset should be empty and raise
        df = pd.DataFrame({
            "grid_cell": ["x"] * 10,
            "stars": [3] * 10,
            "text": ["whatever"] * 10,
        })
        # train_test_split will fail or produce degenerate output; we just need
        # to confirm the function doesn't silently corrupt labels.
        # With zero non-neutral rows the sklearn pipeline will raise — catch it.
        with pytest.raises(Exception):
            train_and_evaluate_classifier(df)


# ---------------------------------------------------------------------------
# get_db_url
# ---------------------------------------------------------------------------

class TestGetDbUrl:
    def test_local_mode_sqlite(self):
        from ml.sentiment import get_db_url

        url = get_db_url("local")
        assert url.startswith("sqlite:///")
        assert "citybite_local.db" in url

    def test_emr_mode_postgres(self, monkeypatch):
        from ml.sentiment import get_db_url

        monkeypatch.setenv("RDS_HOST", "rds.example.com")
        monkeypatch.setenv("RDS_PORT", "5432")
        monkeypatch.setenv("RDS_DB", "citybite")
        monkeypatch.setenv("RDS_USER", "admin")
        monkeypatch.setenv("RDS_PASSWORD", "secret")
        url = get_db_url("emr")
        assert url.startswith("postgresql+psycopg2://")
        assert "rds.example.com" in url


# ---------------------------------------------------------------------------
# ensure_local_table
# ---------------------------------------------------------------------------

class TestEnsureLocalTable:
    def test_creates_table(self, tmp_path, monkeypatch):
        import ml.sentiment as sent_module

        db_path = str(tmp_path / "test.db")
        monkeypatch.setattr(sent_module, "_LOCAL_DB", db_path)

        sent_module.ensure_local_table()

        conn = sqlite3.connect(db_path)
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        assert "grid_sentiment" in tables

    def test_idempotent(self, tmp_path, monkeypatch):
        import ml.sentiment as sent_module

        db_path = str(tmp_path / "test2.db")
        monkeypatch.setattr(sent_module, "_LOCAL_DB", db_path)

        # Calling twice should not raise
        sent_module.ensure_local_table()
        sent_module.ensure_local_table()


# ---------------------------------------------------------------------------
# write_sentiment
# ---------------------------------------------------------------------------

class TestWriteSentiment:
    def test_writes_rows(self, tmp_path, monkeypatch):
        import ml.sentiment as sent_module

        db_path = str(tmp_path / "write_test.db")
        monkeypatch.setattr(sent_module, "_LOCAL_DB", db_path)

        sentiment_pd = pd.DataFrame({
            "grid_cell": ["33.6_-112.0", "33.7_-112.1"],
            "sentiment_score": [0.8, 0.4],
            "positive_count": [80, 40],
            "negative_count": [20, 60],
        })
        db_url = f"sqlite:///{db_path}"
        sent_module.write_sentiment(sentiment_pd, db_url, mode="local")

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM grid_sentiment").fetchone()[0]
        conn.close()
        assert rows == 2

    def test_overwrites_on_rerun(self, tmp_path, monkeypatch):
        import ml.sentiment as sent_module

        db_path = str(tmp_path / "overwrite_test.db")
        monkeypatch.setattr(sent_module, "_LOCAL_DB", db_path)

        df1 = pd.DataFrame({
            "grid_cell": ["a", "b"],
            "sentiment_score": [0.9, 0.1],
            "positive_count": [9, 1],
            "negative_count": [1, 9],
        })
        df2 = pd.DataFrame({
            "grid_cell": ["c"],
            "sentiment_score": [0.5],
            "positive_count": [5],
            "negative_count": [5],
        })
        db_url = f"sqlite:///{db_path}"
        sent_module.write_sentiment(df1, db_url, mode="local")
        sent_module.write_sentiment(df2, db_url, mode="local")

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM grid_sentiment").fetchone()[0]
        conn.close()
        # Second write replaces the first — only 1 row should remain
        assert rows == 1

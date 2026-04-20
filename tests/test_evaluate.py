"""
Unit tests for ml/evaluate.py.

Covers:
  - precision_at_k  (pure DB query logic, tested against in-memory SQLite)
  - evaluate_sentiment_f1  (sklearn, tiny synthetic corpus)

No AWS credentials, no Spark session, no real dataset required.
ALS RMSE evaluation is intentionally excluded — it duplicates als_train tests
and requires a full Spark session + data on disk.
"""

import sqlite3
import tempfile
import os

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(rows: list[tuple]) -> str:
    """
    Write rows into a temp SQLite DB with the als_recommendations schema.
    Returns a SQLAlchemy connection URL string.
    rows: list of (user_id, business_id, predicted_rating, rank)
    """
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp)
    conn.execute("""
        CREATE TABLE als_recommendations (
            user_id          TEXT,
            business_id      TEXT,
            predicted_rating REAL,
            rank             INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO als_recommendations VALUES (?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()
    return f"sqlite:///{tmp}"


def _make_reviews(n_per_class: int = 80) -> pd.DataFrame:
    """Clearly polarised review corpus for fast sklearn training."""
    pos = pd.DataFrame({
        "text":  ["amazing great wonderful excellent fantastic"] * n_per_class,
        "stars": [5] * n_per_class,
    })
    neg = pd.DataFrame({
        "text":  ["terrible awful horrible disgusting worst ever"] * n_per_class,
        "stars": [1] * n_per_class,
    })
    return pd.concat([pos, neg], ignore_index=True)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant_returns_one(self):
        from ml.evaluate import precision_at_k

        rows = [
            ("u1", "b1", 4.5, 1),
            ("u1", "b2", 4.2, 2),
            ("u2", "b3", 5.0, 1),
        ]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert pk == pytest.approx(1.0)

    def test_none_relevant_returns_zero(self):
        from ml.evaluate import precision_at_k

        rows = [
            ("u1", "b1", 2.0, 1),
            ("u1", "b2", 1.5, 2),
            ("u2", "b3", 3.0, 1),
        ]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert pk == pytest.approx(0.0)

    def test_half_relevant(self):
        from ml.evaluate import precision_at_k

        rows = [
            ("u1", "b1", 4.5, 1),
            ("u1", "b2", 2.0, 2),
        ]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert pk == pytest.approx(0.5)

    def test_empty_table_returns_zero(self):
        from ml.evaluate import precision_at_k

        pk = precision_at_k(_make_db([]), k=10, threshold=4.0)
        assert pk == pytest.approx(0.0)

    def test_rank_cutoff_excludes_beyond_k(self):
        from ml.evaluate import precision_at_k

        # rank 11 should be excluded when k=10
        rows = [
            ("u1", "b1", 4.5, 1),   # within k — relevant
            ("u1", "b2", 4.5, 11),  # beyond k — should be ignored
        ]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        # Only rank-1 counts → 1/1 = 1.0
        assert pk == pytest.approx(1.0)

    def test_multi_user_averages_per_user(self):
        from ml.evaluate import precision_at_k

        # u1: 1/1 relevant, u2: 0/1 relevant → mean = 0.5
        rows = [
            ("u1", "b1", 4.5, 1),
            ("u2", "b2", 2.0, 1),
        ]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert pk == pytest.approx(0.5)

    def test_threshold_boundary(self):
        from ml.evaluate import precision_at_k

        # Exactly at threshold should be considered relevant
        rows = [("u1", "b1", 4.0, 1)]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert pk == pytest.approx(1.0)

    def test_custom_threshold(self):
        from ml.evaluate import precision_at_k

        rows = [
            ("u1", "b1", 3.5, 1),  # relevant at threshold=3.0, not at 4.0
            ("u1", "b2", 2.0, 2),
        ]
        pk_low  = precision_at_k(_make_db(rows), k=10, threshold=3.0)
        pk_high = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert pk_low > pk_high

    def test_result_between_zero_and_one(self):
        from ml.evaluate import precision_at_k

        rows = [
            ("u1", "b1", 4.5, 1),
            ("u1", "b2", 1.0, 2),
            ("u2", "b3", 2.0, 1),
            ("u2", "b4", 4.8, 2),
        ]
        pk = precision_at_k(_make_db(rows), k=10, threshold=4.0)
        assert 0.0 <= pk <= 1.0


# ---------------------------------------------------------------------------
# evaluate_sentiment_f1
# ---------------------------------------------------------------------------

class TestEvaluateSentimentF1:
    def test_returns_two_floats(self, tmp_path):
        from ml.evaluate import evaluate_sentiment_f1

        # Write a tiny parquet so the function can read it
        reviews = _make_reviews(80)
        parquet_dir = tmp_path / "reviews_enriched"
        parquet_dir.mkdir()
        reviews.to_parquet(str(parquet_dir / "part-0.parquet"), index=False)

        acc, f1 = evaluate_sentiment_f1(str(tmp_path))
        assert isinstance(acc, float)
        assert isinstance(f1, float)

    def test_accuracy_in_valid_range(self, tmp_path):
        from ml.evaluate import evaluate_sentiment_f1

        reviews = _make_reviews(80)
        parquet_dir = tmp_path / "reviews_enriched"
        parquet_dir.mkdir()
        reviews.to_parquet(str(parquet_dir / "part-0.parquet"), index=False)

        acc, _ = evaluate_sentiment_f1(str(tmp_path))
        assert 0.0 <= acc <= 1.0

    def test_f1_in_valid_range(self, tmp_path):
        from ml.evaluate import evaluate_sentiment_f1

        reviews = _make_reviews(80)
        parquet_dir = tmp_path / "reviews_enriched"
        parquet_dir.mkdir()
        reviews.to_parquet(str(parquet_dir / "part-0.parquet"), index=False)

        _, f1 = evaluate_sentiment_f1(str(tmp_path))
        assert 0.0 <= f1 <= 1.0

    def test_polarised_corpus_meets_target(self, tmp_path):
        """Clearly separable reviews should hit F1 >= 0.80 (project target)."""
        from ml.evaluate import evaluate_sentiment_f1

        reviews = _make_reviews(150)
        parquet_dir = tmp_path / "reviews_enriched"
        parquet_dir.mkdir()
        reviews.to_parquet(str(parquet_dir / "part-0.parquet"), index=False)

        _, f1 = evaluate_sentiment_f1(str(tmp_path))
        assert f1 >= 0.80

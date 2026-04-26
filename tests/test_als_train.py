"""
Unit tests for ml/als_train.py.

All tests run in PySpark local mode — no AWS or database required.
The fixture creates a minimal in-memory ratings DataFrame that exercises
every public function in als_train without touching the filesystem.
"""

import pytest
from pyspark.sql import SparkSession

from ml.als_train import (
    build_user_item_matrix,
    evaluate_rmse,
    generate_recommendations,
    get_db_url,
    train_als,
    write_recommendations,
)


# ---------------------------------------------------------------------------
# Session-scoped SparkSession — reused across all tests for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .appName("CityBite-ALS-Test")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Minimal synthetic ratings DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_reviews(spark):
    """
    10 reviews across 3 users and 4 businesses.
    Stars values span 1-5 so the ALS loss surface is non-trivial.
    """
    data = [
        {"user_id": "u1", "business_id": "b1", "stars": 5.0},
        {"user_id": "u1", "business_id": "b2", "stars": 3.0},
        {"user_id": "u1", "business_id": "b3", "stars": 4.0},
        {"user_id": "u2", "business_id": "b1", "stars": 2.0},
        {"user_id": "u2", "business_id": "b4", "stars": 5.0},
        {"user_id": "u2", "business_id": "b2", "stars": 1.0},
        {"user_id": "u3", "business_id": "b3", "stars": 4.0},
        {"user_id": "u3", "business_id": "b4", "stars": 3.0},
        {"user_id": "u3", "business_id": "b1", "stars": 5.0},
        {"user_id": "u3", "business_id": "b2", "stars": 2.0},
    ]
    return spark.createDataFrame(data)


# ---------------------------------------------------------------------------
# build_user_item_matrix
# ---------------------------------------------------------------------------

class TestBuildUserItemMatrix:
    def test_output_columns(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        assert set(matrix_df.columns) == {"user_idx", "business_idx", "stars"}

    def test_integer_indices(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        row = matrix_df.first()
        assert isinstance(row.user_idx, int)
        assert isinstance(row.business_idx, int)

    def test_user_map_covers_all_users(self, sample_reviews):
        _, user_map, _ = build_user_item_matrix(sample_reviews)
        assert set(user_map.values()) == {"u1", "u2", "u3"}

    def test_biz_map_covers_all_businesses(self, sample_reviews):
        _, _, biz_map = build_user_item_matrix(sample_reviews)
        assert set(biz_map.values()) == {"b1", "b2", "b3", "b4"}

    def test_row_count_preserved(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        assert matrix_df.count() == sample_reviews.count()


# ---------------------------------------------------------------------------
# train_als
# ---------------------------------------------------------------------------

class TestTrainAls:
    def test_model_trains_without_error(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        # Small rank to keep the test fast
        model = train_als(matrix_df, rank=5, max_iter=3, reg_param=0.1)
        assert model is not None

    def test_model_has_user_factors(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        # userFactors is a DataFrame of (id, features) pairs
        assert model.userFactors.count() == 3   # 3 unique users

    def test_model_has_item_factors(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        assert model.itemFactors.count() == 4   # 4 unique businesses


# ---------------------------------------------------------------------------
# evaluate_rmse
# ---------------------------------------------------------------------------

class TestEvaluateRmse:
    def test_rmse_is_positive_float(self, sample_reviews):
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        # Use the full dataset as both train and test to guarantee predictions exist
        rmse = evaluate_rmse(model, matrix_df)
        assert isinstance(rmse, float)
        assert rmse > 0.0

    def test_rmse_reasonable_range(self, sample_reviews):
        """
        With rank=10 and 10 training examples, the model should overfit
        enough to yield RMSE < 2.0 on the training set itself.
        """
        matrix_df, _, _ = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=10, max_iter=15)
        rmse = evaluate_rmse(model, matrix_df)
        assert rmse < 2.0


# ---------------------------------------------------------------------------
# generate_recommendations
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    def test_returns_dataframe(self, sample_reviews):
        import pandas as pd
        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=2)
        assert isinstance(recs, pd.DataFrame)

    def test_expected_columns(self, sample_reviews):
        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=2)
        assert set(recs.columns) == {
            "user_id", "business_id", "predicted_rating", "rank"
        }

    def test_rank_is_1_based(self, sample_reviews):
        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=3)
        assert recs["rank"].min() == 1

    def test_top_n_respected(self, sample_reviews):
        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=2)
        # Each user should have at most 2 recommendations
        per_user = recs.groupby("user_id")["rank"].count()
        assert (per_user <= 2).all()

    def test_user_ids_are_strings(self, sample_reviews):
        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=2)
        assert recs["user_id"].dtype == object  # pandas string dtype


# ---------------------------------------------------------------------------
# get_db_url
# ---------------------------------------------------------------------------

class TestGetDbUrl:
    def test_local_mode_returns_sqlite(self):
        url = get_db_url("local")
        assert url.startswith("sqlite:///")
        assert "citybite_local.db" in url

    def test_emr_mode_returns_postgres(self, monkeypatch):
        monkeypatch.setenv("RDS_HOST", "my-host.rds.amazonaws.com")
        monkeypatch.setenv("RDS_PORT", "5432")
        monkeypatch.setenv("RDS_DB", "citybite")
        monkeypatch.setenv("RDS_USER", "admin")
        monkeypatch.setenv("RDS_PASSWORD", "secret")
        url = get_db_url("emr")
        assert url.startswith("postgresql+psycopg2://")
        assert "my-host.rds.amazonaws.com" in url

    def test_emr_mode_missing_rds_host_raises(self, monkeypatch):
        monkeypatch.delenv("RDS_HOST", raising=False)
        with pytest.raises(KeyError):
            get_db_url("emr")


# ---------------------------------------------------------------------------
# write_recommendations
# ---------------------------------------------------------------------------

class TestWriteRecommendations:
    def test_sqlite_roundtrip(self, sample_reviews, tmp_path):
        """Write recs to a temp SQLite DB and read them back."""
        import pandas as pd
        from sqlalchemy import create_engine, text

        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=2)

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        write_recommendations(recs, db_url)

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = pd.read_sql(text("SELECT * FROM als_recommendations"), conn)

        assert len(result) == len(recs)
        assert set(result.columns) >= {"user_id", "business_id", "predicted_rating", "rank"}

    def test_no_nan_predicted_rating(self, sample_reviews, tmp_path):
        """All written predicted_rating values must be finite (not NaN)."""
        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=3)
        assert not recs["predicted_rating"].isna().any()

    def test_overwrite_on_rerun(self, sample_reviews, tmp_path):
        """Calling write_recommendations twice should not duplicate rows."""
        import pandas as pd
        from sqlalchemy import create_engine, text

        matrix_df, user_map, biz_map = build_user_item_matrix(sample_reviews)
        model = train_als(matrix_df, rank=5, max_iter=3)
        recs = generate_recommendations(model, biz_map, user_map, top_n=2)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        write_recommendations(recs, db_url)
        write_recommendations(recs, db_url)

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = pd.read_sql(text("SELECT * FROM als_recommendations"), conn)

        assert len(result) == len(recs)

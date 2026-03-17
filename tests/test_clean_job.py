"""
Unit tests for pipeline/clean_job.py transformations.

Runs entirely in PySpark local mode — no AWS required.
"""

import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pipeline.clean_job import (
    add_grid_cell,
    add_recency_weight,
    build_enriched,
    clean_businesses,
    clean_reviews,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .appName("CityBite-Test")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def sample_businesses(spark):
    return spark.createDataFrame([
        {"business_id": "b1", "name": "Taco Spot",   "city": "Phoenix", "state": "AZ",
         "latitude": 33.45, "longitude": -112.07, "is_open": 1, "categories": "Mexican"},
        {"business_id": "b2", "name": "Closed Cafe", "city": "Phoenix", "state": "AZ",
         "latitude": 33.46, "longitude": -112.08, "is_open": 0, "categories": "Coffee"},
        {"business_id": "b3", "name": "Pizza Place", "city": "Las Vegas", "state": "NV",
         "latitude": 36.17, "longitude": -115.14, "is_open": 1, "categories": "Pizza"},
        # missing lat/lng → should be dropped
        {"business_id": "b4", "name": "No Coords", "city": "Phoenix", "state": "AZ",
         "latitude": None, "longitude": None, "is_open": 1, "categories": "Unknown"},
    ])


@pytest.fixture
def sample_reviews(spark):
    recent = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    old = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S")
    return spark.createDataFrame([
        {"review_id": "r1", "business_id": "b1", "user_id": "u1", "stars": 5, "date": recent, "text": "Great!"},
        {"review_id": "r2", "business_id": "b1", "user_id": "u2", "stars": 3, "date": old,    "text": "Okay."},
        {"review_id": "r3", "business_id": "b3", "user_id": "u1", "stars": 4, "date": recent, "text": "Good."},
        # null text → should be dropped
        {"review_id": "r4", "business_id": "b1", "user_id": "u3", "stars": 4, "date": recent, "text": None},
        # closed business (b2) → no match after join
        {"review_id": "r5", "business_id": "b2", "user_id": "u1", "stars": 2, "date": recent, "text": "Meh."},
    ])


class TestCleanBusinesses:
    def test_filters_closed(self, sample_businesses):
        result = clean_businesses(sample_businesses)
        ids = {r.business_id for r in result.collect()}
        assert "b2" not in ids

    def test_drops_null_coords(self, sample_businesses):
        result = clean_businesses(sample_businesses)
        ids = {r.business_id for r in result.collect()}
        assert "b4" not in ids

    def test_keeps_open_businesses(self, sample_businesses):
        result = clean_businesses(sample_businesses)
        ids = {r.business_id for r in result.collect()}
        assert {"b1", "b3"}.issubset(ids)


class TestCleanReviews:
    def test_drops_null_text(self, sample_reviews):
        result = clean_reviews(sample_reviews)
        ids = {r.review_id for r in result.collect()}
        assert "r4" not in ids

    def test_keeps_valid_reviews(self, sample_reviews):
        result = clean_reviews(sample_reviews)
        ids = {r.review_id for r in result.collect()}
        assert {"r1", "r2", "r3"}.issubset(ids)


class TestAddRecencyWeight:
    def test_recent_review_higher_weight(self, spark):
        recent = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        old = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d %H:%M:%S")
        df = spark.createDataFrame([
            {"review_id": "r1", "date": recent},
            {"review_id": "r2", "date": old},
        ])
        result = add_recency_weight(df)
        rows = {r.review_id: r.recency_weight for r in result.collect()}
        assert rows["r1"] > rows["r2"]

    def test_weight_between_0_and_1(self, spark):
        old = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d %H:%M:%S")
        df = spark.createDataFrame([{"review_id": "r1", "date": old}])
        result = add_recency_weight(df)
        weight = result.collect()[0].recency_weight
        assert 0.0 < weight <= 1.0


class TestAddGridCell:
    def test_grid_cell_format(self, spark):
        df = spark.createDataFrame([{"latitude": 33.45, "longitude": -112.07}])
        result = add_grid_cell(df)
        cell = result.collect()[0].grid_cell
        # Should be two decimal-formatted numbers joined by "_"
        parts = cell.split("_")
        assert len(parts) == 2
        float(parts[0])  # must be parseable as float
        float(parts[1])

    def test_same_grid_for_nearby_points(self, spark):
        df = spark.createDataFrame([
            {"id": 1, "latitude": 33.41, "longitude": -112.01},
            {"id": 2, "latitude": 33.49, "longitude": -112.09},
        ])
        result = add_grid_cell(df)
        cells = [r.grid_cell for r in result.collect()]
        assert cells[0] == cells[1]


class TestBuildEnriched:
    def test_enriched_has_required_columns(self, sample_businesses, sample_reviews):
        businesses = clean_businesses(sample_businesses)
        reviews = clean_reviews(sample_reviews)
        enriched = build_enriched(reviews, businesses)
        cols = set(enriched.columns)
        assert {"review_id", "business_id", "city", "recency_weight", "grid_cell"}.issubset(cols)

    def test_closed_businesses_excluded(self, sample_businesses, sample_reviews):
        businesses = clean_businesses(sample_businesses)
        reviews = clean_reviews(sample_reviews)
        enriched = build_enriched(reviews, businesses)
        # b2 is closed → r5 must not appear
        ids = {r.review_id for r in enriched.collect()}
        assert "r5" not in ids

    def test_row_count(self, sample_businesses, sample_reviews):
        businesses = clean_businesses(sample_businesses)
        reviews = clean_reviews(sample_reviews)
        enriched = build_enriched(reviews, businesses)
        # r1, r2 (b1 open Phoenix) + r3 (b3 open LV) = 3 rows
        assert enriched.count() == 3

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
    build_metro_map,
    clean_businesses,
    clean_reviews,
    haversine_km,
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
        {"business_id": "b1", "name": "Taco Spot",    "city": "Phoenix",   "state": "AZ",
         "latitude": 33.45, "longitude": -112.07, "is_open": 1,
         "categories": "Mexican, Restaurants", "review_count": 120},
        {"business_id": "b2", "name": "Closed Cafe",  "city": "Phoenix",   "state": "AZ",
         "latitude": 33.46, "longitude": -112.08, "is_open": 0,
         "categories": "Coffee, Restaurants", "review_count": 50},
        {"business_id": "b3", "name": "Pizza Place",  "city": "Las Vegas", "state": "NV",
         "latitude": 36.17, "longitude": -115.14, "is_open": 1,
         "categories": "Pizza, Restaurants", "review_count": 80},
        # missing lat/lng → should be dropped
        {"business_id": "b4", "name": "No Coords",    "city": "Phoenix",   "state": "AZ",
         "latitude": None, "longitude": None, "is_open": 1,
         "categories": "Unknown, Restaurants", "review_count": 10},
        # non-restaurant → should be filtered
        {"business_id": "b5", "name": "Car Wash",     "city": "Phoenix",   "state": "AZ",
         "latitude": 33.47, "longitude": -112.09, "is_open": 1,
         "categories": "Auto Detailing", "review_count": 30},
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
    def test_filters_closed(self, spark, sample_businesses):
        result = clean_businesses(sample_businesses, spark)
        ids = {r.business_id for r in result.collect()}
        assert "b2" not in ids

    def test_drops_null_coords(self, spark, sample_businesses):
        result = clean_businesses(sample_businesses, spark)
        ids = {r.business_id for r in result.collect()}
        assert "b4" not in ids

    def test_filters_non_restaurants(self, spark, sample_businesses):
        result = clean_businesses(sample_businesses, spark)
        ids = {r.business_id for r in result.collect()}
        assert "b5" not in ids

    def test_keeps_open_restaurants(self, spark, sample_businesses):
        result = clean_businesses(sample_businesses, spark)
        ids = {r.business_id for r in result.collect()}
        assert {"b1", "b3"}.issubset(ids)

    def test_metro_area_column_present(self, spark, sample_businesses):
        result = clean_businesses(sample_businesses, spark)
        assert "metro_area" in result.columns

    def test_city_normalized_to_title_case(self, spark, sample_businesses):
        result = clean_businesses(sample_businesses, spark)
        cities = {r.city for r in result.collect()}
        assert all(c == c.title() or c[0].isupper() for c in cities)


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
    def test_enriched_has_required_columns(self, spark, sample_businesses, sample_reviews):
        businesses = clean_businesses(sample_businesses, spark)
        reviews = clean_reviews(sample_reviews)
        enriched = build_enriched(reviews, businesses)
        cols = set(enriched.columns)
        assert {"review_id", "business_id", "city", "metro_area",
                "recency_weight", "grid_cell"}.issubset(cols)

    def test_closed_businesses_excluded(self, spark, sample_businesses, sample_reviews):
        businesses = clean_businesses(sample_businesses, spark)
        reviews = clean_reviews(sample_reviews)
        enriched = build_enriched(reviews, businesses)
        # b2 is closed → r5 must not appear
        ids = {r.review_id for r in enriched.collect()}
        assert "r5" not in ids

    def test_row_count(self, spark, sample_businesses, sample_reviews):
        businesses = clean_businesses(sample_businesses, spark)
        reviews = clean_reviews(sample_reviews)
        enriched = build_enriched(reviews, businesses)
        # r1, r2 (b1 open Phoenix) + r3 (b3 open LV) = 3 rows
        assert enriched.count() == 3


class TestHaversine:
    def test_nashville_to_brentwood_under_50km(self):
        # Nashville TN (36.1627, -86.7816) to Brentwood TN (36.0331, -86.7828) ≈ 14 km
        dist = haversine_km(36.1627, -86.7816, 36.0331, -86.7828)
        assert dist < 50.0

    def test_portland_or_vs_me_over_50km(self):
        # Portland OR (45.5051, -122.6750) to Portland ME (43.6591, -70.2568) ≈ 4700 km
        dist = haversine_km(45.5051, -122.6750, 43.6591, -70.2568)
        assert dist > 50.0

    def test_same_point_is_zero(self):
        assert haversine_km(36.0, -86.0, 36.0, -86.0) == 0.0


class TestBuildMetroMap:
    def _make_anchors(self, rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_nearby_cities_share_metro(self):
        pd_df = self._make_anchors([
            {"city_key": "Nashville_TN", "city": "Nashville", "state": "TN",
             "center_lat": 36.1627, "center_lng": -86.7816, "total_reviews": 1000},
            {"city_key": "Brentwood_TN", "city": "Brentwood", "state": "TN",
             "center_lat": 36.0331, "center_lng": -86.7828, "total_reviews": 200},
        ])
        metro_map = build_metro_map(pd_df, radius_km=50.0)
        assert metro_map["Nashville_TN"] == metro_map["Brentwood_TN"]
        assert metro_map["Nashville_TN"] == "Nashville"

    def test_distant_cities_get_distinct_metros(self):
        pd_df = self._make_anchors([
            {"city_key": "Portland_OR", "city": "Portland", "state": "OR",
             "center_lat": 45.5051, "center_lng": -122.6750, "total_reviews": 500},
            {"city_key": "Portland_ME", "city": "Portland", "state": "ME",
             "center_lat": 43.6591, "center_lng": -70.2568,  "total_reviews": 100},
        ])
        metro_map = build_metro_map(pd_df, radius_km=50.0)
        assert metro_map["Portland_OR"] != metro_map["Portland_ME"]

    def test_anchor_is_highest_reviewed_city(self):
        pd_df = self._make_anchors([
            {"city_key": "Franklin_TN", "city": "Franklin", "state": "TN",
             "center_lat": 35.9251, "center_lng": -86.8689, "total_reviews": 150},
            {"city_key": "Nashville_TN", "city": "Nashville", "state": "TN",
             "center_lat": 36.1627, "center_lng": -86.7816, "total_reviews": 2000},
        ])
        metro_map = build_metro_map(pd_df, radius_km=50.0)
        # Nashville has more reviews → both should map to "Nashville"
        assert metro_map["Franklin_TN"] == "Nashville"
        assert metro_map["Nashville_TN"] == "Nashville"

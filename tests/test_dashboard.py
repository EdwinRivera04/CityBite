"""
Unit tests for dashboard/app.py — pure helper functions only.

Streamlit-decorated loaders and render_* functions are excluded because
they require an active Streamlit server context. The functions tested here
are plain Python with no Streamlit dependency:

  - _grid_cell_to_label   — compass-direction neighborhood naming
  - add_neighborhood_labels — label assignment + duplicate disambiguation
  - _score_color           — traffic-light color from popularity score
  - Bayesian sentiment adjustment formula (inline in render_sentiment_panel)

Streamlit is patched out at import time so the module loads cleanly in pytest.
"""

import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Patch Streamlit before importing dashboard.app
# ---------------------------------------------------------------------------

def _make_st_mock():
    """Return a MagicMock that satisfies every st.* call in app.py."""
    st = MagicMock()
    # cache decorators must return the original function unchanged
    st.cache_data = lambda *a, **kw: (lambda fn: fn)
    st.cache_resource = lambda *a, **kw: (lambda fn: fn)
    return st


_st_mock = _make_st_mock()
sys.modules.setdefault("streamlit", _st_mock)
sys.modules.setdefault("streamlit.components.v1", MagicMock())
sys.modules.setdefault("folium", MagicMock())
sys.modules.setdefault("folium.plugins", MagicMock())
sys.modules.setdefault("sqlalchemy", MagicMock())

# Now safe to import the pure helpers
from dashboard.app import (  # noqa: E402
    _HIGH_SCORE,
    _MED_SCORE,
    _grid_cell_to_label,
    _score_color,
    add_neighborhood_labels,
)


# ---------------------------------------------------------------------------
# _grid_cell_to_label
# ---------------------------------------------------------------------------

class TestGridCellToLabel:
    # Phoenix city center ≈ 33.45, -112.07

    def test_downtown_when_very_close(self):
        label = _grid_cell_to_label("33.4_-112.0", 33.45, -112.05)
        assert label == "Downtown"

    def test_north_direction(self):
        # cell center ≈ 34.15, -112.05 — clearly north of city center
        label = _grid_cell_to_label("34.1_-112.1", 33.45, -112.05)
        assert "North" in label

    def test_south_direction(self):
        label = _grid_cell_to_label("32.8_-112.1", 33.45, -112.05)
        assert "South" in label

    def test_east_direction(self):
        label = _grid_cell_to_label("33.4_-111.4", 33.45, -112.05)
        assert "East" in label

    def test_west_direction(self):
        label = _grid_cell_to_label("33.4_-112.8", 33.45, -112.05)
        assert "West" in label

    def test_suffix_district_for_nearby(self):
        # ~0.15 degrees away → "District" suffix
        label = _grid_cell_to_label("33.6_-112.0", 33.45, -112.05)
        assert any(s in label for s in ("District", "Side", "Outskirts"))

    def test_suffix_outskirts_for_far(self):
        # ~0.6 degrees away → "Outskirts" suffix
        label = _grid_cell_to_label("34.1_-112.0", 33.45, -112.05)
        assert "Outskirts" in label

    def test_invalid_grid_cell_returns_original(self):
        label = _grid_cell_to_label("not_a_valid_cell", 33.45, -112.05)
        assert label == "not_a_valid_cell"

    def test_returns_string(self):
        label = _grid_cell_to_label("33.5_-112.0", 33.45, -112.05)
        assert isinstance(label, str)
        assert len(label) > 0


# ---------------------------------------------------------------------------
# _score_color
# ---------------------------------------------------------------------------

class TestScoreColor:
    def test_high_score_is_deep_red(self):
        assert _score_color(1.0) == "#c62828"
        assert _score_color(_HIGH_SCORE) == "#c62828"

    def test_just_below_high_is_amber(self):
        assert _score_color(_HIGH_SCORE - 0.01) == "#ef6c00"

    def test_medium_score_is_amber(self):
        assert _score_color(_MED_SCORE) == "#ef6c00"
        assert _score_color(0.55) == "#ef6c00"

    def test_just_below_medium_is_green(self):
        assert _score_color(_MED_SCORE - 0.01) == "#2e7d32"

    def test_zero_score_is_green(self):
        assert _score_color(0.0) == "#2e7d32"

    def test_returns_hex_string(self):
        for score in [0.0, 0.3, 0.5, 0.8, 1.0]:
            color = _score_color(score)
            assert color.startswith("#")
            assert len(color) == 7


# ---------------------------------------------------------------------------
# add_neighborhood_labels
# ---------------------------------------------------------------------------

class TestAddNeighborhoodLabels:
    def _grid_df(self, cells: list[tuple[str, float, float]]) -> pd.DataFrame:
        """Build a minimal grid_df from (grid_cell, lat, lng) tuples."""
        return pd.DataFrame(cells, columns=["grid_cell", "center_lat", "center_lng"])

    def test_adds_neighborhood_column(self):
        df = self._grid_df([("33.4_-112.0", 33.45, -112.05)])
        result = add_neighborhood_labels(df)
        assert "neighborhood" in result.columns

    def test_row_count_unchanged(self):
        df = self._grid_df([
            ("33.4_-112.0", 33.45, -112.05),
            ("34.0_-112.0", 34.05, -112.05),
            ("32.9_-112.0", 32.95, -112.05),
        ])
        result = add_neighborhood_labels(df)
        assert len(result) == len(df)

    def test_unique_labels_when_no_duplicates(self):
        df = self._grid_df([
            ("33.4_-112.0", 33.45, -112.05),   # Downtown
            ("34.2_-112.0", 34.25, -112.05),   # North Outskirts
            ("32.7_-112.0", 32.75, -112.05),   # South Outskirts
        ])
        result = add_neighborhood_labels(df)
        assert result["neighborhood"].nunique() == len(result)

    def test_duplicate_labels_disambiguated_with_numbers(self):
        # Two cells that map to the same compass label
        df = self._grid_df([
            ("34.0_-112.0", 34.05, -112.05),
            ("34.1_-112.0", 34.15, -112.05),
        ])
        result = add_neighborhood_labels(df)
        labels = result["neighborhood"].tolist()
        # Both must be distinct
        assert labels[0] != labels[1]
        # At least one should contain a number (disambiguation suffix)
        assert any(any(c.isdigit() for c in lbl) for lbl in labels)

    def test_empty_dataframe_returned_unchanged(self):
        result = add_neighborhood_labels(pd.DataFrame())
        assert result.empty

    def test_missing_grid_cell_column_returned_unchanged(self):
        df = pd.DataFrame({"center_lat": [33.45], "center_lng": [-112.05]})
        result = add_neighborhood_labels(df)
        assert "neighborhood" not in result.columns

    def test_original_columns_preserved(self):
        df = self._grid_df([("33.4_-112.0", 33.45, -112.05)])
        df["extra_col"] = 42
        result = add_neighborhood_labels(df)
        assert "extra_col" in result.columns
        assert "grid_cell" in result.columns


# ---------------------------------------------------------------------------
# Bayesian sentiment adjustment
# ---------------------------------------------------------------------------

class TestBayesianSentimentAdjustment:
    """
    Tests for the formula used in render_sentiment_panel:

        adjusted = (positive_count + k * global_rate) / (positive_count + negative_count + k)

    k=30 is the project default.
    """

    def _adjust(self, pos: int, neg: int, global_rate: float, k: int = 30) -> float:
        return (pos + k * global_rate) / (pos + neg + k)

    def test_zero_reviews_equals_global_rate(self):
        global_rate = 0.65
        score = self._adjust(0, 0, global_rate)
        assert score == pytest.approx(global_rate)

    def test_low_volume_all_positive_pulled_down(self):
        """3 five-star reviews should not yield a score near 1.0."""
        score = self._adjust(3, 0, global_rate=0.6)
        assert score < 0.9

    def test_low_volume_all_negative_pulled_up(self):
        """3 one-star reviews should not yield a score near 0.0."""
        score = self._adjust(0, 3, global_rate=0.6)
        assert score > 0.3

    def test_high_volume_close_to_raw_rate(self):
        """1000 reviews at 80% positive should be close to 0.8."""
        score = self._adjust(800, 200, global_rate=0.6)
        assert abs(score - 0.8) < 0.02

    def test_score_always_in_range(self):
        for pos, neg, rate in [
            (0, 0, 0.5), (1, 0, 0.9), (0, 1, 0.1),
            (100, 0, 0.5), (0, 100, 0.5), (50, 50, 0.5),
        ]:
            score = self._adjust(pos, neg, rate)
            assert 0.0 <= score <= 1.0

    def test_larger_k_pulls_more_toward_global(self):
        """Higher k = stronger prior = score closer to global_rate."""
        pos, neg, global_rate = 5, 0, 0.5
        score_k10 = self._adjust(pos, neg, global_rate, k=10)
        score_k50 = self._adjust(pos, neg, global_rate, k=50)
        # Both above global_rate (positive-heavy), but k=50 should be closer to it
        assert abs(score_k50 - global_rate) < abs(score_k10 - global_rate)

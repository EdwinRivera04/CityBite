"""
CityBite Streamlit Dashboard — Folium heatmap + ALS recommendations.

Layout:
  Sidebar  — city selector, cuisine filter, user ID for personalized recs
  Main     — Folium map: one CircleMarker per grid cell, radius ∝ popularity
  Right    — Top-10 ALS recommendations for the selected user
  Bottom   — Neighborhood sentiment bar chart (grid_sentiment table)

Local dev:  reads from data/citybite_local.db (SQLite)
Production: reads from RDS PostgreSQL via env vars (RDS_HOST, etc.)

Run:
    streamlit run dashboard/app.py
"""

import os

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from streamlit_folium import st_folium

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CityBite",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_LOCAL_DB = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")


@st.cache_resource
def get_engine():
    """
    Return a SQLAlchemy engine.

    Falls back to local SQLite if RDS_HOST is not set, so the app works
    out-of-the-box in local dev without any AWS credentials.
    """
    rds_host = os.environ.get("RDS_HOST")
    if rds_host:
        user = os.environ["RDS_USER"]
        pw = os.environ["RDS_PASSWORD"]
        port = os.environ.get("RDS_PORT", "5432")
        db = os.environ["RDS_DB"]
        url = f"postgresql+psycopg2://{user}:{pw}@{rds_host}:{port}/{db}"
    else:
        url = f"sqlite:///{_LOCAL_DB}"

    return create_engine(url)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading grid data ...")
def load_grid_data(city: str) -> pd.DataFrame:
    """Load grid_aggregates for the selected city from the database."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM grid_aggregates WHERE city = :city"),
            conn,
            params={"city": city},
        )
    return df


@st.cache_data(ttl=3600, show_spinner="Loading city list ...")
def load_cities() -> list[str]:
    """Return the sorted list of cities present in business_scores."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT DISTINCT city FROM business_scores ORDER BY city"),
            conn,
        )
    return df["city"].tolist()


@st.cache_data(ttl=3600, show_spinner="Loading businesses ...")
def load_businesses(city: str, cuisine_filter: str | None) -> pd.DataFrame:
    """
    Load business_scores for a city, optionally filtered by cuisine keyword.

    The categories column is free-text (e.g. "Mexican, Restaurants") so we
    use a LIKE/ILIKE search rather than an exact match.
    """
    engine = get_engine()
    # Build query — SQLite uses LIKE (case-insensitive by default for ASCII)
    if cuisine_filter and cuisine_filter != "All":
        query = text(
            "SELECT * FROM business_scores "
            "WHERE city = :city AND categories LIKE :cat "
            "ORDER BY popularity_score DESC"
        )
        params = {"city": city, "cat": f"%{cuisine_filter}%"}
    else:
        query = text(
            "SELECT * FROM business_scores WHERE city = :city "
            "ORDER BY popularity_score DESC"
        )
        params = {"city": city}

    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


@st.cache_data(ttl=3600, show_spinner="Loading recommendations ...")
def load_recommendations(user_id: str) -> pd.DataFrame:
    """
    Load top-10 ALS recommendations for a specific user.

    Joins with business_scores to surface name, city, and cuisine alongside
    the predicted rating — avoids a second round-trip in the UI layer.
    """
    engine = get_engine()
    query = text(
        """
        SELECT r.rank, r.predicted_rating,
               b.name, b.city, b.categories, b.avg_rating, b.review_count
        FROM   als_recommendations r
        JOIN   business_scores     b ON r.business_id = b.business_id
        WHERE  r.user_id = :uid
        ORDER  BY r.rank
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"uid": user_id})


@st.cache_data(ttl=3600, show_spinner="Loading sentiment ...")
def load_sentiment(city: str) -> pd.DataFrame:
    """
    Load grid_sentiment joined with grid_aggregates so we can label each
    grid cell by its city and sort by sentiment score.
    """
    engine = get_engine()
    query = text(
        """
        SELECT s.grid_cell, s.sentiment_score, s.positive_count, s.negative_count,
               g.center_lat, g.center_lng, g.restaurant_count
        FROM   grid_sentiment  s
        JOIN   grid_aggregates g ON s.grid_cell = g.grid_cell
        WHERE  g.city = :city
        ORDER  BY s.sentiment_score DESC
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"city": city})


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

# Color thresholds for circle marker fill
_HIGH_SCORE = 0.7
_MED_SCORE = 0.4


def _popularity_color(score: float) -> str:
    """Map a normalized popularity score to a traffic-light color."""
    if score >= _HIGH_SCORE:
        return "#e53935"    # red   — hotspot
    if score >= _MED_SCORE:
        return "#fb8c00"    # orange — moderate
    return "#43a047"        # green  — low activity


def build_heatmap(grid_df: pd.DataFrame) -> folium.Map:
    """
    Build a Folium map with one CircleMarker per grid cell.

    Radius scales with avg_popularity (capped so extreme values don't
    swamp the map).  Color encodes the same score in three bands.
    """
    center_lat = grid_df["center_lat"].mean()
    center_lng = grid_df["center_lng"].mean()

    # Use explicit HTTPS tile URLs and a provider fallback to avoid blank
    # basemaps when one tile source is blocked or unavailable.
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; OpenStreetMap contributors &copy; CARTO',
        name="CartoDB Positron",
        max_zoom=20,
        subdomains="abcd",
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; OpenStreetMap contributors',
        name="OpenStreetMap",
        max_zoom=19,
        show=False,
    ).add_to(m)

    # Normalize scores to [0, 1] relative to this city's range so radius
    # is visually consistent regardless of absolute score values
    max_score = grid_df["avg_popularity"].max() or 1.0
    min_score = grid_df["avg_popularity"].min()
    score_range = max(max_score - min_score, 1e-6)

    for _, row in grid_df.iterrows():
        norm = (row["avg_popularity"] - min_score) / score_range
        radius = 6 + norm * 20      # radius in pixels: 6 (low) → 26 (high)
        color = _popularity_color(norm)

        popup_html = (
            f"<b>{row['top_cuisine']}</b><br>"
            f"Popularity: {row['avg_popularity']:.2f}<br>"
            f"Restaurants: {row['restaurant_count']}<br>"
            f"Grid: {row['grid_cell']}"
        )

        folium.CircleMarker(
            location=[row["center_lat"], row["center_lng"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['top_cuisine']} | Score: {row['avg_popularity']:.2f}",
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    return m


# ---------------------------------------------------------------------------
# UI — sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, str, str]:
    """
    Render sidebar controls and return
    (selected_city, cuisine_filter, user_id, map_renderer).
    """
    st.sidebar.title("CityBite")
    st.sidebar.caption("Restaurant popularity intelligence")
    st.sidebar.markdown("---")

    cities = load_cities()
    if not cities:
        st.sidebar.error("No cities found in database. Run seed_local_db.py first.")
        st.stop()

    selected_city = st.sidebar.selectbox("City", cities, index=0)

    # Cuisine filter — derive options from business data for the city
    biz_df = load_businesses(selected_city, None)
    all_cuisines = sorted(
        {
            tag.strip()
            for cats in biz_df["categories"].dropna()
            for tag in cats.split(",")
        }
    )
    cuisine_options = ["All"] + all_cuisines
    cuisine_filter = st.sidebar.selectbox("Cuisine", cuisine_options, index=0)

    st.sidebar.markdown("---")
    user_id = st.sidebar.text_input(
        "User ID (for personalized recs)",
        placeholder="Enter a Yelp user_id ...",
    )

    st.sidebar.markdown("---")
    map_renderer = st.sidebar.radio(
        "Map renderer",
        options=["Auto", "Raw HTML fallback"],
        index=0,
        help="Use Raw HTML fallback if map markers show but basemap tiles are blank.",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Heatmap colors:\n"
        "- 🔴 High popularity\n"
        "- 🟠 Moderate\n"
        "- 🟢 Low activity"
    )

    return selected_city, cuisine_filter, user_id, map_renderer


# ---------------------------------------------------------------------------
# UI — main content
# ---------------------------------------------------------------------------

def render_map_panel(city: str, cuisine_filter: str, map_renderer: str) -> None:
    """Render the Folium map in the main content area."""
    grid_df = load_grid_data(city)

    if grid_df.empty:
        st.warning(f"No grid data for {city}. Run seed_local_db.py to populate the DB.")
        return

    st.subheader(f"Restaurant Popularity Map — {city}")
    m = build_heatmap(grid_df)
    # Primary renderer is st_folium. A manual fallback is provided for
    # environments where the custom component sandbox blocks tile loading.
    if map_renderer == "Raw HTML fallback":
        components.html(m._repr_html_(), height=520, scrolling=False)
    else:
        try:
            st_folium(m, width=None, height=500, use_container_width=True, returned_objects=[])
        except Exception:
            components.html(m._repr_html_(), height=520, scrolling=False)

    # Summary metrics below the map
    col1, col2, col3 = st.columns(3)
    col1.metric("Grid cells", len(grid_df))
    col2.metric("Total restaurants", int(grid_df["restaurant_count"].sum()))
    col3.metric(
        "Avg popularity",
        f"{grid_df['avg_popularity'].mean():.2f}",
    )

    # Filtered business table
    with st.expander(f"Top businesses in {city} — {cuisine_filter}", expanded=False):
        biz_df = load_businesses(city, cuisine_filter)
        if biz_df.empty:
            st.info("No businesses match the current filter.")
        else:
            display_cols = [
                "name", "categories", "avg_rating",
                "review_count", "popularity_score", "grid_cell",
            ]
            st.dataframe(
                biz_df[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )


def render_recommendations_panel(user_id: str) -> None:
    """Render personalized ALS recommendations in the right column."""
    st.subheader("Personalized Recommendations")

    if not user_id:
        st.info("Enter a User ID in the sidebar to see your top-10 picks.")
        return

    recs_df = load_recommendations(user_id)

    if recs_df.empty:
        st.warning(
            f"No recommendations found for user `{user_id}`. "
            "Run `spark-submit ml/als_train.py` to generate them."
        )
        return

    for _, row in recs_df.iterrows():
        with st.container():
            st.markdown(
                f"**{row['rank']}. {row['name']}** &nbsp; "
                f"{'⭐' * round(row['avg_rating'])} "
                f"({row['avg_rating']:.1f} avg, {row['review_count']:,} reviews)"
            )
            st.caption(
                f"{row['categories']} · {row['city']} · "
                f"Predicted: {row['predicted_rating']:.2f} ★"
            )
            st.markdown("---")


def render_sentiment_panel(city: str) -> None:
    """Render per-neighborhood sentiment bar chart at the bottom of the page."""
    sentiment_df = load_sentiment(city)

    if sentiment_df.empty:
        st.info(
            "No sentiment data yet. Run `spark-submit ml/sentiment.py` to populate."
        )
        return

    st.subheader(f"Neighborhood Sentiment — {city}")

    # Bar chart using Streamlit's built-in chart (no extra dependencies)
    chart_df = (
        sentiment_df
        .set_index("grid_cell")[["sentiment_score"]]
        .sort_values("sentiment_score", ascending=False)
        .head(20)           # cap at 20 cells so the chart is readable
    )
    st.bar_chart(chart_df, height=300)

    with st.expander("Sentiment detail table", expanded=False):
        st.dataframe(
            sentiment_df[
                ["grid_cell", "sentiment_score", "positive_count",
                 "negative_count", "restaurant_count"]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    selected_city, cuisine_filter, user_id, map_renderer = render_sidebar()

    # Two-column layout: map (wider) | recommendations
    left_col, right_col = st.columns([3, 1])

    with left_col:
        render_map_panel(selected_city, cuisine_filter, map_renderer)

    with right_col:
        render_recommendations_panel(user_id)

    st.markdown("---")
    render_sentiment_panel(selected_city)


if __name__ == "__main__":
    main()

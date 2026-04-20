"""
CityBite Streamlit Dashboard — Folium heatmap + restaurant pins + ALS recommendations.

Layout:
  Sidebar  — city selector, cuisine filter, user ID
  Left     — Folium map: rectangle tile per grid cell + recommendation pins
  Right    — Top-10 ALS recommendations for the selected user
  Bottom   — Neighborhood sentiment bar chart

Local dev:  reads from data/citybite_local.db (SQLite)
Production: reads from RDS PostgreSQL via env vars (RDS_HOST, etc.)

Run:
    streamlit run dashboard/app.py
"""

import math
import os
from collections import Counter

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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

st.markdown("""
<style>
    .block-container { padding-top: 0.5rem; padding-bottom: 0; }
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stSidebar"] > div { padding-top: 0.5rem; padding-bottom: 0; }
    div[data-testid="stSidebar"] section { padding-top: 0 !important; }
    div[data-testid="stSidebarContent"] { gap: 0; }
    div[data-testid="stSidebar"] hr { margin: 0.4rem 0; }
    div[data-testid="stSidebar"] h1 { margin-bottom: 0; padding-bottom: 0; }
    div[data-testid="stSidebar"] .stSelectbox { margin-bottom: 0; }
    div[data-testid="stSidebar"] .stTextInput { margin-bottom: 0; }
    div[data-testid="stSidebar"] .stSlider { margin-bottom: 0; }
    div[data-testid="stSidebar"] p { margin-bottom: 0.2rem; }
    div[data-testid="stSidebar"] table { margin-top: 0.2rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding-left: 24px; padding-right: 24px; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_LOCAL_DB = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")


@st.cache_resource
def get_engine():
    rds_host = os.environ.get("RDS_HOST", "").strip()
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_cities() -> list[str]:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT DISTINCT city FROM business_scores ORDER BY city"), conn
            )
        return df["city"].tolist()
    except Exception as e:
        st.error(f"Could not load cities: {e}. Make sure the pipeline has run.")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def load_grid_data(city: str) -> pd.DataFrame:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM grid_aggregates WHERE city = :city"),
                conn, params={"city": city},
            )
        return df
    except Exception as e:
        st.error(f"Could not load grid data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_businesses(city: str, cuisine_filter: str | None) -> pd.DataFrame:
    engine = get_engine()
    if cuisine_filter and cuisine_filter != "All":
        query = text(
            "SELECT * FROM business_scores "
            "WHERE city = :city AND LOWER(categories) LIKE LOWER(:cat) "
            "ORDER BY avg_rating DESC, review_count DESC"
        )
        params = {"city": city, "cat": f"%{cuisine_filter}%"}
    else:
        query = text(
            "SELECT * FROM business_scores WHERE city = :city "
            "ORDER BY avg_rating DESC, review_count DESC"
        )
        params = {"city": city}
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)
    except Exception as e:
        st.error(f"Could not load businesses: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_recommendations(user_id: str, city: str) -> tuple[pd.DataFrame, bool]:
    """
    Load top-10 ALS recommendations, preferring the selected city.

    Returns (df, city_filtered) where city_filtered=True means the results
    are specific to the selected city. Falls back to cross-city results if
    no recommendations exist for the user in that city.
    """
    engine = get_engine()
    base_select = """
        SELECT r.rank, r.predicted_rating,
               b.name, b.city, b.categories, b.avg_rating, b.review_count,
               b.latitude, b.longitude
        FROM   als_recommendations r
        JOIN   business_scores b ON r.business_id = b.business_id
        WHERE  r.user_id = :uid
    """
    city_query = text(base_select + " AND b.city = :city ORDER BY r.rank")
    all_query  = text(base_select + " ORDER BY r.rank LIMIT 10")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(city_query, conn, params={"uid": user_id, "city": city})
            if not df.empty:
                return df, True
            df = pd.read_sql(all_query, conn, params={"uid": user_id})
            return df, False
    except Exception as e:
        st.error(f"Could not load recommendations: {e}")
        return pd.DataFrame(), False


@st.cache_data(ttl=3600, show_spinner=False)
def load_sentiment(city: str) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT g.grid_cell, g.center_lat, g.center_lng, g.restaurant_count,
               COALESCE(s.sentiment_score, 0) AS sentiment_score,
               COALESCE(s.positive_count, 0)  AS positive_count,
               COALESCE(s.negative_count, 0)  AS negative_count
        FROM   grid_aggregates g
        LEFT JOIN grid_sentiment s ON s.grid_cell = g.grid_cell
        WHERE  g.city = :city
        ORDER  BY sentiment_score DESC
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"city": city})
    except Exception as e:
        st.error(f"Could not load sentiment data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_cuisines_for_city(city: str) -> list[str]:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT categories FROM business_scores WHERE city = :city"),
                conn, params={"city": city},
            )
        return sorted({
            tag.strip()
            for cats in df["categories"].dropna()
            for tag in cats.split(",")
            if tag.strip()
        })
    except Exception:
        return []
        import streamlit as st


# ---------------------------------------------------------------------------
# Neighborhood naming
# ---------------------------------------------------------------------------

def _grid_cell_to_label(grid_cell: str, city_lat: float, city_lng: float) -> str:
    """
    Convert a grid_cell key like '35.2_-115.0' to a compass-direction
    neighborhood label (e.g. 'North District', 'Downtown', 'Southeast Side').
    """
    try:
        lat_s, lng_s = grid_cell.split("_")
        lat = float(lat_s) + 0.05   # shift to cell center
        lng = float(lng_s) + 0.05
    except ValueError:
        return grid_cell

    dlat = lat - city_lat
    dlng = lng - city_lng
    dist = math.sqrt(dlat ** 2 + dlng ** 2)

    if dist < 0.1:
        return "Downtown"

    # atan2(dlat, dlng) gives angle from East; convert to bearing from North
    bearing = (90 - math.degrees(math.atan2(dlat, dlng))) % 360
    if bearing < 22.5 or bearing >= 337.5:
        direction = "North"
    elif bearing < 67.5:
        direction = "Northeast"
    elif bearing < 112.5:
        direction = "East"
    elif bearing < 157.5:
        direction = "Southeast"
    elif bearing < 202.5:
        direction = "South"
    elif bearing < 247.5:
        direction = "Southwest"
    elif bearing < 292.5:
        direction = "West"
    else:
        direction = "Northwest"

    if dist < 0.25:
        suffix = "District"
    elif dist < 0.45:
        suffix = "Side"
    else:
        suffix = "Outskirts"

    return f"{direction} {suffix}"


def add_neighborhood_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'neighborhood' column to any DataFrame that has
    grid_cell, center_lat, and center_lng columns. Disambiguates duplicate
    labels by appending a number.
    """
    if df.empty or "grid_cell" not in df.columns:
        return df

    city_lat = df["center_lat"].mean()
    city_lng = df["center_lng"].mean()

    raw = [_grid_cell_to_label(gc, city_lat, city_lng) for gc in df["grid_cell"]]

    seen_count: Counter = Counter(raw)
    seen_index: Counter = Counter()
    labels = []
    for label in raw:
        if seen_count[label] > 1:
            seen_index[label] += 1
            labels.append(f"{label} {seen_index[label]}")
        else:
            labels.append(label)

    out = df.copy()
    out["neighborhood"] = labels
    return out


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

_HIGH_SCORE = 0.7
_MED_SCORE  = 0.4


def _score_color(norm: float) -> str:
    """Traffic-light color from a normalized [0,1] popularity score."""
    if norm >= _HIGH_SCORE:
        return "#c62828"   # deep red — hotspot
    if norm >= _MED_SCORE:
        return "#ef6c00"   # amber — moderate
    return "#2e7d32"       # green — quiet


def build_map(
    grid_df: pd.DataFrame,
    businesses_df: pd.DataFrame | None = None,
    recs_df: pd.DataFrame | None = None,
) -> folium.Map:
    """
    Build a Folium map.

    Popularity is shown as a smooth HeatMap gradient (blue -> yellow -> red).
    Invisible rectangles sit on top to provide hover tooltips and click popups
    without interfering with the visual layer.
    When businesses_df is provided, top restaurant pins are added for the
    currently selected city and cuisine filter.
    When recs_df is provided, numbered blue pin markers are added at each
    recommended business location.
    """
    from folium.plugins import HeatMap, MarkerCluster

    center_lat = grid_df["center_lat"].mean()
    center_lng = grid_df["center_lng"].mean()

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=11,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">'
            'OpenStreetMap</a> contributors &copy; '
            '<a href="https://carto.com/attributions">CARTO</a>'
        ),
        control_scale=True,
    )
    max_score = grid_df["avg_popularity"].max() or 1.0
    min_score = grid_df["avg_popularity"].min()
    score_range = max(max_score - min_score, 1e-6)

    grid_df = add_neighborhood_labels(grid_df)

    heat_data = [
        [
            row["center_lat"],
            row["center_lng"],
            (row["avg_popularity"] - min_score) / score_range,
        ]
        for _, row in grid_df.iterrows()
    ]
    HeatMap(
        heat_data,
        min_opacity=0.35,
        radius=55,
        blur=40,
        max_zoom=15,
        gradient={
            0.0: "#313695",
            0.25: "#74add1",
            0.5: "#fee090",
            0.75: "#f46d43",
            1.0: "#a50026",
        },
    ).add_to(m)

    if businesses_df is not None and not businesses_df.empty:
        cluster = MarkerCluster(name="Restaurants").add_to(m)
        for _, biz in businesses_df.iterrows():
            try:
                lat_b = float(biz["latitude"])
                lng_b = float(biz["longitude"])
            except (TypeError, ValueError):
                continue
            if math.isnan(lat_b) or math.isnan(lng_b):
                continue

            score = float(biz.get("popularity_score", 0.0) or 0.0)
            if score >= _HIGH_SCORE:
                icon_color = "red"
            elif score >= _MED_SCORE:
                icon_color = "orange"
            else:
                icon_color = "green"

            popup_html = (
                f"<div style='font-family:sans-serif;min-width:190px'>"
                f"<b style='font-size:14px'>{biz['name']}</b><br>"
                f"<span style='color:#555'>{biz['categories']}</span><br>"
                f"Rating: <b>{float(biz['avg_rating']):.1f}</b> · "
                f"{int(biz['review_count']):,} reviews<br>"
                f"Popularity score: <b>{score:.2f}</b>"
                f"</div>"
            )

            folium.Marker(
                location=[lat_b, lng_b],
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=biz["name"],
                icon=folium.Icon(color=icon_color, icon="cutlery", prefix="fa"),
            ).add_to(cluster)

    for _, row in grid_df.iterrows():
        try:
            lat = float(row["grid_cell"].split("_")[0])
            lng = float(row["grid_cell"].split("_")[1])
        except (ValueError, IndexError):
            lat = row["center_lat"] - 0.05
            lng = row["center_lng"] - 0.05

        hood = row["neighborhood"]
        popup_html = (
            f"<div style='font-family:sans-serif;min-width:160px'>"
            f"<b style='font-size:14px'>{hood}</b><br>"
            f"<span style='color:#555'>{row['top_cuisine']}</span>"
            f"<hr style='margin:6px 0'>"
            f"Popularity score: <b>{row['avg_popularity']:.2f}</b><br>"
            f"Restaurants: <b>{row['restaurant_count']}</b>"
            f"</div>"
        )

        folium.Rectangle(
            bounds=[[lat, lng], [lat + 0.1, lng + 0.1]],
            color="transparent",
            weight=0,
            fill=True,
            fill_color="white",
            fill_opacity=0.01,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{hood}  |  score {row['avg_popularity']:.2f}  ({row['restaurant_count']} restaurants)",
        ).add_to(m)

    if recs_df is not None and not recs_df.empty:
        for _, rec in recs_df.iterrows():
            try:
                lat_r = float(rec["latitude"])
                lng_r = float(rec["longitude"])
            except (TypeError, ValueError):
                continue
            if math.isnan(lat_r) or math.isnan(lng_r):
                continue

            stars = "★" * min(round(float(rec["avg_rating"])), 5)
            popup_html = (
                f"<div style='font-family:sans-serif;min-width:190px'>"
                f"<b style='font-size:14px'>#{int(rec['rank'])}. {rec['name']}</b><br>"
                f"<span style='color:#f9a825'>{stars}</span> "
                f"{rec['avg_rating']:.1f} ({rec['review_count']:,} reviews)<br>"
                f"<span style='color:#555;font-size:12px'>{rec['categories']}</span><br>"
                f"<b style='color:#1565c0'>Predicted: {rec['predicted_rating']:.1f} ★</b>"
                f"</div>"
            )

            folium.Marker(
                location=[lat_r, lng_r],
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=f"#{int(rec['rank'])} {rec['name']}",
                icon=folium.DivIcon(
                    html=(
                        f'<div style="background:#1565c0;color:white;border-radius:50%;'
                        f'width:26px;height:26px;text-align:center;line-height:26px;'
                        f'font-weight:bold;font-size:11px;border:2px solid white;'
                        f'box-shadow:0 2px 6px rgba(0,0,0,0.45)">'
                        f'{int(rec["rank"])}'
                        f'</div>'
                    ),
                    icon_size=(26, 26),
                    icon_anchor=(13, 13),
                ),
            ).add_to(m)

    legend_html = (
        "<div style='position:fixed;bottom:28px;right:10px;z-index:9999;"
        "background:white;padding:10px 14px;border-radius:8px;"
        "box-shadow:0 2px 8px rgba(0,0,0,0.18);font-family:sans-serif;"
        "font-size:12px;line-height:2'>"
        "<b>Popularity</b><br>"
        "<span style='background:#c62828;display:inline-block;width:12px;"
        "height:12px;border-radius:2px;margin-right:6px'></span>High<br>"
        "<span style='background:#ef6c00;display:inline-block;width:12px;"
        "height:12px;border-radius:2px;margin-right:6px'></span>Medium<br>"
        "<span style='background:#2e7d32;display:inline-block;width:12px;"
        "height:12px;border-radius:2px;margin-right:6px'></span>Low"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# UI — sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, str, int]:
    """Render sidebar and return (selected_city, cuisine_filter, user_id, pin_count)."""
    st.sidebar.markdown("## 🍽️ CityBite")
    st.sidebar.caption("Restaurant popularity intelligence")

    cities = load_cities()
    if not cities:
        st.sidebar.error("No cities found. Run the pipeline first.")
        st.stop()

    selected_city = st.sidebar.selectbox("City", cities, index=0)
    all_cuisines = _load_cuisines_for_city(selected_city)
    cuisine_filter = st.sidebar.selectbox("Cuisine", ["All"] + all_cuisines, index=0)

    user_id = st.sidebar.text_input(
        "Yelp User ID",
        placeholder="Paste user_id for personalized picks ...",
        help="Enter a Yelp user_id to see your top-10 restaurant recommendations pinned on the map.",
    )

    pin_count = st.sidebar.slider(
        "Pins on map",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Number of top-rated restaurants to pin on the map.",
    )

    st.sidebar.markdown(
        "**Map key**  \n"
        "🟥 High &nbsp; 🟧 Moderate &nbsp; 🟩 Low  \n"
        "🔵 Your recommendations"
    )

    return selected_city, cuisine_filter, user_id, pin_count


# ---------------------------------------------------------------------------
# UI — main panels
# ---------------------------------------------------------------------------

def render_map_panel(city: str, cuisine_filter: str, user_id: str, pin_count: int = 50) -> None:
    """Render the popularity heatmap with optional recommendation pins."""

    grid_df = load_grid_data(city)

    if grid_df.empty:
        st.warning(f"No grid data for {city}. Run the pipeline to populate the DB.")
        return

    recs_df: pd.DataFrame | None = None
    if user_id:
        recs_df, _ = load_recommendations(user_id, city)
        if recs_df is not None and recs_df.empty:
            recs_df = None

    biz_df = load_businesses(city, cuisine_filter)
    pinned_df = biz_df.head(pin_count)

    c1, c2, c3 = st.columns(3)
    total = int(grid_df['restaurant_count'].sum())
    c1.metric("Neighborhoods", len(grid_df))
    c2.metric("Restaurants in DB", f"{total:,}", help=f"Top {len(pinned_df)} highest-rated pinned on map")
    c3.metric("Avg popularity score", f"{grid_df['avg_popularity'].mean():.2f}")

    label = f"Top restaurants — {city}"
    if cuisine_filter != "All":
        label += f" · {cuisine_filter}"

    with st.expander(label, expanded=False):
        if biz_df.empty:
            st.info("No businesses match the current filter.")
        else:
            st.dataframe(
                biz_df[
                    ["name", "categories", "avg_rating", "review_count", "popularity_score"]
                ].rename(columns={
                    "name": "Name",
                    "categories": "Cuisine",
                    "avg_rating": "Avg Rating",
                    "review_count": "Reviews",
                    "popularity_score": "Score",
                }),
                use_container_width=True,
                hide_index=True,
                height=300,
            )

    m = build_map(grid_df, pinned_df, recs_df)

    # Render as a self-contained HTML iframe.  st_folium intercepts tile-layer
    # requests and only activates them after a user clicks the layer control,
    # leaving the basemap blank on first load.  components.html embeds the full
    # Folium map HTML directly — tiles load immediately without any interaction.
    map_html = m._repr_html_() + """
<script>
(function() {
    var resize = function() {
        var h = (window.parent || window).innerHeight;
        var target = Math.max(300, h - 220);
        var el = window.frameElement;
        if (el) { el.style.height = target + 'px'; el.height = target; }
        document.body.style.height = target + 'px';
    };
    resize();
    window.addEventListener('resize', resize);
})();
</script>"""
    components.html(map_html, height=600, scrolling=False)


def render_recommendations_panel(user_id: str, city: str) -> None:
    """Render the personalized ALS recommendations list."""
    if not user_id:
        st.info(
            "Enter your Yelp User ID in the sidebar to see personalized "
            "restaurant recommendations pinned on the map."
        )
        return

    recs_df, city_filtered = load_recommendations(user_id, city)

    if recs_df.empty:
        st.warning(
            f"No recommendations found for `{user_id}`. "
            "Run the ALS training job to generate them."
        )
        return

    if city_filtered:
        st.caption(f"Your top picks in **{city}** — pinned on the map")
    else:
        st.caption(
            f"No picks found in {city} — showing your top picks across all cities"
        )

    for _, row in recs_df.iterrows():
        rating = float(row["avg_rating"])
        full  = min(round(rating), 5)
        empty = 5 - full
        stars = "★" * full + "☆" * empty

        predicted = float(row["predicted_rating"])
        match_pct = min(int(predicted / 5 * 100), 100)

        with st.container(border=True):
            st.markdown(
                f"**#{int(row['rank'])}.** {row['name']}",
            )
            st.markdown(
                f"<span style='color:#f9a825;font-size:16px'>{stars}</span> "
                f"<span style='color:#555'>{rating:.1f} &nbsp;·&nbsp; "
                f"{int(row['review_count']):,} reviews</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{row['categories']}  ·  {row['city']}")
            st.progress(match_pct / 100, text=f"Match score: {predicted:.1f} / 5")


def render_sentiment_panel(city: str) -> None:
    """Render per-neighborhood sentiment bar chart."""
    sentiment_df = load_sentiment(city)

    if sentiment_df.empty:
        st.info(
            "No sentiment data yet. Run the sentiment job to populate."
        )
        return

    sentiment_df = add_neighborhood_labels(sentiment_df)

    st.caption(
        "Share of positive reviews (4–5 stars) per neighborhood. "
        "Higher = more satisfied diners."
    )

    chart_df = (
        sentiment_df
        .set_index("neighborhood")[["sentiment_score"]]
        .sort_values("sentiment_score", ascending=False)
        .head(20)
    )
    st.bar_chart(chart_df, height=280)

    with st.expander("Full sentiment breakdown", expanded=False):
        display = sentiment_df[
            ["neighborhood", "sentiment_score", "positive_count",
             "negative_count", "restaurant_count"]
        ].copy()
        display["sentiment_score"] = display["sentiment_score"].map("{:.1%}".format)
        st.dataframe(
            display.rename(columns={
                "neighborhood": "Neighborhood",
                "sentiment_score": "Positive Rate",
                "positive_count": "Positive Reviews",
                "negative_count": "Negative Reviews",
                "restaurant_count": "Restaurants",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    selected_city, cuisine_filter, user_id, pin_count = render_sidebar()

    tab_map, tab_recs, tab_sentiment = st.tabs(["🗺️ Map", "⭐ Top Picks For You", "😊 Neighborhood Sentiment"])

    with tab_map:
        render_map_panel(selected_city, cuisine_filter, user_id, pin_count)

    with tab_recs:
        render_recommendations_panel(user_id, selected_city)

    with tab_sentiment:
        render_sentiment_panel(selected_city)


if __name__ == "__main__":
    main()

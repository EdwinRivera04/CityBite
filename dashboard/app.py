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
                text("SELECT DISTINCT metro_area FROM business_scores ORDER BY metro_area"), conn
            )
        return df["metro_area"].tolist()
    except Exception as e:
        st.error(f"Could not load cities: {e}. Make sure the pipeline has run.")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def load_grid_data(city: str) -> pd.DataFrame:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM grid_aggregates WHERE metro_area = :city"),
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
            "WHERE metro_area = :city AND LOWER(categories) LIKE LOWER(:cat) "
            "ORDER BY popularity_score DESC"
        )
        params = {"city": city, "cat": f"%{cuisine_filter}%"}
    else:
        query = text(
            "SELECT * FROM business_scores WHERE metro_area = :city "
            "ORDER BY popularity_score DESC"
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
    city_query = text(base_select + " AND b.metro_area = :city ORDER BY r.rank")
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
def find_proxy_user(city: str, cuisines: tuple) -> str | None:
    """
    Find the user whose ALS recommendations best match any of the given cuisines
    in the selected city. Returns a user_id string or None if no match found.
    cuisines must be a tuple (hashable) for st.cache_data.
    """
    if not cuisines:
        return None
    engine = get_engine()
    # Build OR clause dynamically for each cuisine
    cuisine_clauses = " OR ".join(
        f"LOWER(b.categories) LIKE LOWER(:cat{i})" for i in range(len(cuisines))
    )
    query = text(f"""
        SELECT r.user_id,
               COUNT(*)                  AS matches,
               AVG(r.predicted_rating)   AS avg_pred
        FROM   als_recommendations r
        JOIN   business_scores b ON r.business_id = b.business_id
        WHERE  b.metro_area = :city
          AND  ({cuisine_clauses})
        GROUP  BY r.user_id
        ORDER  BY matches DESC, avg_pred DESC
        LIMIT  1
    """)
    params = {"city": city}
    for i, c in enumerate(cuisines):
        params[f"cat{i}"] = f"%{c}%"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        return df["user_id"].iloc[0] if not df.empty else None
    except Exception as e:
        st.error(f"Could not find a matching taste profile: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_profiles(city: str) -> pd.DataFrame:
    """
    Load business profiles for NLP search.

    Tries business_profiles first (rich profiles with review text).
    Falls back to business_scores.categories when the NLP index hasn't been
    run for this metro area — so search works everywhere immediately.
    """
    engine = get_engine()

    # Try rich profiles table first
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT business_id, name, city, latitude, longitude, "
                    "avg_rating, review_count, popularity_score, profile_text "
                    "FROM business_profiles WHERE metro_area = :city"
                ),
                conn, params={"city": city},
            )
        if not df.empty:
            return df
    except Exception:
        pass  # table missing or query failed — fall through to fallback

    # Fallback: build profile_text from categories stored in business_scores
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT business_id, name, city, latitude, longitude, "
                    "avg_rating, review_count, popularity_score, categories AS profile_text "
                    "FROM business_scores WHERE metro_area = :city"
                ),
                conn, params={"city": city},
            )
        if not df.empty:
            # Weight categories 3× to mirror what nlp_index.py does
            df["profile_text"] = df["profile_text"].fillna("").apply(
                lambda c: f"{c} {c} {c}".strip()
            )
        return df
    except Exception as e:
        st.error(f"Could not load profiles: {e}")
        return pd.DataFrame()


def nlp_search(query: str, profiles_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top_n businesses whose profile_text best matches the query
    using TF-IDF cosine similarity.
    """
    if profiles_df.empty or not query.strip():
        return pd.DataFrame()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        st.error("scikit-learn is required for NLP search. Run: pip install scikit-learn")
        return pd.DataFrame()

    corpus = profiles_df["profile_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Blend: 60% TF-IDF relevance + 25% log-popularity + 15% avg_rating
    review_counts = profiles_df["review_count"].fillna(0).astype(float).values
    log_pop = np.log1p(review_counts)
    pop_norm = log_pop / (log_pop.max() or 1.0)

    ratings = profiles_df["avg_rating"].fillna(0).astype(float).values
    rating_norm = ratings / 5.0

    scores = tfidf_scores * 0.6 + pop_norm * 0.25 + rating_norm * 0.15

    # Only return results where the query actually matched something
    top_idx = np.argsort(scores)[::-1][:top_n]
    result = profiles_df.iloc[top_idx].copy()
    result["match_score"] = tfidf_scores[top_idx]
    result["final_score"] = scores[top_idx]
    return result[result["match_score"] > 0].reset_index(drop=True)


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
        WHERE  g.metro_area = :city
        ORDER  BY sentiment_score DESC
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"city": city})
    except Exception as e:
        st.error(f"Could not load sentiment data: {e}")
        return pd.DataFrame()


_CUISINE_ALLOWLIST: frozenset = frozenset([
    # Cuisines by origin
    "mexican", "italian", "chinese", "japanese", "thai", "indian", "french",
    "mediterranean", "greek", "spanish", "vietnamese", "korean",
    "american (traditional)", "american (new)", "southern", "cajun/creole",
    "middle eastern", "latin american", "caribbean", "hawaiian", "filipino",
    "taiwanese", "cantonese", "szechuan", "dim sum", "ethiopian", "african",
    "german", "irish", "british", "turkish", "persian/iranian", "pakistani",
    "portuguese", "scandinavian",
    # Food types / styles
    "pizza", "burgers", "sandwiches", "tacos", "sushi bars", "sushi", "ramen",
    "steakhouses", "seafood", "barbeque", "chicken wings", "chicken shop",
    "hot dogs", "cheesesteaks", "wraps", "salad", "soup", "poke", "waffles",
    "creperies", "fondue", "tapas/small plates", "hot pot", "buffets",
    "comfort food", "soul food", "gastropubs", "diners", "fast food",
    "food trucks", "food stands", "caterers",
    # Meal types
    "breakfast & brunch", "brunch",
    # Drink establishments
    "wine bars", "cocktail bars", "sports bars", "dive bars", "pubs",
    "breweries", "brewpubs", "wineries", "distilleries", "sake bars",
    "whiskey bars", "beer gardens",
    # Coffee / tea / juice
    "coffee & tea", "cafes", "juice bars & smoothies", "bubble tea", "tea rooms",
    # Desserts / bakeries
    "desserts", "ice cream & frozen yogurt", "bakeries", "patisserie/cake shop",
    "cupcakes", "donuts", "candy stores", "chocolatiers & shops", "gelato",
    # Other specific food
    "delis", "acai bowls", "empanadas", "falafel", "gyros", "kebab",
    "noodles", "dumplings", "bagels", "fish & chips",
])


@st.cache_data(ttl=3600, show_spinner=False)
def _load_cuisines_for_city(city: str) -> list[str]:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT categories FROM business_scores WHERE metro_area = :city"),
                conn, params={"city": city},
            )
        return sorted({
            tag.strip()
            for cats in df["categories"].dropna()
            for tag in cats.split(",")
            if tag.strip() and tag.strip().lower() in _CUISINE_ALLOWLIST
        })
    except Exception:
        return []


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

def render_sidebar() -> tuple[str, str, int]:
    """Render sidebar and return (city, cuisine_filter, pin_count)."""
    st.sidebar.markdown("## CityBite")
    st.sidebar.caption("Restaurant popularity intelligence")

    cities = load_cities()
    if not cities:
        st.sidebar.error("No cities found. Run the pipeline first.")
        st.stop()

    selected_city = st.sidebar.selectbox("City", cities, index=0)
    all_cuisines = _load_cuisines_for_city(selected_city)
    cuisine_filter = st.sidebar.selectbox("Cuisine", ["All"] + all_cuisines, index=0)

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
        "🔴 High &nbsp; 🟠 Moderate &nbsp; 🟢 Low  \n"
        "📍 Your recommendations"
    )

    return selected_city, cuisine_filter, pin_count


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


def render_recommendations_panel(city: str, effective_user_id: str | None) -> None:
    """Render recommendation controls and the ALS recommendations list."""

    # ── Mode toggle ────────────────────────────────────────────────────────
    rec_mode = st.radio(
        "Find picks by",
        options=["cuisine", "user_id"],
        format_func=lambda x: "Cuisine Taste" if x == "cuisine" else "Yelp User ID",
        horizontal=True,
        label_visibility="collapsed",
        key="rec_mode",
    )

    # ── Input controls ─────────────────────────────────────────────────────
    if rec_mode == "cuisine":
        all_cuisines = _load_cuisines_for_city(city)
        st.multiselect(
            "Cuisine",
            options=all_cuisines,
            max_selections=3,
            placeholder="Select up to 3 cuisines...",
            label_visibility="collapsed",
            key="rec_cuisines",
        )
        selected_cuisines = st.session_state.get("rec_cuisines", [])
        if not selected_cuisines:
            st.caption("Select up to 3 cuisines above to get taste-based recommendations.")
            return
        if not effective_user_id:
            cuisine_str = ", ".join(selected_cuisines)
            st.warning(f"No taste profile found for {cuisine_str} in {city}. Run the ALS training job to generate recommendations.")
            return
        cuisine_str = " + ".join(selected_cuisines)
        st.caption(f"Top picks for **{cuisine_str}** lovers in **{city}** — pinned on the map")

    else:
        st.text_input(
            "Yelp User ID",
            placeholder="Paste user_id ...",
            label_visibility="collapsed",
            key="rec_user_id_text",
        )
        if not effective_user_id:
            st.caption("Enter a Yelp User ID above to see personalized recommendations.")
            return

    # ── Recommendations list ───────────────────────────────────────────────
    recs_df, city_filtered = load_recommendations(effective_user_id, city)

    if recs_df.empty:
        st.warning("No recommendations found. Run the ALS training job to generate them.")
        return

    if rec_mode == "user_id":
        if city_filtered:
            st.caption(f"Your top picks in **{city}** — pinned on the map")
        else:
            st.caption(f"No picks found in {city} — showing your top picks across all cities")

    for _, row in recs_df.iterrows():
        rating = float(row["avg_rating"])
        full  = min(round(rating), 5)
        empty = 5 - full
        stars = "★" * full + "☆" * empty

        predicted = float(row["predicted_rating"])
        match_pct = min(int(predicted / 5 * 100), 100)

        st.markdown(
            f"<div style='border:1px solid #444;border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
            f"<b>#{int(row['rank'])}. {row['name']}</b><br>"
            f"<span style='color:#f9a825;font-size:16px'>{stars}</span> "
            f"<span style='color:#888'>{rating:.1f} &nbsp;·&nbsp; {int(row['review_count']):,} reviews</span><br>"
            f"<span style='color:#aaa;font-size:12px'>{row['categories']} · {row['city']}</span><br>"
            f"<div style='background:#333;border-radius:4px;height:6px;margin-top:6px'>"
            f"<div style='background:#1565c0;width:{match_pct}%;height:6px;border-radius:4px'></div></div>"
            f"<span style='color:#888;font-size:11px'>Match score: {predicted:.1f} / 5</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_nlp_panel(city: str) -> None:
    """Render the natural-language restaurant search panel."""
    st.caption(
        f"Describe what you're craving and we'll find the best match in **{city}**. "
        "Try: *spicy ramen with great broth*, *outdoor patio brunch*, *authentic tacos*."
    )

    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. cozy Italian pasta with good wine...",
        key="nlp_query",
    )

    if not query or not query.strip():
        return

    profiles_df = load_profiles(city)
    if profiles_df.empty:
        st.warning(f"No restaurant data found for {city}. Make sure the pipeline has run.")
        return

    with st.spinner("Searching..."):
        results = nlp_search(query, profiles_df)

    if results.empty:
        st.info("No matches found. Try different keywords.")
        return

    st.markdown(f"**Top {len(results)} matches** for *\"{query}\"*")

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        rating = float(row.get("avg_rating") or 0.0)
        full  = min(round(rating), 5)
        stars = "★" * full + "☆" * (5 - full)
        # final_score is already in [0,1] — show it directly so bars reflect true spread
        match_pct = min(int(float(row["final_score"]) * 100), 100)

        st.markdown(
            f"<div style='border:1px solid #444;border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
            f"<b>#{rank}. {row['name']}</b><br>"
            f"<span style='color:#f9a825;font-size:16px'>{stars}</span> "
            f"<span style='color:#888'>{rating:.1f} &nbsp;·&nbsp; {int(row['review_count']):,} reviews</span><br>"
            f"<span style='color:#aaa;font-size:12px'>{row['city']}</span><br>"
            f"<div style='background:#333;border-radius:4px;height:6px;margin-top:6px'>"
            f"<div style='background:#c62828;width:{match_pct}%;height:6px;border-radius:4px'></div></div>"
            f"<span style='color:#888;font-size:11px'>Match: {match_pct}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_sentiment_panel(city: str) -> None:
    """Render per-neighborhood diner satisfaction scores."""
    sentiment_df = load_sentiment(city)

    if sentiment_df.empty:
        st.info("No sentiment data yet. Run the sentiment job to populate.")
        return

    # Use grid_df as the label source so neighborhood names match the map exactly
    grid_df = load_grid_data(city)
    labeled_grid = add_neighborhood_labels(grid_df)[["grid_cell", "neighborhood"]]
    sentiment_df = sentiment_df.merge(labeled_grid, on="grid_cell", how="left")

    # Bayesian-adjusted satisfaction: shrink low-volume neighborhoods toward the
    # global average so a place with 3 five-star reviews doesn't rank above a
    # busy neighborhood with thousands. k=30 means ~30 reviews of "prior" weight.
    total_pos = sentiment_df["positive_count"].sum()
    total_rev = (sentiment_df["positive_count"] + sentiment_df["negative_count"]).sum()
    global_rate = total_pos / total_rev if total_rev > 0 else 0.5
    k = 30
    sentiment_df["satisfaction"] = (
        (sentiment_df["positive_count"] + k * global_rate)
        / (sentiment_df["positive_count"] + sentiment_df["negative_count"] + k)
        * 10
    ).round(1)

    all_rows = (
        sentiment_df
        .sort_values("satisfaction", ascending=False)
        .reset_index(drop=True)
    )

    st.caption(f"Diner satisfaction score (0–10) based on review sentiment — {city}")

    display = all_rows[["neighborhood", "satisfaction", "positive_count", "negative_count", "restaurant_count"]].copy()
    display.index = range(1, len(display) + 1)

    row_px = 35
    header_px = 38
    table_height = header_px + row_px * len(display)

    st.dataframe(
        display.rename(columns={
            "neighborhood": "Neighborhood",
            "satisfaction": "Satisfaction (/ 10)",
            "positive_count": "Positive Reviews",
            "negative_count": "Negative Reviews",
            "restaurant_count": "Restaurants",
        }),
        use_container_width=True,
        height=table_height,
        column_config={
            "Satisfaction (/ 10)": st.column_config.ProgressColumn(
                "Satisfaction (/ 10)",
                min_value=0,
                max_value=10,
                format="%.1f",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    selected_city, cuisine_filter, pin_count = render_sidebar()

    # Resolve effective_user_id from widget session state BEFORE rendering any
    # tab so the map pins and the recs panel always use the same value in one pass.
    rec_mode = st.session_state.get("rec_mode", "cuisine")
    effective_user_id: str | None = None
    if rec_mode == "cuisine":
        selected_cuisines = st.session_state.get("rec_cuisines", [])
        if selected_cuisines:
            effective_user_id = find_proxy_user(selected_city, tuple(selected_cuisines))
    else:
        raw = st.session_state.get("rec_user_id_text", "") or ""
        effective_user_id = raw.strip() or None

    tab_map, tab_recs, tab_nlp, tab_sentiment = st.tabs(
        ["Map", "Top Picks For You", "Describe What You Want", "Neighborhood Sentiment"]
    )

    with tab_map:
        render_map_panel(selected_city, cuisine_filter, effective_user_id, pin_count)

    with tab_recs:
        render_recommendations_panel(selected_city, effective_user_id)

    with tab_nlp:
        render_nlp_panel(selected_city)

    with tab_sentiment:
        render_sentiment_panel(selected_city)


if __name__ == "__main__":
    main()

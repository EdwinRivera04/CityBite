CREATE TABLE IF NOT EXISTS business_scores (
    business_id      VARCHAR(50) PRIMARY KEY,
    name             TEXT,
    city             VARCHAR(100),
    metro_area       VARCHAR(100),
    latitude         FLOAT,
    longitude        FLOAT,
    grid_cell        VARCHAR(20),
    categories       TEXT,
    avg_rating       FLOAT,
    review_count     INT,
    popularity_score FLOAT,
    last_updated     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS grid_aggregates (
    grid_cell         VARCHAR(20) PRIMARY KEY,
    metro_area        VARCHAR(100),
    center_lat        FLOAT,
    center_lng        FLOAT,
    avg_popularity    FLOAT,
    restaurant_count  INT,
    top_cuisine       TEXT
);

CREATE TABLE IF NOT EXISTS als_recommendations (
    user_id          VARCHAR(50),
    business_id      VARCHAR(50),
    predicted_rating FLOAT,
    rank             INT,
    PRIMARY KEY (user_id, business_id)
);

CREATE TABLE IF NOT EXISTS grid_sentiment (
    grid_cell       VARCHAR(20) PRIMARY KEY,
    sentiment_score FLOAT,
    positive_count  INT,
    negative_count  INT,
    last_updated    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_business_city   ON business_scores(city);
CREATE INDEX IF NOT EXISTS idx_business_metro  ON business_scores(metro_area);
CREATE INDEX IF NOT EXISTS idx_grid_metro      ON grid_aggregates(metro_area);
CREATE INDEX IF NOT EXISTS idx_als_user        ON als_recommendations(user_id);

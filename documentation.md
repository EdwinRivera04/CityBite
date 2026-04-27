# CityBite — Project Documentation

**Course:** CS 4266 — Big Data Systems, Vanderbilt University  
**Authors:** Syeda Ali (Person B — ML & Visualization) · Edwin Rivera (Person A — Infrastructure & Pipeline)  
**Stack:** AWS S3 · EMR · RDS PostgreSQL · PySpark · scikit-learn · Streamlit · Folium

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture](#3-architecture)
4. [Prerequisites](#4-prerequisites)
5. [Environment Setup](#5-environment-setup)
6. [Data](#6-data)
7. [AWS Infrastructure](#7-aws-infrastructure)
8. [Pipeline Jobs](#8-pipeline-jobs)
   - [upload.py — S3 Ingest](#uploadpy--s3-ingest)
   - [clean_job.py — PySpark Cleaning](#clean_jobpy--pyspark-cleaning)
   - [aggregate_job.py — Spark SQL Aggregation](#aggregate_jobpy--spark-sql-aggregation)
   - [submit_emr.py — EMR Orchestration](#submit_emrpy--emr-orchestration)
9. [ML Models](#9-ml-models)
   - [als_train.py — Collaborative Filtering](#als_trainpy--collaborative-filtering)
   - [sentiment.py — Sentiment Classifier](#sentimentpy--sentiment-classifier)
   - [nlp_index.py — Business Profile Builder](#nlp_indexpy--business-profile-builder)
   - [evaluate.py — Standalone Evaluation](#evaluatepy--standalone-evaluation)
   - [seed_local_db.py — Local DB Seeding](#seed_local_dbpy--local-db-seeding)
10. [Database Schema](#10-database-schema)
11. [Dashboard](#11-dashboard)
12. [Testing](#12-testing)
13. [Local Development Workflow](#13-local-development-workflow)
14. [Production Workflow (AWS)](#14-production-workflow-aws)
15. [Common Issues and Fixes](#15-common-issues-and-fixes)

---

## 1. Project Overview

CityBite is a restaurant popularity intelligence platform that ingests, processes, and analyzes 7M+ Yelp reviews across 150K+ businesses. It produces interactive, city-level maps of restaurant popularity, personalized recommendations, and neighborhood sentiment rankings.

**Core value proposition:** A plain star average is an unreliable guide. CityBite replaces it with a composite popularity score that weights rating quality, review volume (log-scaled to reduce viral bias), and recency — then makes the results explorable through a live Streamlit dashboard.

**End-to-end pipeline:**

```
Yelp JSON (~9 GB)
    └── S3 raw zone
          └── PySpark clean job (EMR)
                └── S3 processed zone (Parquet, partitioned by metro area)
                      ├── Spark SQL aggregate job  ──►  business_scores + grid_aggregates (RDS)
                      ├── ALS recommender          ──►  als_recommendations (RDS)
                      ├── TF-IDF sentiment job     ──►  grid_sentiment (RDS)
                      └── NLP profile builder      ──►  business_profiles (RDS)
                                                              │
                                                    Streamlit dashboard (EC2)
```

**Team responsibilities:**

| Person | Owner |
|---|---|
| Person A (Edwin) | S3, EMR, PySpark pipeline, RDS provisioning |
| Person B (Syeda) | ALS recommender, sentiment classifier, NLP search, Streamlit dashboard |

---

## 2. Repository Structure

```
CityBite/
├── CLAUDE.md                    # Project instructions for Claude Code
├── README.md                    # Public-facing overview
├── documentation.md             # This file
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Local dev environment (optional)
├── .env.example                 # Environment variable template
│
├── assets/                      # Generated figures (notebook output)
│   ├── CityBite-architecture.jpg
│   ├── fig_raw_data.png
│   ├── fig_before_after.png
│   ├── fig_top10_cities.png
│   ├── fig_als_rmse.png
│   ├── fig_sentiment.png
│   └── fig_spatial.png
│
├── data/
│   ├── sample/                  # 1% Yelp subset for local dev
│   │   ├── yelp_academic_dataset_business.json
│   │   └── yelp_academic_dataset_review.json
│   └── citybite_local.db        # SQLite database (generated; not committed)
│
├── pipeline/
│   ├── upload.py                # Boto3: local Yelp JSON → S3 raw zone
│   ├── clean_job.py             # PySpark: raw JSON → enriched Parquet
│   ├── aggregate_job.py         # Spark SQL: Parquet → business_scores + grid_aggregates
│   └── submit_emr.py            # Launch transient EMR cluster or add step to existing
│
├── ml/
│   ├── als_train.py             # Spark MLlib ALS recommender
│   ├── sentiment.py             # TF-IDF + LogisticRegression sentiment classifier
│   ├── nlp_index.py             # Natural-language business profile builder
│   ├── evaluate.py              # Standalone RMSE + F1 + Precision@k evaluation
│   ├── seed_local_db.py         # Seed SQLite from Parquet (local dev)
│   ├── seed_rds.py              # Seed RDS from Parquet (production)
│   └── push_local_to_rds.py     # Push local SQLite data to RDS
│
├── dashboard/
│   └── app.py                   # Streamlit app (Folium heatmap + recs + NLP + sentiment)
│
├── notebooks/
│   └── analysis.ipynb           # End-to-end analysis and evaluation notebook
│
├── infra/
│   ├── create_rds.py            # Provision RDS PostgreSQL via boto3
│   ├── schema.sql               # PostgreSQL + SQLite table definitions
│   └── cron_setup.sh            # Cron job setup for nightly pipeline runs
│
└── tests/
    ├── test_als_train.py
    ├── test_clean_job.py
    ├── test_sentiment.py
    ├── test_upload.py
    └── test_submit_emr.py
```

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT: Yelp Academic Dataset (~9 GB, 7M+ reviews, 150K businesses)      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  pipeline/upload.py  (boto3 multipart)
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  S3 RAW ZONE   s3://citybite/raw/                                        │
│    business.json  ·  review.json  ·  user.json                           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  pipeline/clean_job.py  (PySpark on EMR)
                                │    • drop nulls, filter open restaurants
                                │    • metro-area clustering (haversine, 50 km)
                                │    • add recency_weight + grid_cell
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  S3 PROCESSED ZONE   s3://citybite/processed/reviews_enriched/           │
│    Parquet, partitioned by metro_area                                    │
└──────┬──────────────────────────┬──────────────────┬─────────────────────┘
       │                          │                  │
  aggregate_job.py           als_train.py       sentiment.py + nlp_index.py
  (Spark SQL)                (Spark MLlib)      (scikit-learn + PySpark)
       │                          │                  │
       ▼                          ▼                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RDS PostgreSQL  (or SQLite locally)                                     │
│    business_scores  ·  grid_aggregates  ·  als_recommendations           │
│    grid_sentiment   ·  business_profiles                                 │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard  (dashboard/app.py, served on EC2)                  │
│    Tab 1: Folium heatmap — popularity by grid cell                       │
│    Tab 2: ALS top-10 recommendations (by cuisine taste or user ID)       │
│    Tab 3: Natural-language restaurant search (TF-IDF)                    │
│    Tab 4: Neighborhood sentiment ranking (Bayesian-adjusted)             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.10 | 3.10+ required for `str | None` type hints |
| Java | 11 | Required by PySpark; set `JAVA_HOME` |
| AWS CLI | any | Run `aws configure` before using boto3 scripts |
| AWS account | — | S3, EMR, RDS, EC2 access required |
| Yelp dataset | — | Download from yelp.com/dataset |

---

## 5. Environment Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
pyspark==3.5.0
boto3==1.34.0
pandas==2.1.0
scikit-learn==1.3.0
streamlit==1.35.0
folium==0.15.0
streamlit-folium==0.15.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
python-dotenv==1.0.0
numpy==1.26.0
geopy==2.4.1

# Testing
pytest==7.4.0
pytest-mock==3.12.0
moto[s3,rds,emr]==4.2.0
```

### Environment variables

Copy `.env.example` to `.env` and populate all values:

```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
S3_BUCKET=citybite-560580021963

EMR_CLUSTER_ID=j-XXXXXXXXXX        # only needed for persistent-cluster mode

RDS_HOST=your-rds-endpoint.rds.amazonaws.com
RDS_PORT=5432
RDS_DB=citybite
RDS_USER=citybite_user
RDS_PASSWORD=your_password
```

All scripts load `.env` automatically via `python-dotenv`. On EMR, environment variables are injected through the cluster `Configurations` block (handled by `submit_emr.py`).

### Windows — PySpark local mode

Windows requires Hadoop native binaries for local Spark writes. Set `HADOOP_HOME` to a directory containing `bin/winutils.exe` and `bin/hadoop.dll` (both from a Hadoop 3.x build):

```bash
set HADOOP_HOME=C:\hadoop
```

`clean_job.py` validates this on startup and raises an explicit error if the binaries are missing.

---

## 6. Data

### Yelp Academic Dataset

Download from: https://www.yelp.com/dataset  
Files needed:
- `yelp_academic_dataset_business.json` (~120 MB)
- `yelp_academic_dataset_review.json` (~5 GB)
- `yelp_academic_dataset_user.json` (~3.5 GB)

Place downloaded files in `data/raw/` for production use, or use the pre-included 1% sample in `data/sample/` for local development.

**Sample dataset (included in repo):**

| File | Rows | Approximate size |
|---|---|---|
| `yelp_academic_dataset_business.json` | 300 businesses | 73 KB |
| `yelp_academic_dataset_review.json` | 3,000 reviews | 749 KB |

Cities covered by sample: Phoenix (100), Las Vegas (100), Charlotte (100).

### S3 Zone Structure

```
s3://citybite/
├── raw/
│   ├── yelp_academic_dataset_business.json
│   ├── yelp_academic_dataset_review.json
│   └── yelp_academic_dataset_user.json
├── processed/
│   └── reviews_enriched/
│       ├── metro_area=Phoenix/
│       ├── metro_area=Las Vegas/
│       └── metro_area=Charlotte/
├── gold/
│   ├── business_scores/metro_area=Phoenix/
│   └── grid_aggregates/metro_area=Phoenix/
├── scripts/                     # Pipeline scripts uploaded by submit_emr.py
│   ├── clean_job.py
│   ├── aggregate_job.py
│   ├── als_train.py
│   ├── sentiment.py
│   ├── nlp_index.py
│   └── bootstrap.sh
└── logs/
    └── emr/                     # EMR step logs
```

---

## 7. AWS Infrastructure

### Step 1 — Create S3 bucket

```bash
aws s3 mb s3://citybite --region us-east-1
```

### Step 2 — Provision RDS PostgreSQL

`infra/create_rds.py` provisions a `db.t3.micro` PostgreSQL 15.7 instance (free-tier eligible) in the default VPC.

```bash
python infra/create_rds.py
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--instance-id` | `citybite-dev` | RDS instance identifier |
| `--db-name` | `$RDS_DB` | Database name |
| `--username` | `$RDS_USER` | Master username |
| `--password` | `$RDS_PASSWORD` | Master password (required) |
| `--dry-run` | `false` | Print config without creating |

Defaults: `db.t3.micro`, PostgreSQL 15.7, 20 GB gp2 storage, single-AZ, publicly accessible. The script polls until the instance reaches `available` status (~5–10 minutes) and prints the endpoint to add to `.env`.

After provisioning, apply the schema:

```bash
psql -h $RDS_HOST -U $RDS_USER -d $RDS_DB -f infra/schema.sql
```

### Step 3 — Configure EMR IAM roles

The EMR EC2 instance profile (`EMR_EC2_DefaultRole`) must have:
- `AmazonS3FullAccess`
- `AmazonRDSFullAccess`

Add an inbound rule to the RDS security group allowing TCP port 5432 from the EMR master node's security group.

### Step 4 — EMR cluster configuration

`submit_emr.py` launches clusters using:
- EMR 6.15.0 with Spark 3.5
- 1 on-demand master: `m5.xlarge`
- 2 spot core nodes: `m5.xlarge` (bid $0.10/hr vs ~$0.19 on-demand — ~70% savings)
- Auto-terminates when all steps finish (`KeepJobFlowAliveWhenNoSteps: false`)

A bootstrap script (`infra/bootstrap.sh`) installs Python dependencies on all nodes before any step runs.

---

## 8. Pipeline Jobs

### `upload.py` — S3 Ingest

Uploads local Yelp JSON files to the S3 raw zone. Files larger than 100 MB use multipart upload (4 concurrent parts). Each upload is verified with `head_object` to confirm the remote file size matches the local size.

**Usage:**

```bash
python pipeline/upload.py --source data/raw/ --bucket citybite --prefix raw/
python pipeline/upload.py --source data/sample/ --bucket citybite --prefix raw/  # dev
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--source` | required | Local directory containing JSON files |
| `--bucket` | `$S3_BUCKET` | Target S3 bucket |
| `--prefix` | `raw/` | S3 key prefix |

**Behavior:** Recursively finds all `.json` files under `--source`, uploads each, verifies content length. Exits with code 1 if any upload fails.

---

### `clean_job.py` — PySpark Cleaning

Reads raw Yelp JSON, cleans and enriches the data, and writes Parquet partitioned by `metro_area`.

**Input columns consumed:**

| Source | Columns kept |
|---|---|
| `business.json` | `business_id`, `is_open`, `name`, `city`, `state`, `latitude`, `longitude`, `categories`, `review_count` |
| `review.json` | `review_id`, `business_id`, `user_id`, `stars`, `date`, `text` |

**Processing steps:**

1. **Filter businesses** — Keep only open (`is_open == 1`) food/drink establishments. Uses a `FOOD_DRINK_ANCHORS` frozenset of 50+ category tags (Restaurants, Food, Bars, Coffee & Tea, etc.) to exclude non-food businesses (salons, auto shops, etc.).
2. **Normalize names** — City: `initcap(trim(...))`. State: `upper(trim(...))`. Prevents `"las vegas"` and `"Las Vegas"` producing separate groups.
3. **Metro-area clustering** — Greedy single-linkage haversine clustering with a 50 km radius. Cities are sorted by descending review volume so the highest-traffic city in each geographic cluster becomes the metro anchor name. Handles disambiguation (e.g., Portland OR vs. Portland ME) by appending the state code.
4. **Join** — Reviews joined to businesses on `business_id` (inner join — unmatched reviews are dropped).
5. **Recency weight** — `1 / (1 + days_since_review / 365)`. Reviews from one year ago have weight 0.5; reviews from two years ago have weight 0.33.
6. **Grid cell** — `floor(lat / 0.1) × 0.1` and `floor(lng / 0.1) × 0.1`, formatted as `"33.4_-112.1"`. Each cell covers approximately 11 km².
7. **Write** — Parquet, overwrite mode, partitioned by `metro_area`.

**Output columns:**

```
review_id, business_id, user_id, stars, date, text,
name, city, state, metro_area, latitude, longitude,
categories, review_count,
recency_weight, grid_cell
```

**Usage:**

```bash
# Local
spark-submit pipeline/clean_job.py \
    --input data/sample/ --output data/processed/ --mode local

# EMR (via submit_emr.py — recommended)
python pipeline/submit_emr.py clean
```

**Direct EMR (manual):**

```bash
spark-submit pipeline/clean_job.py \
    --input s3://citybite/raw/ \
    --output s3://citybite/processed/reviews_enriched/ \
    --mode emr
```

---

### `aggregate_job.py` — Spark SQL Aggregation

Reads the enriched Parquet and computes two aggregate tables: per-business popularity scores and per-grid-cell summaries. Writes both to S3 as Parquet and (in EMR mode) to RDS via JDBC.

**Popularity score formula:**

```
popularity_score = 0.4 × avg(stars)
                 + 0.4 × log(count(*) + 1)
                 + 0.2 × (sum(recency_weight) / count(*))
```

| Component | Weight | Rationale |
|---|---|---|
| `avg(stars)` | 40% | Quality signal |
| `log(review_count + 1)` | 40% | Engagement signal; log-scale caps viral outliers |
| `avg(recency_weight)` | 20% | Recency signal; discounts stale businesses |

**Grid aggregation:** The most common `categories` value per grid cell is determined using a `ROW_NUMBER()` window function (ranked by descending count, with alphabetical tie-breaking for determinism). This is the `top_cuisine` column displayed in heatmap popups.

**Output tables:**

`business_scores`:

| Column | Type | Description |
|---|---|---|
| `business_id` | VARCHAR(50) PK | Yelp business ID |
| `name` | TEXT | Business name |
| `city` | VARCHAR(100) | City |
| `metro_area` | VARCHAR(100) | Clustered metro area name |
| `latitude` / `longitude` | FLOAT | Coordinates |
| `grid_cell` | VARCHAR(20) | `"lat_lng"` bucket key |
| `categories` | TEXT | Comma-separated Yelp tags |
| `avg_rating` | FLOAT | Mean star rating |
| `review_count` | INT | Total reviews |
| `recency_score` | FLOAT | Mean recency weight |
| `popularity_score` | FLOAT | Composite score |

`grid_aggregates`:

| Column | Type | Description |
|---|---|---|
| `grid_cell` | VARCHAR(20) PK | `"lat_lng"` bucket key |
| `metro_area` | VARCHAR(100) | Metro area |
| `center_lat` / `center_lng` | FLOAT | Centroid of business coordinates in cell |
| `avg_popularity` | FLOAT | Mean popularity score across businesses in cell |
| `restaurant_count` | INT | Number of businesses in cell |
| `top_cuisine` | TEXT | Most common category in cell |

**Usage:**

```bash
# Local (no RDS)
spark-submit pipeline/aggregate_job.py \
    --input data/processed/ --output data/gold/ --mode local --skip-jdbc

# EMR (via submit_emr.py — recommended)
python pipeline/submit_emr.py aggregate
```

**Options:**

| Flag | Description |
|---|---|
| `--input` | Path to `reviews_enriched/` Parquet |
| `--output` | Base output path for Parquet files |
| `--mode` | `local` or `emr` |
| `--skip-jdbc` | Omit JDBC write to RDS (for local testing) |

---

### `submit_emr.py` — EMR Orchestration

Uploads pipeline scripts to S3 and either launches a transient auto-terminating cluster or adds steps to an existing cluster.

**Two modes:**

| Mode | Flag | Cost | Use case |
|---|---|---|---|
| Transient (default) | _(omit `--cluster-id`)_ | ~$2–4/run | Standard production run |
| Persistent | `--cluster-id j-XXXX` | ongoing | Interactive debugging |

**Available jobs:**

| Job key | Script | Description |
|---|---|---|
| `clean` | `pipeline/clean_job.py` | PySpark cleaning job |
| `aggregate` | `pipeline/aggregate_job.py` | Spark SQL aggregation (needs PostgreSQL JDBC driver) |
| `setup` | _(pip install)_ | Install Python deps on existing cluster |
| `als` | `ml/als_train.py` | ALS recommender training |
| `sentiment` | `ml/sentiment.py` | Sentiment classifier + grid aggregation |
| `nlp` | `ml/nlp_index.py` | Business profile builder |

**Usage:**

```bash
# Single job — transient cluster
python pipeline/submit_emr.py clean

# Chain multiple jobs on one transient cluster
python pipeline/submit_emr.py clean aggregate

# Existing cluster — submit step and wait
python pipeline/submit_emr.py --job als --cluster-id j-XXXX --wait
```

**Transient cluster config:** Scripts are uploaded to `s3://$S3_BUCKET/scripts/` before cluster launch. RDS credentials are injected via the EMR `Configurations` block (`spark-env` → `export`). Cluster auto-terminates on step completion (`ActionOnFailure: TERMINATE_CLUSTER`). Logs go to `s3://$S3_BUCKET/logs/emr/`.

---

## 9. ML Models

### `als_train.py` — Collaborative Filtering

Trains a Spark MLlib Alternating Least Squares (ALS) recommender on the user-business rating matrix and writes top-10 recommendations per user to the database.

**Algorithm overview:**

ALS factorizes the sparse user-item rating matrix into two low-rank matrices (user factors and item factors) by alternately solving for each while holding the other fixed. The dot product of a user's factor vector and a business's factor vector predicts the user's rating for that business.

**Processing pipeline:**

1. **Load reviews** — Read `reviews_enriched` Parquet; select `user_id`, `business_id`, `stars`.
2. **Build integer index** — `StringIndexer` converts string IDs to dense integer indices (required by Spark ALS). Indexers are fit on the full dataset so training and test sets share the same vocabulary.
3. **Train/test split** — 80/20, `seed=42`.
4. **Train ALS** — Default hyperparameters: `rank=20`, `maxIter=10`, `regParam=0.1`, `coldStartStrategy="drop"`, `nonnegative=True`.
5. **Evaluate RMSE** — `RegressionEvaluator` on held-out test set. Rows with NaN predictions (cold-start users/items) are dropped before evaluation.
6. **Generate recommendations** — `recommendForAllUsers(10)` produces a DataFrame of `(user_idx, [business_idx, rating])` structs. These are exploded and decoded back to string IDs using the reverse index maps. Uses `toPandas()` + vectorized `.map()` instead of `collect()` to handle 1.8M users × 10 recommendations efficiently.
7. **Write** — SQLite (`local` mode) or PostgreSQL via psycopg2 `COPY` (`emr` mode). `COPY` is used instead of `INSERT` for the full dataset because it is orders of magnitude faster for millions of rows.

**Hyperparameter tuning (optional):**

Pass `--cv` to enable a 2-fold `CrossValidator` grid search over `rank ∈ {10, 20}` × `regParam ∈ {0.01, 0.1}`. Use this on EMR with the full dataset.

**Usage:**

```bash
# Local
spark-submit ml/als_train.py --input data/processed/ --mode local

# With tuning
spark-submit ml/als_train.py --input data/processed/ --mode local --cv

# EMR
python pipeline/submit_emr.py als
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to `processed/` directory |
| `--mode` | `local` | `local` or `emr` |
| `--rank` | `20` | Number of latent factors |
| `--max-iter` | `10` | ALS iterations |
| `--reg-param` | `0.1` | L2 regularization |
| `--cv` | off | Enable CrossValidator grid search |

**Target metric:** RMSE < 1.5 on 20% held-out test set.

---

### `sentiment.py` — Sentiment Classifier

Trains a TF-IDF + Logistic Regression binary classifier on review text and writes per-grid-cell positive review counts to the database.

**Labeling scheme:**

| Stars | Label |
|---|---|
| ≥ 4 | positive (1) |
| ≤ 2 | negative (0) |
| 3 | excluded (neutral — ambiguous signal) |

**Two-stage output:**

1. **Classifier evaluation** — Trains on up to 200K labeled reviews (80/20 split, `random_state=42`, stratified). Reports accuracy, weighted F1, and confusion matrix. Runs only in `local` mode (sklearn does not run on YARN workers).
2. **Grid sentiment aggregation** — Counts positive and negative reviews per `grid_cell`, computes `sentiment_score = positive / (positive + negative)`. Runs in both modes: pandas in `local`, PySpark in `emr`.

**scikit-learn pipeline:**

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
    ('clf',   LogisticRegression(max_iter=1000))
])
```

Unigrams and bigrams (ngram_range=(1,2)) capture phrase-level signals like "not good" that unigrams miss.

**Database write strategy:** SQLite uses SQLAlchemy `to_sql`; PostgreSQL uses psycopg2 `COPY` with tab-separated CSV in a `StringIO` buffer — avoids pandas/SQLAlchemy version conflicts present on EMR Python 3.7.

**Usage:**

```bash
# Local
spark-submit ml/sentiment.py --input data/processed/ --mode local

# EMR
python pipeline/submit_emr.py sentiment
```

**Target metric:** Weighted F1 ≥ 0.80.

---

### `nlp_index.py` — Business Profile Builder

Builds a `business_profiles` table that combines Yelp category tags with real review text snippets into a single searchable `profile_text` field. This table powers the natural-language restaurant search in the dashboard.

**Profile construction:**

For each business:
1. Collect up to 20 review text snippets, each truncated to the first 120 characters.
2. Concatenate the categories string three times (to give category terms 3× the TF-IDF weight of review words).
3. Append all snippets.

```
profile_text = "{cats} {cats} {cats} {snippet_1} {snippet_2} ... {snippet_20}"
```

The resulting text is stored in the `business_profiles` table and vectorized at query time by the dashboard's `_tfidf_for_city()` function.

**Fallback behavior:** If `business_profiles` is not populated for a city, the dashboard falls back to `business_scores.categories` (with 3× repetition) so NLP search works everywhere without requiring the full indexing job.

**Usage:**

```bash
# Local
spark-submit ml/nlp_index.py --input data/processed/ --mode local

# EMR
python pipeline/submit_emr.py nlp
```

---

### `evaluate.py` — Standalone Evaluation

Provides importable evaluation functions and a standalone CLI for running evaluation passes against any processed dataset or seeded database, without retraining from scratch.

**Functions:**

| Function | Description |
|---|---|
| `evaluate_als_rmse(input_path)` | Trains a minimal ALS model locally and returns test RMSE |
| `evaluate_sentiment_f1(input_path)` | Trains TF-IDF + LR classifier locally and returns `(accuracy, f1)` |
| `precision_at_k(db_url, k=10, threshold=4.0)` | Mean precision@k across all users in `als_recommendations`. Considers a recommendation "relevant" if `predicted_rating ≥ threshold`. Offline optimistic metric — useful for comparing model versions. |

**Usage:**

```bash
python ml/evaluate.py --mode all --input data/processed/
python ml/evaluate.py --mode als --input data/processed/
python ml/evaluate.py --mode sentiment --input data/processed/
python ml/evaluate.py --mode als --db sqlite:///data/citybite_local.db --k 10
```

---

### `seed_local_db.py` — Local DB Seeding

Person B's local dev shortcut. Reads processed Parquet with PySpark, computes `business_scores` and `grid_aggregates` using the same formula as `aggregate_job.py`, and writes both tables to `data/citybite_local.db` (SQLite).

Run this once after `clean_job.py` produces `data/processed/reviews_enriched/`. The Streamlit dashboard auto-detects the SQLite file when `RDS_HOST` is not set in the environment.

**Usage:**

```bash
python ml/seed_local_db.py --input data/processed/
```

**Idempotent:** Uses `if_exists="replace"` — safe to re-run after pipeline changes.

---

## 10. Database Schema

The schema is defined in `infra/schema.sql` and mirrored in `ml/seed_local_db.py` for SQLite. All tables use the same column names across PostgreSQL and SQLite so dashboard queries work unchanged in both environments.

### `business_scores`

```sql
CREATE TABLE business_scores (
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
    recency_score    FLOAT,
    popularity_score FLOAT,
    last_updated     TIMESTAMP DEFAULT NOW()
);
```

### `grid_aggregates`

```sql
CREATE TABLE grid_aggregates (
    grid_cell         VARCHAR(20) PRIMARY KEY,
    metro_area        VARCHAR(100),
    center_lat        FLOAT,
    center_lng        FLOAT,
    avg_popularity    FLOAT,
    restaurant_count  INT,
    top_cuisine       TEXT
);
```

### `als_recommendations`

```sql
CREATE TABLE als_recommendations (
    user_id          VARCHAR(50),
    business_id      VARCHAR(50),
    predicted_rating FLOAT,
    rank             INT,
    PRIMARY KEY (user_id, business_id)
);
```

### `grid_sentiment`

```sql
CREATE TABLE grid_sentiment (
    grid_cell       VARCHAR(20) PRIMARY KEY,
    sentiment_score FLOAT,
    positive_count  INT,
    negative_count  INT,
    last_updated    TIMESTAMP DEFAULT NOW()
);
```

### `business_profiles`

```sql
CREATE TABLE business_profiles (
    business_id      VARCHAR(50) PRIMARY KEY,
    name             TEXT,
    metro_area       VARCHAR(100),
    city             VARCHAR(100),
    latitude         FLOAT,
    longitude        FLOAT,
    avg_rating       FLOAT,
    review_count     INT,
    popularity_score FLOAT,
    profile_text     TEXT
);
```

### Indexes

```sql
CREATE INDEX idx_business_city     ON business_scores(city);
CREATE INDEX idx_business_metro    ON business_scores(metro_area);
CREATE INDEX idx_grid_metro        ON grid_aggregates(metro_area);
CREATE INDEX idx_als_user          ON als_recommendations(user_id);
CREATE INDEX idx_als_business      ON als_recommendations(business_id);
CREATE INDEX idx_profiles_metro    ON business_profiles(metro_area);
```

---

## 11. Dashboard

`dashboard/app.py` is a Streamlit application served on EC2. It auto-detects the database backend: if `RDS_HOST` is set in the environment it connects to PostgreSQL; otherwise it falls back to `data/citybite_local.db` (SQLite).

### Database connection

```python
@st.cache_resource
def get_engine():
    rds_host = os.environ.get("RDS_HOST", "").strip()
    if rds_host:
        url = f"postgresql+psycopg2://{user}:{pw}@{rds_host}:{port}/{db}"
    else:
        url = f"sqlite:///{_LOCAL_DB}"
    return create_engine(url)
```

All data-loading functions are decorated with `@st.cache_data(ttl=3600)` so the database is queried at most once per hour per unique input combination.

### Tab 1 — Map

A Folium heatmap showing restaurant popularity across the selected city.

**Key components:**

- **HeatMap layer** — `avg_popularity` values are normalized to `[0, 1]` and plotted using a blue-yellow-red gradient (`0.0: "#313695"` → `1.0: "#a50026"`).
- **MarkerCluster layer** — Top-N restaurant pins (configurable via slider, default 50). Pin color reflects normalized popularity: green (popular) / orange (moderate) / gray (quiet). Popups show name, cuisine, star rating, review count, and score.
- **Invisible rectangles** — Transparent `folium.Rectangle` objects sit on top of the heatmap to provide hover tooltips and click popups for each grid cell without obscuring the visual layer.
- **Neighborhood naming** — The top 10 grid cells by restaurant count are reverse-geocoded via Nominatim (geopy) in parallel using `ThreadPoolExecutor`. All other cells receive a compass-direction label (`"North District"`, `"Downtown"`, etc.) derived from their bearing and distance from the city centroid. Nominatim results are cached in a module-level dictionary for the server lifetime.
- **Caching strategy** — `build_map_html()` renders the Folium map to an HTML string and caches it with `@st.cache_data`. `components.html()` receives the same string on every rerun when inputs are unchanged, preventing the iframe from being recreated.

**Controls:**
- Cuisine filter — populated dynamically from `business_scores.categories` for the selected city; filtered against an allowlist of 80+ recognized cuisine/establishment types.
- Pins on map slider — 10–500, step 10.

### Tab 2 — Top Picks For You

ALS-powered recommendations in two input modes:

**Cuisine Taste mode:** User selects up to 3 cuisines. The `find_proxy_user()` function queries `als_recommendations` joined with `business_scores` to find the user whose top picks most overlap with the selected cuisines in the selected city (ranked by match count, then average predicted rating). That user's top-10 recommendations are displayed.

**Yelp User ID mode:** User pastes their Yelp user ID directly. Recommendations are filtered to the selected city; falls back to cross-city results if no city-specific recommendations exist.

Each recommendation card shows: name, star display, review count, cuisine, city, and an ALS match-score progress bar.

### Tab 3 — Describe What You Want

Natural-language restaurant search using TF-IDF cosine similarity.

**Algorithm:**

```
final_score = 0.60 × tfidf_cosine_similarity
            + 0.25 × log1p(review_count) / max_log_count
            + 0.15 × avg_rating / 5.0
```

The TF-IDF vectorizer is fitted once per city via `_tfidf_for_city()` (`@st.cache_resource` — lives for the server lifetime, no serialization overhead). Results with `match_score < 0.05` are filtered out.

**Vectorizer config:** `max_features=20_000`, `ngram_range=(1, 2)`, `min_df=1`, `sublinear_tf=True`.

### Tab 4 — Citywide Neighborhood Sentiment

Per-neighborhood diner satisfaction scores derived from `grid_sentiment`, with Bayesian shrinkage to prevent low-volume neighborhoods from dominating the ranking.

**Bayesian adjustment formula:**

```
satisfaction = (positive_count + k × global_rate) / (positive_count + negative_count + k) × 10
```

where `global_rate` = citywide positive review fraction and `k = 30` pseudo-counts. Results are displayed in a `st.dataframe` with a `ProgressColumn` showing satisfaction on a 0–10 scale. Neighborhood names come from `add_neighborhood_labels()` and match the map exactly.

### Running the dashboard

```bash
# Local
streamlit run dashboard/app.py

# Production (EC2)
nohup streamlit run dashboard/app.py --server.port 8501 &
```

---

## 12. Testing

The test suite uses `pytest` with `moto` for AWS service mocking. All 139 tests run locally without AWS credentials and without a running Spark cluster.

```bash
pytest tests/ -q
```

**Test modules:**

| File | Tests | Description |
|---|---|---|
| `tests/test_als_train.py` | ALS pipeline | Matrix construction, training, RMSE evaluation, recommendation generation, DB write |
| `tests/test_clean_job.py` | PySpark cleaning | Null dropping, `is_open` filter, category filtering, metro clustering, grid cell assignment |
| `tests/test_sentiment.py` | Sentiment classifier | Label assignment, grid aggregation, F1 evaluation |
| `tests/test_upload.py` | S3 upload | Multipart threshold, key construction, size verification (moto S3) |
| `tests/test_submit_emr.py` | EMR submission | Step building, cluster launch, step polling (moto EMR) |

**Results (sample dataset):** 139 passed in ~21.75 s.

---

## 13. Local Development Workflow

No AWS account required. All computation runs in PySpark local mode against `data/sample/` and `data/citybite_local.db`.

```bash
# 1. Clean and enrich the sample data
spark-submit pipeline/clean_job.py \
    --input data/sample/ --output data/processed/ --mode local

# 2. Seed SQLite with business_scores + grid_aggregates
python ml/seed_local_db.py --input data/processed/

# 3. Train ALS and write recommendations to SQLite
spark-submit ml/als_train.py --input data/processed/ --mode local

# 4. Train sentiment classifier and write grid_sentiment to SQLite
spark-submit ml/sentiment.py --input data/processed/ --mode local

# 5. (Optional) Build NLP profiles for natural-language search
spark-submit ml/nlp_index.py --input data/processed/ --mode local

# 6. Launch dashboard
streamlit run dashboard/app.py

# 7. Run tests
pytest tests/ -q

# 8. Run the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## 14. Production Workflow (AWS)

```bash
# Step 1 — Upload raw Yelp data to S3
python pipeline/upload.py --source data/raw/ --bucket citybite --prefix raw/

# Step 2 — Run the cleaning job on EMR (transient cluster, auto-terminates)
python pipeline/submit_emr.py clean

# Step 3 — Run aggregation on EMR (needs PostgreSQL JDBC driver; handled internally)
python pipeline/submit_emr.py aggregate

# Step 4 — Run all ML jobs (chain on a single transient cluster)
python pipeline/submit_emr.py setup als sentiment nlp

# Step 5 — Launch the Streamlit dashboard on EC2
nohup streamlit run dashboard/app.py --server.port 8501 &
```

### Nightly cron schedule

Set up on the EMR master node via `infra/cron_setup.sh`:

```bash
# Run cleaning + aggregation at 2 AM daily
0 2 * * * cd /home/hadoop && \
    python submit_emr.py clean && \
    python submit_emr.py aggregate
```

---

## 15. Common Issues and Fixes

### PySpark can't find Java

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### Windows: Spark local write fails

`clean_job.py` raises a clear error if `HADOOP_HOME` is not set or if `winutils.exe` / `hadoop.dll` are missing. Set `HADOOP_HOME` to a directory containing both files under `bin/`, sourced from the same Hadoop 3.x build.

### EMR job fails with S3 permissions error

Ensure `EMR_EC2_DefaultRole` instance profile has `AmazonS3FullAccess` attached. Re-attach in IAM console if missing.

### RDS connection refused from EMR

Add an inbound security group rule on the RDS instance: TCP port 5432, source = the EMR master node's security group.

### ALS returns NaN predictions

Expected for users or businesses not seen in the training split. `coldStartStrategy="drop"` is set by default, which drops those rows from evaluation rather than producing NaN RMSE. No action needed.

### ALS RMSE is high (≥ 1.5) on sample data

The sample dataset has only 300 businesses and 3,000 reviews. The sparse user-item matrix leaves most pairs unseen in training. RMSE improves substantially at full scale (7M+ reviews). Use `--cv` for grid search on the full dataset.

### Folium map is blank in Streamlit

The dashboard uses `components.html()` (not `st_folium`) to render the map HTML string. If the map is blank, check the browser console for CORS errors — this can happen when serving over a non-standard port without `--server.enableCORS false`.

### Streamlit: "No runtime found, using MemoryCacheStorageManager"

This warning appears when Streamlit functions are imported outside a running Streamlit server (e.g., in the notebook). It is informational only — caching falls back to in-memory storage and all functions work correctly.

### Parquet reads are slow on the full dataset

Add a city/metro filter before any aggregation to trigger partition pruning:

```python
df.filter(col("metro_area") == selected_city)
```

Without this, Spark scans all partitions regardless of the query.

### psycopg2 COPY fails with "invalid byte sequence"

Review text may contain tab characters, newlines, or backslashes that corrupt the COPY stream. `nlp_index.py` sanitizes all text columns before writing:

```python
out[col] = out[col].str.replace("\\", "", regex=False) \
                   .str.replace("\t", " ", regex=False) \
                   .str.replace("\n", " ", regex=False)
```

If you see this error from `als_train.py` or `sentiment.py`, apply the same sanitization to the affected column.

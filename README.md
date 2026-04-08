# CityBite

Restaurant popularity intelligence platform built on AWS. Processes 7M+ Yelp reviews through a two-zone S3 data lake on Amazon EMR, trains a Spark MLlib ALS recommender and scikit-learn sentiment classifier, and serves results through a Streamlit + Folium dashboard.

## Architecture

![CityBite Architecture](assets/big-data-project-arch.png)

---

## Repo Structure

```
citybite/
├── pipeline/
│   ├── upload.py          # Boto3 multipart upload -> S3 raw zone
│   ├── clean_job.py       # PySpark: raw JSON -> enriched reviews (partitioned by city)
│   ├── aggregate_job.py   # Spark SQL: popularity scores + grid aggregates -> RDS
│   └── submit_emr.py      # Submit jobs to EMR (transient or persistent cluster)
├── ml/
│   ├── als_train.py       # Spark MLlib ALS recommender (top-10 recs per user -> RDS)
│   ├── sentiment.py       # scikit-learn TF-IDF + LogisticRegression -> RDS
│   └── evaluate.py        # RMSE, precision@k evaluation helpers
├── dashboard/
│   └── app.py             # Streamlit app: Folium heatmap + rec panel + sentiment chart
├── infra/
│   ├── bootstrap.sh       # EMR bootstrap: pip-installs numpy/pandas/sqlalchemy/psycopg2
│   ├── create_rds.py      # Provision RDS PostgreSQL (db.t3.micro)
│   ├── schema.sql         # 4-table PostgreSQL schema
│   └── cron_setup.sh      # Nightly cron on EMR master node
├── data/
│   └── sample/
│       └── generate_sample.py  # Synthetic Yelp data for local dev
├── downloaded_data/            # Real Yelp JSON files (gitignored, ~9 GB)
├── tests/
│   └── test_sentiment.py  # 20 unit tests for sentiment pipeline (no AWS needed)
├── notebooks/
│   └── analysis.ipynb     # ALS + sentiment analysis notebook
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.10+
- Java 11 (required for PySpark local mode)
- AWS CLI configured (`aws configure`)

```bash
# Verify Java
java -version   # must be 11 or 17

# Windows: set JAVA_HOME if spark-submit can't find it
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11...
```

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # fill in your AWS credentials and RDS endpoint
```

### `.env` reference

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=citybite-560580021963

# Set to a running persistent cluster ID to submit steps to it.
# Comment out (or remove) to use auto-terminating transient clusters instead.
# EMR_CLUSTER_ID=j-XXXXXXXXXXXX

RDS_HOST=your-rds-endpoint.rds.amazonaws.com
RDS_PORT=5432
RDS_DB=citybite
RDS_USER=citybite_admin
RDS_PASSWORD=...
```

---

## Local Development (no AWS required)

### 1. Generate sample data

```bash
python data/sample/generate_sample.py
```

Creates synthetic Yelp-format JSON in `data/sample/` (~300 businesses, 3 000 reviews).

### 2. Run the cleaning pipeline locally

```bash
spark-submit pipeline/clean_job.py \
  --input  data/sample/ \
  --output data/processed/ \
  --mode   local
```

### 3. Run the aggregation pipeline locally

```bash
spark-submit pipeline/aggregate_job.py \
  --input  data/processed/ \
  --output data/gold/ \
  --mode   local
```

### 4. Run the ML models locally

```bash
# ALS recommender (writes to data/citybite_local.db)
spark-submit ml/als_train.py --input data/processed/ --mode local

# Sentiment classifier (trains sklearn model, writes grid scores to local DB)
spark-submit ml/sentiment.py --input data/processed/ --mode local
```

Expected results: ALS RMSE < 1.5, sentiment F1 > 0.80.

### 5. Launch the dashboard locally

```bash
streamlit run dashboard/app.py
```

The dashboard auto-detects `RDS_HOST` in `.env`. If it is not set it falls back to `data/citybite_local.db` (SQLite), so it works entirely offline after step 3.

Open `http://localhost:8501` in your browser.

### 6. Run unit tests

```bash
pytest tests/ -v
```

---

## Downloading the Full Yelp Dataset

Required for the AWS pipeline. The zip (~4 GB download, ~9 GB uncompressed) contains several folders; only `yelp_dataset/` is needed.

```bash
# 1. Download
curl -L -o Yelp-JSON.zip https://business.yelp.com/external-assets/files/Yelp-JSON.zip

# 2. Extract only the dataset folder
unzip Yelp-JSON.zip 'yelp_dataset/*'

# 3. Move into downloaded_data/ and clean up
mkdir -p downloaded_data
mv yelp_dataset/* downloaded_data/
rm -rf yelp_dataset Yelp-JSON.zip
```

After extraction:

```
downloaded_data/
├── yelp_academic_dataset_business.json   (~120 MB)
├── yelp_academic_dataset_review.json     (~6.5 GB)
├── yelp_academic_dataset_user.json       (~3.3 GB)
└── yelp_academic_dataset_checkin.json
```

> `downloaded_data/` is gitignored — never commit it.

---

## AWS Pipeline

### Step 1 — Create S3 bucket

```bash
aws s3 mb s3://citybite-560580021963 --region us-east-1
```

### Step 2 — Upload Yelp data to S3

```bash
python pipeline/upload.py \
  --source downloaded_data/ \
  --bucket citybite-560580021963 \
  --prefix raw/
```

### Step 3 — Provision RDS

```bash
python infra/create_rds.py       # prints the RDS endpoint when done
# Copy the endpoint into .env as RDS_HOST, then apply schema:
psql -h $RDS_HOST -U $RDS_USER -d $RDS_DB -f infra/schema.sql
```

### Step 4 — IAM permissions

The IAM user running `submit_emr.py` needs:

- `AmazonEMRFullAccess` — to create / manage clusters and submit steps
- `AmazonS3FullAccess` — to read/write data and upload scripts
- `AmazonRDSFullAccess` — for the EC2 instance profile on EMR nodes

Attach these in the AWS Console: IAM > Users > your user > Add permissions.

### Step 5 — Run the cleaning and aggregation pipeline

```bash
# Spins up a transient cluster (spot core nodes, auto-terminates, ~$2-4)
python pipeline/submit_emr.py clean aggregate
```

Logs land at `s3://citybite-560580021963/logs/emr/<cluster-id>/`.

### Step 6 — Train ML models on EMR

```bash
# Transient cluster (recommended — cheapest, ~$2-4 per run)
python pipeline/submit_emr.py als sentiment

# Persistent / manually-created cluster
python pipeline/submit_emr.py setup   --cluster-id j-XXXXXXXXXXXX --wait
python pipeline/submit_emr.py als sentiment --cluster-id j-XXXXXXXXXXXX
```

The `setup` step must run first on any cluster that was **not** launched by `submit_emr.py` (e.g. created via the AWS console), because those clusters lack the bootstrap action that installs `numpy`, `pandas`, `sqlalchemy`, and `psycopg2-binary`.

Available jobs: `clean`, `aggregate`, `als`, `sentiment`, `setup`.

### Step 7 — Monitor progress

```bash
# One-shot status table
aws emr list-steps \
  --cluster-id j-XXXXXXXXXXXX \
  --query "Steps[*].[Name,Status.State]" \
  --output table

# PowerShell live poll (every 60 s)
while ($true) {
  aws emr list-steps --cluster-id j-XXXXXXXXXXXX \
    --query "Steps[*].[Name,Status.State]" --output table
  Start-Sleep 60
}
```

### Step 8 — Read step logs on Windows

```bash
aws s3 cp s3://citybite-560580021963/<cluster-id>/steps/<step-id>/stdout.gz stdout.gz
.venv\Scripts\python -c "import gzip; print(gzip.open('stdout.gz').read().decode())"
del stdout.gz
```

---

## S3 Zone Layout

```
s3://citybite-560580021963/
├── raw/
│   ├── business/yelp_academic_dataset_business.json
│   ├── review/yelp_academic_dataset_review.json
│   └── user/yelp_academic_dataset_user.json
├── processed/
│   ├── reviews_enriched/city=Phoenix/
│   ├── business_scores/city=Phoenix/
│   └── grid_aggregates/city=Phoenix/
├── scripts/        <- pipeline + ML scripts, auto-uploaded by submit_emr.py
└── logs/emr/       <- EMR step logs (only for transient clusters)
```

---

## Database Schema

| Table | Key columns | Written by |
|---|---|---|
| `business_scores` | `business_id`, `popularity_score`, `grid_cell` | `aggregate_job.py` |
| `grid_aggregates` | `grid_cell`, `avg_popularity`, `restaurant_count` | `aggregate_job.py` |
| `als_recommendations` | `user_id`, `business_id`, `predicted_rating`, `rank` | `als_train.py` |
| `grid_sentiment` | `grid_cell`, `sentiment_score`, `positive_count` | `sentiment.py` |

---

## Dashboard Features

- **Popularity heatmap** — 0.1° x 0.1° grid squares colored by weighted popularity score (CartoDB Positron basemap, loads automatically)
- **Cuisine filter** — sidebar dropdown derived live from the selected city's business data
- **Personalized recommendations** — enter a Yelp `user_id` to see top-10 ALS picks pinned on the map as numbered blue markers; city-filtered first, falls back to cross-city
- **Neighborhood labels** — grid cells labeled by compass direction from city center (e.g. "North District", "Downtown", "Southeast Side") throughout the map, sentiment chart, and tables
- **Sentiment chart** — bar chart of positive-review rate per neighborhood for the selected city

---

## Cost Estimate (3-week class project)

| Service | Config | Estimated cost |
|---|---|---|
| S3 | ~12 GB stored + requests | ~$0.50 |
| RDS | db.t3.micro, free tier | $0 (or ~$2/demo week) |
| EMR | Transient, spot core nodes, ~5 runs | ~$10-15 |
| EC2 dashboard | t3.micro (demo week only) | ~$2 |
| **Total** | | **~$12-20** |

> Transient cluster mode (`KeepJobFlowAliveWhenNoSteps=False`) prevents runaway charges from forgetting to terminate a cluster.

---

## Common Issues

**`ModuleNotFoundError: No module named 'dotenv'` on EMR**
Expected — `python-dotenv` is not installed on EMR nodes. The import is wrapped in `try/except` in all ML scripts; RDS credentials are passed via the cluster's `Configurations` block instead.

**EMR step fails in under 10 seconds with "Unknown Error"**
The cluster is missing the bootstrap action (usually a manually-created cluster). Run the `setup` job first:
```bash
python pipeline/submit_emr.py setup --cluster-id j-XXXXXXXXXXXX --wait
```

**`AccessDeniedException: not authorized to perform elasticmapreduce:RunJobFlow`**
The IAM user lacks permission to create clusters. Go to IAM > Users > your user > Add permissions > attach `AmazonEMRFullAccess`.

**RDS connection refused from EMR**
Add an inbound rule to the RDS security group allowing TCP port 5432 from the EMR master node's security group.

**ALS RMSE >= 1.5**
Re-run with `--cv` to enable cross-validated hyperparameter search:
```bash
python pipeline/submit_emr.py als --cluster-id j-XXXXXXXXXXXX
# edit als job config to pass --cv, then resubmit
```

**Folium map tiles blank in Streamlit**
`st_folium` intercepts tile-layer requests and only activates them after user interaction. The dashboard uses `components.html(m._repr_html_(), ...)` instead, which embeds the map as a self-contained iframe — tiles load immediately with no clicks required.

**PySpark can't find Java (local)**
```bash
# Linux / macOS
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Windows (PowerShell)
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-11.x.x-hotspot"
```

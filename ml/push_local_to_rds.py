"""
Copy ML output tables from local SQLite → RDS PostgreSQL.

Run this after:
    spark-submit ml/als_train.py --input data/processed/ --mode local
    spark-submit ml/sentiment.py  --input data/processed/ --mode local

Those commands write als_recommendations and grid_sentiment to the local
SQLite file.  This script reads those two tables and upserts them to RDS
so the live dashboard reflects the ML results.

Usage:
    python ml/push_local_to_rds.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_LOCAL_DB = os.path.join(_PROJECT_ROOT, "data", "citybite_local.db")

TABLES = ["als_recommendations", "grid_sentiment", "business_profiles"]


def main() -> None:
    for var in ("RDS_HOST", "RDS_DB", "RDS_USER", "RDS_PASSWORD"):
        if not os.environ.get(var):
            raise SystemExit(f"Missing env var: {var}. Check your .env file.")

    local_engine = create_engine(f"sqlite:///{_LOCAL_DB}")
    rds_url = (
        f"postgresql+psycopg2://{os.environ['RDS_USER']}:{os.environ['RDS_PASSWORD']}"
        f"@{os.environ['RDS_HOST']}:{os.environ.get('RDS_PORT', '5432')}/{os.environ['RDS_DB']}"
    )
    rds_engine = create_engine(rds_url)

    for table in TABLES:
        print(f"Copying {table} ...")
        df = pd.read_sql(f"SELECT * FROM {table}", local_engine)
        if df.empty:
            print(f"  WARNING: {table} is empty in local SQLite — did the job finish?")
            continue
        df.to_sql(table, rds_engine, if_exists="replace", index=False,
                  method="multi", chunksize=2000)
        print(f"  {len(df):,} rows → RDS {table}")

    print("Done.")


if __name__ == "__main__":
    main()

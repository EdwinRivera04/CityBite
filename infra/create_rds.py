"""
Provision a cost-minimized RDS PostgreSQL instance for CityBite.

Uses db.t3.micro (free-tier eligible) in the default VPC.
Prints the endpoint so you can update .env.

Usage:
    python infra/create_rds.py
    python infra/create_rds.py --instance-id citybite-dev --dry-run
"""

import argparse
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Cost-minimized defaults (free-tier eligible)
DEFAULTS = {
    "instance_class": "db.t3.micro",
    "engine": "postgres",
    "engine_version": "15.7",
    "allocated_storage": 20,        # GB (minimum)
    "backup_retention": 1,          # day (minimum to stay free)
    "multi_az": False,              # single-AZ saves ~50%
    "publicly_accessible": True,    # needed for local dev access
}


def build_rds_client() -> boto3.client:
    return boto3.client(
        "rds",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def instance_exists(rds: boto3.client, instance_id: str) -> bool:
    try:
        rds.describe_db_instances(DBInstanceIdentifier=instance_id)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "DBInstanceNotFound":
            return False
        raise


def create_instance(rds: boto3.client, instance_id: str, db_name: str,
                    username: str, password: str) -> dict:
    return rds.create_db_instance(
        DBInstanceIdentifier=instance_id,
        DBName=db_name,
        DBInstanceClass=DEFAULTS["instance_class"],
        Engine=DEFAULTS["engine"],
        EngineVersion=DEFAULTS["engine_version"],
        MasterUsername=username,
        MasterUserPassword=password,
        AllocatedStorage=DEFAULTS["allocated_storage"],
        BackupRetentionPeriod=DEFAULTS["backup_retention"],
        MultiAZ=DEFAULTS["multi_az"],
        PubliclyAccessible=DEFAULTS["publicly_accessible"],
        StorageType="gp2",
        Tags=[
            {"Key": "Project", "Value": "CityBite"},
            {"Key": "Environment", "Value": "dev"},
        ],
    )


def wait_for_available(rds: boto3.client, instance_id: str, timeout: int = 600) -> str:
    print(f"Waiting for {instance_id} to become available (up to {timeout}s) ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = rds.describe_db_instances(DBInstanceIdentifier=instance_id)
        instance = resp["DBInstances"][0]
        status = instance["DBInstanceStatus"]
        print(f"  Status: {status}")
        if status == "available":
            return instance["Endpoint"]["Address"]
        time.sleep(15)
    raise TimeoutError(f"Instance {instance_id} did not become available within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision RDS PostgreSQL for CityBite")
    parser.add_argument("--instance-id", default="citybite-dev")
    parser.add_argument("--db-name", default=os.getenv("RDS_DB", "citybite"))
    parser.add_argument("--username", default=os.getenv("RDS_USER", "citybite_user"))
    parser.add_argument("--password", default=os.getenv("RDS_PASSWORD"))
    parser.add_argument("--dry-run", action="store_true", help="Print config without creating")
    args = parser.parse_args()

    if not args.password:
        print("ERROR: Set RDS_PASSWORD in .env or pass --password", file=sys.stderr)
        sys.exit(1)

    config = {**DEFAULTS, "instance_id": args.instance_id, "db_name": args.db_name, "username": args.username}
    print("RDS config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\nDry run — no instance created.")
        return

    rds = build_rds_client()

    if instance_exists(rds, args.instance_id):
        print(f"\nInstance {args.instance_id} already exists.")
        resp = rds.describe_db_instances(DBInstanceIdentifier=args.instance_id)
        endpoint = resp["DBInstances"][0].get("Endpoint", {}).get("Address", "not yet available")
    else:
        print(f"\nCreating RDS instance {args.instance_id} ...")
        create_instance(rds, args.instance_id, args.db_name, args.username, args.password)
        endpoint = wait_for_available(rds, args.instance_id)

    print(f"\nEndpoint: {endpoint}")
    print(f"\nAdd to .env:\n  RDS_HOST={endpoint}")
    print("\nApply schema:\n  psql -h {endpoint} -U {args.username} -d {args.db_name} -f infra/schema.sql")


if __name__ == "__main__":
    main()

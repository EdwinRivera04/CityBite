"""
Upload local Yelp JSON files to S3 raw zone with multipart support.

Usage:
    python pipeline/upload.py --source data/raw/ --bucket citybite --prefix raw/
    python pipeline/upload.py --source data/sample/ --bucket citybite --prefix raw/  # dev
"""

import argparse
import math
import os
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

MULTIPART_THRESHOLD_MB = 100
MB = 1024 * 1024


def build_s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _transfer_config() -> TransferConfig:
    return TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD_MB * MB,
        multipart_chunksize=MULTIPART_THRESHOLD_MB * MB,
        max_concurrency=4,
        use_threads=True,
    )


def _s3_key(source_dir: Path, file_path: Path, prefix: str) -> str:
    relative = file_path.relative_to(source_dir)
    return f"{prefix.rstrip('/')}/{relative}"


def verify_upload(s3: boto3.client, bucket: str, key: str, local_size: int) -> bool:
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        return response["ContentLength"] == local_size
    except ClientError:
        return False


def upload_file(s3: boto3.client, local_path: Path, bucket: str, key: str) -> bool:
    size_mb = local_path.stat().st_size / MB
    mode = "multipart" if size_mb >= MULTIPART_THRESHOLD_MB else "single"
    print(f"  Uploading {local_path.name} ({size_mb:.1f} MB) [{mode}] → s3://{bucket}/{key}")

    try:
        s3.upload_file(
            str(local_path),
            bucket,
            key,
            Config=_transfer_config(),
        )
    except ClientError as e:
        print(f"  ERROR uploading {local_path.name}: {e}", file=sys.stderr)
        return False

    ok = verify_upload(s3, bucket, key, local_path.stat().st_size)
    status = "OK" if ok else "SIZE MISMATCH"
    print(f"  Verified {local_path.name}: {status}")
    return ok


def upload_directory(source_dir: Path, bucket: str, prefix: str) -> tuple[int, int]:
    s3 = build_s3_client()
    files = [f for f in source_dir.rglob("*.json") if f.is_file()]

    if not files:
        print(f"No JSON files found in {source_dir}")
        return 0, 0

    print(f"Found {len(files)} file(s) in {source_dir}")
    success, failed = 0, 0

    for file_path in sorted(files):
        key = _s3_key(source_dir, file_path, prefix)
        if upload_file(s3, file_path, bucket, key):
            success += 1
        else:
            failed += 1

    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Yelp data to S3 raw zone")
    parser.add_argument("--source", required=True, help="Local source directory")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "citybite-560580021963"))
    parser.add_argument("--prefix", default="raw/", help="S3 key prefix")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: {source} is not a directory", file=sys.stderr)
        sys.exit(1)

    success, failed = upload_directory(source, args.bucket, args.prefix)
    print(f"\nUpload complete: {success} succeeded, {failed} failed.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

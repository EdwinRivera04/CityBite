"""
Unit tests for pipeline/upload.py using moto S3 mock.

No real AWS calls are made.
"""

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_s3

from pipeline.upload import (
    _s3_key,
    build_s3_client,
    upload_directory,
    verify_upload,
)

BUCKET = "test-citybite"
REGION = "us-east-1"


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_REGION", REGION)


@pytest.fixture
def s3_bucket(aws_credentials):
    with mock_s3():
        conn = boto3.client("s3", region_name=REGION)
        conn.create_bucket(Bucket=BUCKET)
        yield conn


@pytest.fixture
def sample_json_files(tmp_path):
    data = [{"business_id": "b1", "name": "Test Restaurant"}]
    file1 = tmp_path / "yelp_academic_dataset_business.json"
    file2 = tmp_path / "yelp_academic_dataset_review.json"
    for f in (file1, file2):
        f.write_text("\n".join(json.dumps(r) for r in data))
    return tmp_path


class TestS3Key:
    def test_basic_key(self, tmp_path):
        file = tmp_path / "test.json"
        key = _s3_key(tmp_path, file, "raw/")
        assert key == "raw/test.json"

    def test_prefix_trailing_slash_normalized(self, tmp_path):
        file = tmp_path / "test.json"
        key = _s3_key(tmp_path, file, "raw")
        assert key == "raw/test.json"

    def test_nested_file(self, tmp_path):
        subdir = tmp_path / "business"
        subdir.mkdir()
        file = subdir / "business.json"
        key = _s3_key(tmp_path, file, "raw/")
        assert key == "raw/business/business.json"


class TestVerifyUpload:
    def test_returns_true_on_size_match(self, s3_bucket, tmp_path):
        content = b"hello world"
        local = tmp_path / "test.json"
        local.write_bytes(content)
        s3_bucket.put_object(Bucket=BUCKET, Key="raw/test.json", Body=content)

        from pipeline.upload import verify_upload
        assert verify_upload(s3_bucket, BUCKET, "raw/test.json", len(content)) is True

    def test_returns_false_on_missing_key(self, s3_bucket, tmp_path):
        assert verify_upload(s3_bucket, BUCKET, "raw/missing.json", 100) is False

    def test_returns_false_on_size_mismatch(self, s3_bucket, tmp_path):
        s3_bucket.put_object(Bucket=BUCKET, Key="raw/test.json", Body=b"hello")
        assert verify_upload(s3_bucket, BUCKET, "raw/test.json", 999) is False


class TestUploadDirectory:
    def test_uploads_all_json_files(self, s3_bucket, sample_json_files, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", BUCKET)

        with mock_s3():
            conn = boto3.client("s3", region_name=REGION)
            conn.create_bucket(Bucket=BUCKET)

            # Patch build_s3_client to return mocked client
            import pipeline.upload as upload_mod
            monkeypatch.setattr(upload_mod, "build_s3_client", lambda: conn)

            success, failed = upload_directory(sample_json_files, BUCKET, "raw/")

        assert failed == 0
        assert success == 2

    def test_returns_zero_on_empty_dir(self, tmp_path, monkeypatch):
        import pipeline.upload as upload_mod
        monkeypatch.setattr(upload_mod, "build_s3_client", lambda: None)
        success, failed = upload_directory(tmp_path, BUCKET, "raw/")
        assert success == 0 and failed == 0

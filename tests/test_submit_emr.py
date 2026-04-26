"""
Unit tests for pipeline/submit_emr.py using moto EMR/S3 mocks.

No real AWS calls are made.
"""

import os
import pytest
import boto3
from moto import mock_s3
from unittest.mock import MagicMock, patch

REGION = "us-east-1"
BUCKET = "test-citybite"


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.setenv("S3_BUCKET", BUCKET)


@pytest.fixture
def s3_with_bucket(aws_credentials):
    with mock_s3():
        conn = boto3.client("s3", region_name=REGION)
        conn.create_bucket(Bucket=BUCKET)
        yield conn


class TestBuildStep:
    def test_clean_step_has_correct_script(self, aws_credentials):
        from pipeline.submit_emr import _build_step
        step = _build_step("clean")
        args = step["HadoopJarStep"]["Args"]
        assert any("clean_job.py" in a for a in args)
        assert step["Name"] == "CityBite-Clean"

    def test_aggregate_step_has_correct_script(self, aws_credentials):
        from pipeline.submit_emr import _build_step
        step = _build_step("aggregate")
        args = step["HadoopJarStep"]["Args"]
        assert any("aggregate_job.py" in a for a in args)

    def test_transient_step_terminates_cluster_on_failure(self, aws_credentials):
        from pipeline.submit_emr import _build_step
        step = _build_step("clean")
        assert step["ActionOnFailure"] == "TERMINATE_CLUSTER"


class TestTransientCluster:
    def _make_emr_mock(self, cluster_id: str = "j-TESTCLUSTER") -> MagicMock:
        emr = MagicMock()
        emr.run_job_flow.return_value = {"JobFlowId": cluster_id}
        return emr

    def test_launches_and_returns_cluster_id(self, aws_credentials):
        from pipeline.submit_emr import launch_transient_cluster
        emr = self._make_emr_mock("j-ABC123")
        result = launch_transient_cluster(emr, ["clean"])
        assert result == "j-ABC123"
        emr.run_job_flow.assert_called_once()

    def test_keep_alive_is_false(self, aws_credentials):
        from pipeline.submit_emr import launch_transient_cluster
        emr = self._make_emr_mock()
        launch_transient_cluster(emr, ["clean"])
        call_kwargs = emr.run_job_flow.call_args[1]
        assert call_kwargs["Instances"]["KeepJobFlowAliveWhenNoSteps"] is False

    def test_chains_multiple_jobs_as_steps(self, aws_credentials):
        from pipeline.submit_emr import launch_transient_cluster
        emr = self._make_emr_mock()
        launch_transient_cluster(emr, ["clean", "aggregate"])
        call_kwargs = emr.run_job_flow.call_args[1]
        assert len(call_kwargs["Steps"]) == 2

    def test_core_nodes_use_spot(self, aws_credentials):
        from pipeline.submit_emr import launch_transient_cluster
        emr = self._make_emr_mock()
        launch_transient_cluster(emr, ["clean"])
        call_kwargs = emr.run_job_flow.call_args[1]
        instance_groups = call_kwargs["Instances"]["InstanceGroups"]
        core = next(g for g in instance_groups if g["InstanceRole"] == "CORE")
        assert core["Market"] == "SPOT"


class TestUploadScripts:
    def test_skips_missing_local_file(self, aws_credentials, capsys, tmp_path, monkeypatch):
        # chdir to an empty temp dir so all relative script paths are missing
        monkeypatch.chdir(tmp_path)
        with mock_s3():
            conn = boto3.client("s3", region_name=REGION)
            conn.create_bucket(Bucket=BUCKET)
            from pipeline.submit_emr import upload_scripts
            upload_scripts(["aggregate"])
            captured = capsys.readouterr()
            assert "WARNING" in captured.out

    def test_uploads_existing_script(self, s3_with_bucket, tmp_path, monkeypatch):
        fake_script = tmp_path / "clean_job.py"
        fake_script.write_text("# fake")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "clean_job.py").write_text("# fake")

        with mock_s3():
            conn = boto3.client("s3", region_name=REGION)
            conn.create_bucket(Bucket=BUCKET)
            import pipeline.submit_emr as mod
            monkeypatch.setattr(mod, "S3_BUCKET", BUCKET)

            from pipeline.submit_emr import upload_scripts
            upload_scripts(["clean"])  # should not raise


class TestBuildStepEdgeCases:
    def test_unknown_job_raises_key_error(self, aws_credentials):
        from pipeline.submit_emr import _build_step
        with pytest.raises(KeyError):
            _build_step("bogus_job_that_does_not_exist")

    def test_inject_env_includes_rds_host(self, aws_credentials, monkeypatch):
        monkeypatch.setenv("RDS_HOST", "test-host.rds.amazonaws.com")
        monkeypatch.setenv("RDS_PASSWORD", "secret")
        from pipeline.submit_emr import _build_step
        step = _build_step("als", inject_env=True)
        bash_cmd = step["HadoopJarStep"]["Args"][-1]
        assert "RDS_HOST" in bash_cmd
        assert "test-host.rds.amazonaws.com" in bash_cmd

    def test_setup_job_uses_bash(self, aws_credentials):
        from pipeline.submit_emr import _build_step
        step = _build_step("setup")
        assert step["HadoopJarStep"]["Args"][0] == "bash"

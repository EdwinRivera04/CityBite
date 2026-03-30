"""
Submit a CityBite pipeline job on EMR.

Two modes:
  --transient (default) — spins up a new cluster, runs the job, terminates automatically.
                          Core nodes use spot instances (~70% cheaper).
  --cluster-id <id>     — submits a step to an existing long-running cluster.

Usage:
    # Recommended: transient cluster (cheapest, ~$2-4/run)
    python pipeline/submit_emr.py --job clean
    python pipeline/submit_emr.py --job clean aggregate   # chain both jobs

    # Existing cluster
    python pipeline/submit_emr.py --job clean --cluster-id j-XXXX --wait
"""

import argparse
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "citybite-560580021963")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SCRIPT_BASE = f"s3://{S3_BUCKET}/scripts"

# Cost-minimized cluster config: 1 on-demand master + 2 spot core nodes
CLUSTER_CONFIG = {
    "release": "emr-6.15.0",
    "master_type": "m5.xlarge",
    "core_type": "m5.xlarge",
    "core_count": 2,
    "log_uri": f"s3://{S3_BUCKET}/logs/emr/",
}

JOB_CONFIGS: dict[str, dict] = {
    "clean": {
        "name": "CityBite-Clean",
        "script": f"{SCRIPT_BASE}/clean_job.py",
        "args": [
            "--input", f"s3://{S3_BUCKET}/raw/",
            "--output", f"s3://{S3_BUCKET}/processed/reviews_enriched/",
            "--mode", "emr",
        ],
    },
    "aggregate": {
        "name": "CityBite-Aggregate",
        "script": f"{SCRIPT_BASE}/aggregate_job.py",
        "args": [
            "--input", f"s3://{S3_BUCKET}/processed/reviews_enriched/",
            "--output", f"s3://{S3_BUCKET}/processed/",
            "--mode", "emr",
        ],
    },
}


def build_emr_client() -> boto3.client:
    return boto3.client(
        "emr",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def upload_scripts(jobs: list[str]) -> None:
    local_map = {"clean": "pipeline/clean_job.py", "aggregate": "pipeline/aggregate_job.py"}
    s3 = boto3.client("s3", region_name=AWS_REGION)
    for job in jobs:
        local_path = local_map[job]
        if not os.path.exists(local_path):
            print(f"WARNING: {local_path} not found locally — using existing S3 version.")
            continue
        s3_key = f"scripts/{local_path.split('/')[-1]}"
        print(f"  Uploading {local_path} → s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(local_path, S3_BUCKET, s3_key)


def _build_step(job: str) -> dict:
    config = JOB_CONFIGS[job]
    return {
        "Name": config["name"],
        "ActionOnFailure": "TERMINATE_CLUSTER",  # fail fast on transient clusters
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": ["spark-submit", "--deploy-mode", "cluster", config["script"]] + config["args"],
        },
    }


def launch_transient_cluster(emr: boto3.client, jobs: list[str]) -> str:
    """Create a cluster that auto-terminates when all steps finish."""
    steps = [_build_step(job) for job in jobs]
    job_names = "+".join(jobs)

    resp = emr.run_job_flow(
        Name=f"citybite-{job_names}-{int(time.time())}",
        ReleaseLabel=CLUSTER_CONFIG["release"],
        Applications=[{"Name": "Spark"}],
        LogUri=CLUSTER_CONFIG["log_uri"],

        Instances={
            "KeepJobFlowAliveWhenNoSteps": False,   # ← auto-terminate when done
            "TerminationProtected": False,
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "Market": "ON_DEMAND",           # master stays on-demand (stable)
                    "InstanceRole": "MASTER",
                    "InstanceType": CLUSTER_CONFIG["master_type"],
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core",
                    "Market": "SPOT",                # core nodes use spot (~70% cheaper)
                    "InstanceRole": "CORE",
                    "InstanceType": CLUSTER_CONFIG["core_type"],
                    "InstanceCount": CLUSTER_CONFIG["core_count"],
                    "BidPrice": "0.10",              # max bid: $0.10/hr per node (on-demand ~$0.19)
                },
            ],
        },

        Steps=steps,
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        VisibleToAllUsers=True,
        Tags=[
            {"Key": "Project", "Value": "CityBite"},
            {"Key": "Mode", "Value": "transient"},
        ],
    )

    cluster_id = resp["JobFlowId"]
    print(f"Launched transient cluster {cluster_id} with {len(steps)} step(s): {job_names}")
    print(f"Cluster will auto-terminate when all steps finish.")
    return cluster_id


def add_step_to_cluster(emr: boto3.client, cluster_id: str, job: str) -> str:
    step = _build_step(job)
    step["ActionOnFailure"] = "CONTINUE"  # don't kill a persistent cluster on failure
    resp = emr.add_job_flow_steps(JobFlowId=cluster_id, Steps=[step])
    step_id = resp["StepIds"][0]
    print(f"Submitted step {step_id} ({JOB_CONFIGS[job]['name']}) to cluster {cluster_id}")
    return step_id


def wait_for_cluster(emr: boto3.client, cluster_id: str, timeout: int = 7200) -> str:
    print(f"Waiting for cluster {cluster_id} to finish (up to {timeout // 60}m) ...")
    deadline = time.time() + timeout
    terminal = {"TERMINATED", "TERMINATED_WITH_ERRORS", "WAITING"}
    while time.time() < deadline:
        resp = emr.describe_cluster(ClusterId=cluster_id)
        state = resp["Cluster"]["Status"]["State"]
        print(f"  Cluster state: {state}")
        if state in terminal:
            return state
        time.sleep(30)
    raise TimeoutError(f"Cluster {cluster_id} did not finish within {timeout}s")


def wait_for_step(emr: boto3.client, cluster_id: str, step_id: str, timeout: int = 3600) -> str:
    print(f"Waiting for step {step_id} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = emr.describe_step(ClusterId=cluster_id, StepId=step_id)
        state = resp["Step"]["Status"]["State"]
        print(f"  Step state: {state}")
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            return state
        time.sleep(30)
    raise TimeoutError(f"Step {step_id} did not finish within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CityBite pipeline on EMR")
    parser.add_argument(
        "jobs", nargs="*",
        help="Job(s) to run in order, e.g. clean aggregate. Defaults to --job flag.",
    )
    parser.add_argument("--job", choices=list(JOB_CONFIGS.keys()),
                        help="Single job (alternative to positional args)")
    parser.add_argument("--cluster-id", default=os.getenv("EMR_CLUSTER_ID", ""),
                        help="Existing cluster ID. Omit to launch a transient cluster (default).")
    parser.add_argument("--wait", action="store_true",
                        help="Block until job(s) finish. Always true for transient clusters.")
    args = parser.parse_args()

    jobs = args.jobs or ([args.job] if args.job else [])
    if not jobs:
        parser.error("Specify at least one job: clean or aggregate")

    invalid = [j for j in jobs if j not in JOB_CONFIGS]
    if invalid:
        parser.error(f"Unknown job(s): {invalid}. Choose from: {list(JOB_CONFIGS.keys())}")

    print(f"Uploading scripts for: {jobs}")
    upload_scripts(jobs)

    emr = build_emr_client()

    if args.cluster_id:
        # Persistent cluster mode
        step_ids = [add_step_to_cluster(emr, args.cluster_id, job) for job in jobs]
        if args.wait:
            for step_id in step_ids:
                state = wait_for_step(emr, args.cluster_id, step_id)
                if state != "COMPLETED":
                    print(f"Step {step_id} ended: {state}", file=sys.stderr)
                    sys.exit(1)
        print("Done.")
    else:
        # Transient cluster mode (default — cheapest)
        cluster_id = launch_transient_cluster(emr, jobs)
        final_state = wait_for_cluster(emr, cluster_id)
        if final_state == "TERMINATED_WITH_ERRORS":
            print(f"Cluster {cluster_id} terminated with errors.", file=sys.stderr)
            print(f"Check logs: s3://{S3_BUCKET}/logs/emr/{cluster_id}/", file=sys.stderr)
            sys.exit(1)
        print(f"Cluster {cluster_id} finished: {final_state}")


if __name__ == "__main__":
    main()

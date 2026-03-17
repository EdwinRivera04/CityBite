#!/usr/bin/env bash
# Install nightly cron jobs on the EMR master node.
# Run this script once after SSHing into the master node.
#
# Usage:
#   ssh hadoop@<master-dns> 'bash -s' < infra/cron_setup.sh

set -euo pipefail

SCRIPT_DIR="/home/hadoop"
LOG_DIR="/home/hadoop/logs"

mkdir -p "$LOG_DIR"

# Runs both jobs on a single transient cluster (1 cluster spin-up instead of 2)
CRON_JOB="0 2 * * * cd $SCRIPT_DIR && \
  python $SCRIPT_DIR/pipeline/submit_emr.py clean aggregate \
  >> $LOG_DIR/pipeline_\$(date +\%Y\%m\%d).log 2>&1"

# Add job only if not already present
(crontab -l 2>/dev/null | grep -qF "submit_emr.py") \
  && echo "Cron job already exists — skipping." \
  || (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "Current crontab:"
crontab -l

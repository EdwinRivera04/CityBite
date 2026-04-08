#!/bin/bash
# Bootstrap action: install Python dependencies on all EMR nodes.
# Installs via both pip3 and python3 -m pip because EMR 7.x (AL2023) can have
# pip3 and PYSPARK_PYTHON resolve to different Python installations.
set -e
sudo pip3 install numpy pandas sqlalchemy psycopg2-binary
sudo python3 -m pip install numpy pandas sqlalchemy psycopg2-binary 2>/dev/null || true

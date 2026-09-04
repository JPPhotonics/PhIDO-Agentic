#!/usr/bin/env bash
# Detached, resumable launcher for the K=3 topology-correctness re-run.
# Survives harness reaping via setsid (see below). Re-invoking simply resumes:
# the driver tops each prompt back up to K and skips completed reps.
#   setsid nohup bash _run_correctness_k3.sh > results/_correctness_k3.log 2>&1 < /dev/null &
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_K="${E2_K:-3}"
export E2_MODEL="${E2_MODEL:-o3-mini}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _score_correctness.py

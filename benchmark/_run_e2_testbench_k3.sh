#!/usr/bin/env bash
# Detached, resumable launcher for the K=3 powered E2-funnel testbench re-run
# (post-context-fix; all reps from committed code ea482e1). Survives harness
# reaping via setsid. Re-invoking simply resumes: the driver tops each prompt
# back up to K and skips completed reps.
#   setsid nohup bash _run_e2_testbench_k3.sh > results/e2_testbench_k3.log 2>&1 < /dev/null &
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export K_REPEATS="${K_REPEATS:-3}"
export E2_MODEL="${E2_MODEL:-o3-mini}"
export E2_OUTPUT_DIR="${E2_OUTPUT_DIR:-/home/tony/PhIDOv1/wt-graphrag/benchmark/results/e2_k3_artifacts}"
mkdir -p "$E2_OUTPUT_DIR"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u run_e2_testbench.py

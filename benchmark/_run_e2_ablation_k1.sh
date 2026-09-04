#!/usr/bin/env bash
set -euo pipefail
cd /home/tony/PhIDOv1/wt-graphrag/benchmark
export K_REPEATS=1
export N_PER_LEVEL=5
export E2_MODEL=o3-mini
export RIGID_MODEL=o1
export E2_OUTPUT_DIR=results/e2_ablation_artifacts
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag/benchmark
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python run_e2_ablation.py

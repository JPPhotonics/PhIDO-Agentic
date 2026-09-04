#!/usr/bin/env bash
set -a; source /home/tony/PhIDOv1/wt-graphrag/.env; set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _probe_l3.py

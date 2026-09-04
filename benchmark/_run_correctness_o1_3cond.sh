#!/usr/bin/env bash
# Detached, resumable launcher for the o1 THREE-CONDITION topology-correctness run:
#   baseline | agentic (gate ON) | agentic_nogate (enforcement OFF).
# Both arms run on o1 (E2_MODEL), correctness = topology F1 vs b3_gold, matched model.
# Survives harness reaping via setsid. Re-invoking resumes: the driver tops each prompt
# back up to K and skips completed reps (incremental JSON at E2_CORRECTNESS_OUT).
#   setsid nohup bash _run_correctness_o1_3cond.sh \
#     > results/_correctness_o1_3cond.log 2>&1 < /dev/null &
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_K="${E2_K:-3}"
export E2_MODEL="${E2_MODEL:-o1}"
export RIGID_MODEL="${RIGID_MODEL:-o1}"
export E2_CORRECTNESS_OUT="${E2_CORRECTNESS_OUT:-results/_correctness_o1_3cond.json}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _score_correctness.py

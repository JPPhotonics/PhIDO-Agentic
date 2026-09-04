#!/usr/bin/env bash
# Detached, resumable launcher for the gpt-5.4 THREE-CONDITION topology-correctness FULL TEST on
# the v2 gold set (12 L3 + 12 L4, paper Table-1 axis; L1/L2 dropped):
#   baseline | agentic (gate ON) | agentic_nogate (enforcement OFF).
# All arms run on gpt-5.4 (E2_MODEL), model-MATCHED, correctness = topology F1 vs b3_gold_v2.
# gpt-5.4 uses OPENAI_API_KEY from .env (no Claude Code OAuth / no Max-5x quota). Survives harness
# reaping via setsid; re-invoking resumes (tops each prompt back up to K, skips completed reps).
#   setsid nohup bash _run_correctness_gpt54_v2.sh \
#     > results/_correctness_gpt54_v2.log 2>&1 < /dev/null &
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_K="${E2_K:-3}"
export E2_MODEL="${E2_MODEL:-gpt-5.4}"
export RIGID_MODEL="${RIGID_MODEL:-gpt-5.4}"
# v2 gold + paired prompts (12 L3 + 12 L4). Override both together to run a different set.
export E2_GOLD="${E2_GOLD:-b3_gold_v2.json}"
export E2_PROMPTS="${E2_PROMPTS:-e2_prompts_v2.json}"
export PHIDO_HCOLLAPSE="${PHIDO_HCOLLAPSE:-1}"
export E2_CORRECTNESS_OUT="${E2_CORRECTNESS_OUT:-results/_correctness_gpt54_v2.json}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _score_correctness.py

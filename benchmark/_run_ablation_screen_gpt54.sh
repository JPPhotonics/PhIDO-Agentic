#!/usr/bin/env bash
# TINY screening pilot: run the FULL feature-ablation grid (11 arms) on the CORRECTNESS metric
# (topology F1 vs v2 gold) at K=1 over a 6-prompt balanced subset (3 L3 + 3 L4, cheaper L4s only),
# to eliminate arms that don't help before committing to the expensive full run. gpt-5.4, all arms
# enforcement-OFF (selection-gate confound held constant). Detached + resumable (tops each cell to
# K, skips done). ~66 runs.
#   setsid nohup bash _run_ablation_screen_gpt54.sh \
#     > results/_ablation_screen_gpt54.log 2>&1 < /dev/null &
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_ARMS=ablation
export E2_K="${E2_K:-1}"
export E2_MODEL="${E2_MODEL:-gpt-5.4}"
export RIGID_MODEL="${RIGID_MODEL:-gpt-5.4}"
export E2_GOLD="${E2_GOLD:-b3_gold_v2.json}"
export E2_PROMPTS="${E2_PROMPTS:-e2_prompts_v2.json}"
export PHIDO_HCOLLAPSE="${PHIDO_HCOLLAPSE:-1}"
# 3 L3 (reused + new + Benes) + 3 L4 (crossbar-16, 1x32 tree-31, Clements-28) — skip the
# 112/120/131-node monsters to keep the screen cheap; they re-enter only at the full run.
export E2_PROMPT_SUBSET="${E2_PROMPT_SUBSET:-L3_01,L3_10,L3_12,L4_09,L4_11,L4_05}"
export E2_CORRECTNESS_OUT="${E2_CORRECTNESS_OUT:-results/_ablation_screen_gpt54.json}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _score_correctness.py

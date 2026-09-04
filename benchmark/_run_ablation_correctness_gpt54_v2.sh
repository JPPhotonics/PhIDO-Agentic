#!/usr/bin/env bash
# Detached, resumable launcher for the gpt-5.4 FULL 9-ARM ABLATION on v2 gold (12 L3 + 12 L4).
# Arms (E2_ARMS=ablation): baseline + 8 agentic — base_agentic, full(=kg+gate+critic, app default),
# full_minus_{kg,gate,critic}, base_plus_{kg,gate,critic}. Routing DROPPED (held 0, no routing arms).
# All agentic arms run enforcement-OFF so the selection-gate confound is held constant while
# attributing kg/gate/critic. All arms on gpt-5.4 (OPENAI_API_KEY from .env), model-matched.
# Correctness = topology F1 vs b3_gold_v2 (score on edgeF1/GED — compF1 is saturated/degenerate).
# Survives harness reaping via setsid; re-invoking resumes (tops each prompt to K, skips done reps).
#   setsid nohup bash _run_ablation_correctness_gpt54_v2.sh \
#     > results/_ablation_correctness_gpt54_v2.log 2>&1 < /dev/null &
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_ARMS="ablation"
export E2_K="${E2_K:-3}"
export E2_MODEL="${E2_MODEL:-gpt-5.4}"
export RIGID_MODEL="${RIGID_MODEL:-gpt-5.4}"
export E2_GOLD="${E2_GOLD:-b3_gold_v2.json}"
export E2_PROMPTS="${E2_PROMPTS:-e2_prompts_v2.json}"
export PHIDO_HCOLLAPSE="${PHIDO_HCOLLAPSE:-1}"
export E2_CORRECTNESS_OUT="${E2_CORRECTNESS_OUT:-results/_ablation_correctness_gpt54_v2.json}"
export E2_DOT_DIR="${E2_DOT_DIR:-results/_ablation_correctness_gpt54_v2_dots}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _score_correctness.py

#!/usr/bin/env bash
# Pilot: agentic arm on gpt-5.4 (OpenAI), edges+DOT saved, HCOLLAPSE scoring.
# Uses OPENAI_API_KEY from .env — no Claude Code OAuth, no Max-5x quota (unlike the Opus pilot).
#   bash benchmark/_run_pilot_gpt54.sh                       # default: 6 L3, agentic_nogate, K=1
#   PILOT_PROMPTS=L3_2 E2_K=1 bash benchmark/_run_pilot_gpt54.sh
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_MODEL="${E2_MODEL:-gpt-5.4}"
export PHIDO_HCOLLAPSE=1
# Gold + paired prompts default to the v2 set (12 L3 + 12 L4); override both to use another set.
export E2_GOLD="${E2_GOLD:-b3_gold_v2.json}"
export E2_PROMPTS="${E2_PROMPTS:-e2_prompts_v2.json}"
export E2_DOT_DIR="${E2_DOT_DIR:-results/pilot_gpt54_dot}"
# Default pilot spread over the v2 ids: reused + new L3, plus small L4 (crossbar/Benes/OPA/tree).
export PILOT_PROMPTS="${PILOT_PROMPTS:-L3_01,L3_10,L3_11,L3_12,L4_09,L4_08,L4_10,L4_11}"
export PILOT_ARM="${PILOT_ARM:-run_agentic_nogate}"
export E2_K="${E2_K:-1}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _pilot_gpt54.py

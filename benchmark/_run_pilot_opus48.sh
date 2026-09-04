#!/usr/bin/env bash
# Pilot: agentic arm on claude-opus-4-8 via Claude Code OAuth, edges+DOT saved, HCOLLAPSE scoring.
# RUN THIS IN A QUIET WINDOW with Claude Code IDLE — the Max-5x subscription quota is SHARED with
# any active Claude Code session, and a busy session starves the pilot (observed: full 429 storm).
# The OAuth access token expires (~hours); this short pilot must finish inside that window, or
# Claude Code must refresh ~/.claude/.credentials.json while it runs (the token is re-read per call).
#   bash benchmark/_run_pilot_opus48.sh            # default: L3_1, agentic_nogate, K=1
#   PILOT_PROMPTS=L3_1,L3_2,L3_3,L3_4,L3_5,L3_6 PILOT_ARM=run_agentic_nogate E2_K=1 bash ...
set -a
source /home/tony/PhIDOv1/wt-graphrag/.env
set +a
unset ANTHROPIC_API_KEY                 # must not coexist with the OAuth token (else 400)
cd /home/tony/PhIDOv1/wt-graphrag/benchmark || exit 1
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export PHIDO_CLAUDE_CODE_AUTH=1
export E2_MODEL="${E2_MODEL:-claude-opus-4-8}"
export PHIDO_HCOLLAPSE=1
export PHIDO_ANTHROPIC_MAX_RETRIES="${PHIDO_ANTHROPIC_MAX_RETRIES:-8}"
export E2_DOT_DIR="${E2_DOT_DIR:-results/pilot_dot}"
export PILOT_PROMPTS="${PILOT_PROMPTS:-L3_1}"
export PILOT_ARM="${PILOT_ARM:-run_agentic_nogate}"
export E2_K="${E2_K:-1}"
exec /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python -u _pilot_opus48.py

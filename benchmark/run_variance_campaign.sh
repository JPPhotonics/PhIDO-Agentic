#!/usr/bin/env bash
# Seeded multi-build variance campaign — temperature as a VARIABLE.
#
# Builds (default): 3 at PRODUCTION temp (KB_BUILD_TEMPERATURE unset -> structured=0.5, call_google=0.1,
# the exact regime all prior analysis used) + 2 at SEEDED temp (KB_BUILD_TEMPERATURE=0 everywhere).
# Each build: reset KB -> rebuild -> capture KB structural stats -> run the full E1 suite (with the
# kg_embed / hybrid_embed fix arms). Per-build checkpoint dirs; a failed build is skipped (E1 not run on a
# partial KB) and the campaign continues.
#
# Answers: (1) production-regime build-to-build variance -> is the 0.319->0.167 functional swing noise?
#          (2) does temp=0 reduce that variance, and does it change KB quality vs the 0.5 builds?
#
# Usage:  bash benchmark/run_variance_campaign.sh
set -u
cd /home/tony/PhIDOv1/wt-graphrag
export CUDA_VISIBLE_DEVICES=""
ROOT=benchmark/results/variance
mkdir -p "$ROOT"
PY=.venv/bin/python

# Per-build regime: "prod" (defaults, env unset) or "temp0" (KB_BUILD_TEMPERATURE=0).
REGIMES=(prod prod prod temp0 temp0)

echo "===== VARIANCE CAMPAIGN START: ${#REGIMES[@]} builds: ${REGIMES[*]} ====="
i=0
for regime in "${REGIMES[@]}"; do
  i=$((i+1))
  BD="$ROOT/build_${i}_${regime}"; mkdir -p "$BD"
  if [ "$regime" = "temp0" ]; then export KB_BUILD_TEMPERATURE=0; else unset KB_BUILD_TEMPERATURE; fi
  echo "===== BUILD $i (${regime}, KB_BUILD_TEMPERATURE='${KB_BUILD_TEMPERATURE:-unset}'): reset ====="
  $PY reinitialize_neo4j_kb.py > "$BD/reset.log" 2>&1
  echo "===== BUILD $i (${regime}): process_papers ====="
  $PY process_papers.py > "$BD/process_papers.log" 2>&1
  if ! grep -q "Batch Processing Complete: 17/17 successful" "$BD/process_papers.log"; then
    echo "!!! BUILD $i (${regime}) FAILED: not 17/17 (see $BD/process_papers.log). Skipping E1, continuing."
    touch "$BD/FAILED"
    continue
  fi
  echo "===== BUILD $i (${regime}): KB stats ====="
  $PY benchmark/kb_stats.py "$BD/kb_stats.json" > /dev/null 2>&1
  echo "===== BUILD $i (${regime}): E1 suite ====="
  $PY benchmark/run_e1_retrieval.py benchmark/e1_queries.json          > "$BD/e1_curated.log" 2>&1
  $PY benchmark/run_e1_retrieval.py benchmark/e1_queries_testbench.json > "$BD/e1_testbench.log" 2>&1
  $PY benchmark/run_e1_stratified.py                                    > "$BD/e1_stratified.log" 2>&1
  cp -f benchmark/results/e1_retrieval_e1_queries.json           "$BD/e1_curated.json" 2>/dev/null
  cp -f benchmark/results/e1_retrieval_e1_queries_testbench.json "$BD/e1_testbench.json" 2>/dev/null
  cp -f benchmark/results/e1_stratified.json                     "$BD/e1_stratified.json" 2>/dev/null
  echo "===== BUILD $i (${regime}) COMPLETE ====="
done
echo "===== VARIANCE CAMPAIGN COMPLETE ====="

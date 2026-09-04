#!/usr/bin/env bash
# A2 KB of record — single clean PRODUCTION-regime build for the round-2 faithfulness bundles.
#
# Prod regime = KB_BUILD_TEMPERATURE UNSET (structured=0.5, call_google=0.1) — the exact regime the
# DIA 0.675 measurement / A+B fix / 0.842 projection were computed on. Mirrors the variance campaign's
# proven per-build steps (reset -> process_papers[+PDK ingest post-batch] -> structural stats), minus
# the E1 suite (retrieval is already characterized; this build is for A2 bundles).
#
# Usage:  bash benchmark/build_a2_kb.sh
set -u
cd /home/tony/PhIDOv1/wt-graphrag
export CUDA_VISIBLE_DEVICES=""
unset KB_BUILD_TEMPERATURE                 # PROD regime (defaults)
PY=.venv/bin/python
BD=benchmark/results/a2_kb_build; mkdir -p "$BD"

echo "===== A2 KB BUILD START (prod, KB_BUILD_TEMPERATURE='${KB_BUILD_TEMPERATURE:-unset}') ====="
echo "===== reset ====="
$PY reinitialize_neo4j_kb.py > "$BD/reset.log" 2>&1
echo "===== process_papers (17 papers + PDK ingest post-batch) ====="
$PY process_papers.py > "$BD/process_papers.log" 2>&1
if ! grep -q "Batch Processing Complete: 17/17 successful" "$BD/process_papers.log"; then
  echo "!!! A2 KB BUILD FAILED: not 17/17 successful (see $BD/process_papers.log)."
  touch "$BD/FAILED"
  exit 1
fi
echo "===== KB stats ====="
$PY benchmark/kb_stats.py "$BD/kb_stats.json" > /dev/null 2>&1
cat "$BD/kb_stats.json"
echo "===== A2 KB BUILD COMPLETE ====="

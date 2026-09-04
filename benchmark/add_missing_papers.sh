#!/usr/bin/env bash
# Incremental ingest of the 7 papers the A2 build #1 failed to parse (Axiomatic quota), appending
# to the live KB without re-ingesting the 10 already present. Guard active (short Axiomatic parse
# -> local PDF fallback). Liu #13 (annotator-critical) sorts first -> first claim on any quota.
set -u
cd /home/tony/PhIDOv1/wt-graphrag
export CUDA_VISIBLE_DEVICES=""
export SKIP_PDK_INGEST=1          # PDK_Cells already exist; just propagate new components onto them
unset KB_BUILD_TEMPERATURE        # prod regime
BD=benchmark/results/a2_add_missing; mkdir -p "$BD"
PY=.venv/bin/python

echo "===== ADD MISSING PAPERS START ($(ls papers_missing/*.pdf | wc -l) papers) ====="
$PY process_papers.py papers_missing 2>&1 | tee "$BD/process_papers.log"
echo "===== per-paper extraction outcome ====="
grep -E "Processing:|AxDocumentParser output length|Extracted [0-9]+ raw|No entities|local PDF text extraction" "$BD/process_papers.log"
echo "===== KB stats after ====="
$PY benchmark/kb_stats.py "$BD/kb_stats.json" 2>/dev/null && cat "$BD/kb_stats.json"
echo "===== ADD MISSING PAPERS COMPLETE ====="

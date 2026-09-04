#!/usr/bin/env bash
cd /home/tony/PhIDOv1/wt-graphrag/benchmark
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag/benchmark
PY=/home/tony/PhIDOv1/wt-graphrag/.venv/bin/python
for i in $(seq 1 200); do   # ~200 x 300s ≈ 16h ceiling
  out=$($PY run_e2_attribution.py 2>/dev/null | grep -o 'READY=[A-Za-z]*')
  echo "$(date '+%m-%d %H:%M') poll $i: $out"
  [ "$out" = "READY=True" ] && { echo "ATTRIBUTION COMPLETE -> results/e2_ablation_attribution.md"; break; }
  sleep 300
done

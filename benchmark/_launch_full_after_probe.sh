#!/bin/bash
# Wait for the probe workers (exact PIDs) to exit, then launch the full harder-12
# sweep (K=3, N=3, $50 cap) into the SAME shard files so probe reps are resume-skipped.
set -u
cd /home/tony/PhIDOv1/wt-graphrag
PROBE_PIDS="614803 614804 614805"
echo "[orch] waiting for probe pids: $PROBE_PIDS"
for PID in $PROBE_PIDS; do
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
  echo "[orch] probe pid $PID exited"
done
echo "[orch] all probe workers done; launching full harder-12 sweep (K=3, N=3, cap=\$50)"
for W in 0 1 2; do
  CUDA_VISIBLE_DEVICES="" E2_MODEL=qwen/qwen3.6-27b \
  K_REPEATS=3 N_WORKERS=3 WORKER_ID=$W \
  RUN_TIMEOUT=1800 COST_CAP_USD=50 \
  PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark \
  nohup .venv/bin/python benchmark/run_qwen_trace_ablation.py \
    > benchmark/logs/full_w$W.log 2>&1 &
  echo "[orch] launched full worker $W pid $!"
done
echo "[orch] full sweep launched at $(date)"

#!/bin/bash
# Stop the free-tier Nemotron supervisor and its workers, then start the paid-route supervisor.
# Safe to run once credits are on the OpenRouter key. Idempotent: refuses if a paid supervisor is already up.
cd /home/tony/PhIDOv1/wt-graphrag
if pgrep -f "^bash benchmark/_nemo_supervisor_paid.sh" >/dev/null; then echo "paid supervisor already running"; exit 0; fi
echo "stopping free-tier supervisor + workers"
pkill -f "benchmark/_nemo_supervisor.sh" ; sleep 1
pkill -f "run_qwen_trace_ablation.py" ; pkill -f "run_qwen_rigid_baseline.py" ; sleep 5
pgrep -fa "run_qwen_trace_ablation|run_qwen_rigid_baseline|_nemo_supervisor.sh" && { echo "workers still alive; aborting"; exit 1; }
echo "launching paid-route supervisor"
nohup bash benchmark/_nemo_supervisor_paid.sh > benchmark/logs/nemo_paid_supervisor.log 2>&1 &
sleep 2; pgrep -fa "^bash benchmark/_nemo_supervisor_paid.sh" && echo "started; log: benchmark/logs/nemo_paid_supervisor.log"

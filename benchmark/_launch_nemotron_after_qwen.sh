#!/bin/bash
# Chain: (wait for Qwen agentic sweep) -> Qwen rigid harder-12 (completes Qwen n=24
# rigid arm) -> Nemotron agentic all-24 -> Nemotron rigid all-24. Fully resumable;
# each driver skips reps already banked, so a mid-chain death just needs a relaunch.
set -u
cd /home/tony/PhIDOv1/wt-graphrag
PY=.venv/bin/python
PP=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
NEMO="nvidia/nemotron-3-ultra-550b-a55b:free"
HARD12="L3_03,L3_05,L3_06,L3_08,L3_11,L4_01,L4_02,L4_03,L4_04,L4_05,L4_06,L4_07"
ALL24="L3_01,L3_02,L3_03,L3_04,L3_05,L3_06,L3_07,L3_08,L3_09,L3_10,L3_11,L3_12,L4_01,L4_02,L4_03,L4_04,L4_05,L4_06,L4_07,L4_08,L4_09,L4_10,L4_11,L4_12"
QWEN_ORCH_PID=623832
log(){ echo "[nemo-orch $(date +%H:%M:%S)] $*"; }

# --- Phase 0: wait for the Qwen launch-orchestrator to spawn the full workers ---
log "waiting for Qwen launch-orchestrator pid $QWEN_ORCH_PID to exit"
while kill -0 "$QWEN_ORCH_PID" 2>/dev/null; do sleep 30; done
sleep 45  # let the full workers spin up before we start polling

# --- Phase 1: wait for the Qwen agentic full sweep to finish ---
log "waiting for Qwen agentic sweep (run_qwen_trace_ablation.py) to finish"
while pgrep -f run_qwen_trace_ablation.py >/dev/null 2>&1; do sleep 60; done
log "Qwen agentic sweep DONE"

# --- Phase 2: Qwen rigid on the harder 12 (easy 12 already banked; resume-skipped) ---
# If a manual rigid run is already in flight (started by hand), wait for it to finish
# first — two processes writing qwen_rigid_baseline.json would clobber each other.
# After it exits, this run is a resume (skips whatever the manual run already banked).
while pgrep -f run_qwen_rigid_baseline.py >/dev/null 2>&1; do sleep 30; done
log "Qwen rigid harder-12 starting (resume)"
CUDA_VISIBLE_DEVICES="" E2_MODEL=qwen/qwen3.6-27b RUN_TAG=qwen \
  K_REPEATS=3 PROMPT_IDS="$HARD12" RUN_TIMEOUT=1800 \
  PYTHONPATH=$PP $PY benchmark/run_qwen_rigid_baseline.py > benchmark/logs/qwen_rigid_hard.log 2>&1
log "Qwen rigid harder-12 DONE"

# --- Phase 3: Nemotron agentic, all 24 prompts, N=2 (gentle on the free-tier rate limit) ---
log "Nemotron agentic all-24 starting (N=2)"
for W in 0 1; do
  CUDA_VISIBLE_DEVICES="" E2_MODEL="$NEMO" RUN_TAG=nemotron \
    K_REPEATS=3 N_WORKERS=2 WORKER_ID=$W RUN_TIMEOUT=1800 PROMPT_IDS="$ALL24" \
    PYTHONPATH=$PP nohup $PY benchmark/run_qwen_trace_ablation.py \
      > benchmark/logs/nemo_agentic_w$W.log 2>&1 &
done
wait  # both Nemotron agentic workers
log "Nemotron agentic all-24 DONE"

# --- Phase 4: Nemotron rigid, all 24 ---
log "Nemotron rigid all-24 starting"
CUDA_VISIBLE_DEVICES="" E2_MODEL="$NEMO" RUN_TAG=nemotron \
  K_REPEATS=3 RUN_TIMEOUT=1800 \
  PYTHONPATH=$PP $PY benchmark/run_qwen_rigid_baseline.py > benchmark/logs/nemo_rigid.log 2>&1
log "Nemotron rigid all-24 DONE — full chain complete"

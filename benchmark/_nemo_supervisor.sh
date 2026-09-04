#!/bin/bash
# Unattended Nemotron scaffold-first benchmark: rigid_baseline + base_agentic, all 24
# prompts, K=3, on the free tier. Starts the shared embedding daemon, then loops
# (launch workers -> wait -> completeness check) until both arms are full. Each pass
# purges retryable-infra reps (status-aware resume) so load-shed holes self-heal.
# Shared-box courtesy: nice 19, 2 embedding threads, N=2 agentic workers, disk floor.
set -u
cd /home/tony/PhIDOv1/wt-graphrag
set -a; source .env >/dev/null 2>&1; set +a
PY=.venv/bin/python
export PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag:/home/tony/PhIDOv1/wt-graphrag/benchmark
export E2_MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"
export RUN_TAG=nemotron
export K_REPEATS=3
export RUN_TIMEOUT=7200
export EMPTY_RESPONSE_MAX_ATTEMPTS=20   # ~35 min patience per call (was 10 ≈ 13 min)
export DISK_FLOOR_GB=30
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
# NotFoundError added 2026-08-11: OpenRouter returns 404 when the free upstream provider
# is unavailable (transient outage 08-10 poisoned plus_gate/plus_critic/full) — infra, not model.
# "failed:Interpreter failed to produce DesignIntent" added 2026-08-28: the generic status that
# masked swallowed structured-output errors (repair-loop ValueError on empty/invalid content,
# provider 400s) during the 08-26/27 refill. Post-fix code emits specific statuses instead
# ("failed:LLM refused to produce DesignIntent: <exc>"), so this exact string cannot recur —
# listing it purges the legacy masked reps exactly once.
export RETRY_STATUSES="error:EmptyResponseError,error:APIConnectionError,error:InternalServerError,error:APITimeoutError,error:RateLimitError,error:NotFoundError,failed:Interpreter failed to produce DesignIntent"
ALL24="L3_01,L3_02,L3_03,L3_04,L3_05,L3_06,L3_07,L3_08,L3_09,L3_10,L3_11,L3_12,L4_01,L4_02,L4_03,L4_04,L4_05,L4_06,L4_07,L4_08,L4_09,L4_10,L4_11,L4_12"
LOG=benchmark/logs
mkdir -p "$LOG"
log(){ echo "[nemo-sup $(date +%m-%d\ %H:%M)] $*"; }

# ── embedding daemon (idempotent) ──────────────────────────────────────────
if ! curl -sf --max-time 3 http://127.0.0.1:8876/health > /dev/null 2>&1; then
  log "starting embedding daemon"
  nohup nice -n 19 $PY benchmark/_embed_daemon.py > "$LOG/embed_daemon.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf --max-time 3 http://127.0.0.1:8876/health > /dev/null 2>&1 && break
    sleep 5
  done
fi
curl -sf --max-time 3 http://127.0.0.1:8876/health || { log "daemon failed to start"; exit 1; }
echo

complete_check() {
  $PY - <<'EOF'
import json, glob, os, sys
RETRY = tuple("error:EmptyResponseError,error:APIConnectionError,error:InternalServerError,error:APITimeoutError,error:RateLimitError,error:NotFoundError,failed:Interpreter failed to produce DesignIntent".split(","))
def good(r): return not str(r.get("status","")).startswith(RETRY)
ALL24 = [f"L3_{i:02d}" for i in range(1,13)] + [f"L4_{i:02d}" for i in range(1,13)]
ARMS = ["base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]
ag = {arm: {p: 0 for p in ALL24} for arm in ARMS}
for f in glob.glob("benchmark/results/nemotron_trace_ablation_w*.json"):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items():
            ag[arm][pid] = ag[arm].get(pid, 0) + sum(good(r) for r in reps)
rg = {p: 0 for p in ALL24}
for rf in glob.glob("benchmark/results/nemotron_rigid_baseline*.json"):
    if rf.endswith(".bak") or ".pre_shard" in rf:
        continue
    for pid, reps in json.load(open(rf)).get("rigid_baseline", {}).items():
        rg[pid] = rg.get(pid, 0) + sum(good(r) for r in reps)
counts = {arm: sum(min(v, 3) for v in d.values()) for arm, d in ag.items()}
nr = sum(min(v, 3) for v in rg.values())
line = "  ".join(f"{arm.replace('base_','').replace('_baseline','')} {n}/72" for arm, n in counts.items())
print(f"rigid {nr}/72  {line}", file=sys.stderr)
sys.exit(0 if (nr >= 72 and all(n >= 72 for n in counts.values())) else 1)
EOF
}

wait_for_quota() {
  # Launch only after TWO consecutive healthy probes. Waits out: daily quota, 404 outage,
  # 5xx / "overloaded" upstream errors, and 2xx load-shed (no choices) — the 08-12 mode
  # that poisoned critic/full with silent "failed to produce DesignIntent" reps.
  HEALTHY=0
  while [ "$HEALTHY" -lt 2 ]; do
    RESP=$(curl -s --max-time 25 https://openrouter.ai/api/v1/chat/completions \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" \
      -d "{\"model\":\"$E2_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK\"}],\"max_tokens\":5}")
    if echo "$RESP" | grep -q "free-models-per-day"; then
      RESET_MS=$(echo "$RESP" | grep -oE '"X-RateLimit-Reset":"[0-9]+"' | grep -oE "[0-9]+" | head -1)
      NOW=$(date +%s); WAIT=$(( ${RESET_MS:-0} / 1000 - NOW + 120 ))
      [ "$WAIT" -lt 60 ] && WAIT=3600
      log "daily free-tier quota exhausted; sleeping $((WAIT/60)) min until reset"; HEALTHY=0; sleep "$WAIT"
    elif echo "$RESP" | grep -qE '"code": *404'; then
      log "model endpoint 404 (provider outage); sleeping 30 min"; HEALTHY=0; sleep 1800
    elif echo "$RESP" | grep -qiE '"code": *5[0-9][0-9]|overloaded|Upstream error|"code": *429'; then
      log "upstream degraded ($(echo "$RESP" | head -c 120)); sleeping 15 min"; HEALTHY=0; sleep 900
    elif ! echo "$RESP" | grep -q '"content"'; then
      log "load-shed probe (2xx without content): $(echo "$RESP" | head -c 120); sleeping 15 min"; HEALTHY=0; sleep 900
    else
      HEALTHY=$((HEALTHY+1)); log "probe healthy ($HEALTHY/2)"; [ "$HEALTHY" -lt 2 ] && sleep 60
    fi
  done
}

for PASS in $(seq 1 200); do
  if complete_check; then
    log "COMPLETE after pass $((PASS-1)) — both arms full"
    break
  fi
  wait_for_quota
  log "pass $PASS: launching workers"
  PIDS=()
  for W in 0 1; do
    PROMPT_IDS="$ALL24" N_WORKERS=2 WORKER_ID=$W \
      nohup nice -n 19 $PY benchmark/run_qwen_trace_ablation.py \
      >> "$LOG/nemo_agentic_w$W.log" 2>&1 &
    PIDS+=($!)
  done
  for RW in 0 1 2; do
    N_WORKERS=3 WORKER_ID=$RW \
      nohup nice -n 19 $PY benchmark/run_qwen_rigid_baseline.py \
      >> "$LOG/nemo_rigid_w$RW.log" 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  log "pass $PASS: workers exited"
  sleep 120
done
log "supervisor exiting"

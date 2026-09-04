#!/usr/bin/env bash
# Launch the post-hoc layout+DRC pass for one SOURCE with N_WORKERS sharded worker loops.
# Each worker re-execs its python process every CHUNK reps (LIMIT) so JAX/SAX/gdsfactory caches
# cannot grow without bound over hundreds of layouts on this memory-constrained shared box.
# Resumable: re-running this script picks up where each shard left off.
#
#   SOURCE=gpt54 N_WORKERS=3 benchmark/_launch_layout.sh
#   SOURCE=qwen  N_WORKERS=3 benchmark/_launch_layout.sh
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${SOURCE:-gpt54}"
N_WORKERS="${N_WORKERS:-3}"
CHUNK="${CHUNK:-40}"
PY="$WT/.venv-layout/bin/python"          # kfactory 0.21.7 (lock) — routing works here; .venv is pinned 0.21.1 and cannot route
LOGDIR="$WT/benchmark/results"
export PYTHONPATH="$WT:$WT/benchmark"
export SOURCE N_WORKERS
export LIMIT="$CHUNK"   # per-invocation rep cap; the loop below re-execs until the shard is exhausted

for ((w=0; w<N_WORKERS; w++)); do
  (
    export WORKER_ID=$w
    log="$LOGDIR/_layout_${SOURCE}_w${w}.log"
    while true; do
      out="$($PY "$WT/benchmark/_layout_from_dot.py" 2>&1 | tee -a "$log" | grep -v 'UserWarning\|warnings.warn')"
      # the driver prints "... N to run;" before starting and "DONE wK: n reps this invocation" after
      todo="$(printf '%s\n' "$out" | sed -n 's/.* \([0-9]\+\) to run;.*/\1/p' | head -1)"
      done_n="$(printf '%s\n' "$out" | sed -n 's/^DONE w[0-9]*: \([0-9]\+\) reps.*/\1/p' | tail -1)"
      if [[ -z "$todo" || "$todo" == "0" || -z "$done_n" || "$done_n" == "0" ]]; then
        echo "[launcher w$w] finished (todo=${todo:-?} done_n=${done_n:-?})" | tee -a "$log"
        break
      fi
    done
  ) &
  sleep 2
done
wait
echo "[launcher] all $N_WORKERS workers finished for SOURCE=$SOURCE"

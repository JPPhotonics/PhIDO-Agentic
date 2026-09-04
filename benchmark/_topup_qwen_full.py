"""Top-up the Qwen agentic sweep's under-filled cells (the hard-L4 `full`-arm gap).

The three-worker sweep left `full` at 65/72 reps: L4_01 has 1 rep, L4_04 and L4_07
have none.  Those are three of the hardest L4 prompts, so their absence flatters the
`full` arm relative to the others — the gap has to be closed before the arm is
comparable.

Two hazards this driver exists to avoid (both observed in the w0..w2 shards):

1. *Cross-shard double counting.*  Scoring merges shards by summing reps per
   (arm, prompt), so re-running a cell into a shard that does not already own it
   inflates N.  We therefore never touch w0..w2: new reps land in their own shard
   (``qwen_trace_ablation_w3.json``) and the deficit is computed from the MERGED
   count across every existing shard.

2. *Trace-filename collision.*  Trace files live in one flat namespace keyed
   ``{arm}__{pid}__rep{N}.json``, so two shards that both wrote rep1..N silently
   overwrote each other (6 cells store 4 reps but have only 3 traces).  Here the rep
   index continues from the merged count and we hard-assert the target file is free.

RUN_TIMEOUT defaults to 9000s: the sweep's slowest banked rep took 6137s (102 min) and
recorded no timeouts, so the original cap was well above the 1800s used by the rigid /
Nemotron phases.  A 1800s cap here would manufacture ``status="timeout"`` artifacts.

Run one prompt per process (they are independent) to keep wall-clock near the
critical path:

  cd /home/tony/PhIDOv1/wt-graphrag
  for P in L4_01 L4_04 L4_07; do
    CUDA_VISIBLE_DEVICES="" E2_MODEL=qwen/qwen3.6-27b RUN_TAG=qwen TOPUP_PROMPT=$P \
    PYTHONPATH=.:benchmark nohup .venv/bin/python benchmark/_topup_qwen_full.py \
      > benchmark/logs/qwen_topup_$P.log 2>&1 &
  done

Resumable: re-running skips whatever is already banked.  DRY=1 reports the plan only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("E2_MODEL", "qwen/qwen3.6-27b")
os.environ.setdefault("RUN_TAG", "qwen")
os.environ.setdefault("RUN_TIMEOUT", "9000")
os.environ["N_WORKERS"] = "1"  # keep the imported driver's own OUT path inert

ROOT = Path(__file__).resolve().parents[1]

# Importing the driver reuses its environment exactly: the SentenceTransformer
# memoization, E2_GOLD/E2_PROMPTS defaults, and run_one's timeout/usage accounting.
import run_qwen_trace_ablation as drv  # noqa: E402

ARM = os.getenv("TOPUP_ARM", "full")
K = int(os.getenv("K_REPEATS", "3"))
RESULTS = ROOT / "benchmark" / "results"
OUT = RESULTS / "qwen_trace_ablation_w3.json"
TRACE_DIR = drv.TRACE_DIR
TARGETS = os.getenv("TOPUP_PROMPT", "L4_01,L4_04,L4_07").split(",")


def merged_count(arm: str, pid: str) -> int:
    """Reps already banked for (arm, pid) across every shard, including this one."""
    n = 0
    for shard in sorted(RESULTS.glob("qwen_trace_ablation_w*.json")):
        try:
            d = json.loads(shard.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        n += len(d.get(arm, {}).get(pid, []))
    return n


def main() -> None:
    dry = os.getenv("DRY") == "1"
    for pid in TARGETS:
        have = merged_count(ARM, pid)
        need = max(0, K - have)
        print(f"[topup] {ARM}/{pid}: merged={have} target={K} -> {need} rep(s) to run",
              flush=True)
        if dry:
            for r in range(have, K):
                f = TRACE_DIR / f"{ARM}__{pid}__rep{r + 1}.json"
                print(f"         would write {f.name} (exists={f.exists()})")
            continue

        for _ in range(need):
            idx = merged_count(ARM, pid)  # re-read: another process may have banked one
            if idx >= K:
                break
            trace_f = TRACE_DIR / f"{ARM}__{pid}__rep{idx + 1}.json"
            # Never clobber a trace: that is the exact bug that cost 6 cells their traces.
            assert not trace_f.exists(), f"refusing to overwrite existing trace {trace_f}"

            rec, trace = drv.run_one(drv.GOLD[pid]["prompt"], drv.ARMS[ARM])

            trace_f.write_text(json.dumps(
                {"arm": ARM, "prompt_id": pid, "rep": idx + 1,
                 "model": drv.MODEL, "trace": trace}, indent=1, default=str))
            store = json.loads(OUT.read_text()) if OUT.exists() else {}
            store.setdefault(ARM, {}).setdefault(pid, []).append(rec)
            OUT.write_text(json.dumps(store, indent=1, default=str))
            print(f"[topup {ARM:16} {pid} rep{idx + 1}/{K}] {rec['status']:14} "
                  f"nodes={len(rec.get('nodes', {}))} edges={len(rec.get('edges', []))} "
                  f"calls={rec['n_llm_calls']} {rec['latency_s']}s "
                  f"${rec.get('cost_usd', 0):.3f}", flush=True)
    print(f"\nDONE topup {ARM} {TARGETS} -> {OUT}")


if __name__ == "__main__":
    main()

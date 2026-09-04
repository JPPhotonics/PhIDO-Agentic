"""Qwen3.6-27B (OpenRouter) downsized ablation + trace capture.

Runs the 5 AGENTIC arms of the additive family on the 12 high-gpt-5.4-success prompts, capturing
(a) the scoreable topology (nodes + typed edges + dot) for the uplift ablation, and (b) the full
per-run tool-interaction trace (mcp_servers.llm_client._TRACE_BUFFER) for the trace analysis.

rigid_baseline is DEFERRED (its llm_api structured front-end is o3-mini-pinned → needs a refactor
or OpenAI quota; tracked separately). Reasoning is enabled on Qwen via the llm_client extra_body.

Serial + resumable + per-run wall-clock cap (SIGALRM) so a runaway agentic loop can't hang the grid;
a capped run is recorded as status="timeout" (a valid did-not-converge datum).

Run (CPU embeddings REQUIRED for the kg arm — this GPU can't run the embedder):
  CUDA_VISIBLE_DEVICES="" E2_MODEL=qwen/qwen3.6-27b K_REPEATS=1 \
  PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_qwen_trace_ablation.py
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("E2_MODEL", "qwen/qwen3.6-27b")
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(str(ROOT / ".env"))
# _score_correctness loads these at import (v2 = 12 L3 + 12 L4 set); give absolute paths
os.environ.setdefault("E2_GOLD", str(ROOT / "benchmark" / "b3_gold_v2.json"))
os.environ.setdefault("E2_PROMPTS", str(ROOT / "benchmark" / "e2_prompts_v2.json"))

# Embedder: route encode() to the shared daemon (_embed_daemon.py) so this worker holds no
# resident model; falls back to a memoized local instance if the daemon is down. Must run
# BEFORE the pipeline imports SentenceTransformer. (No pipeline-source change.)
import _embed_proxy  # noqa: E402

print(f"[embed] mode: {_embed_proxy.install()}", flush=True)

import _score_correctness as SC  # noqa: E402  (sets MODEL from E2_MODEL at import)
from mcp_servers import llm_client  # noqa: E402

MODEL = os.getenv("E2_MODEL", "qwen/qwen3.6-27b")
K = int(os.getenv("K_REPEATS", "1"))
TIMEOUT = int(os.getenv("RUN_TIMEOUT", "1800"))  # 30-min per-run cap
# Process-level sharding: launch N_WORKERS processes (WORKER_ID 0..N-1); each takes a round-robin
# slice of the (arm,prompt) grid and writes its own shard. Merge shards at scoring time.
N_WORKERS = int(os.getenv("N_WORKERS", "1"))
WORKER_ID = int(os.getenv("WORKER_ID", "0"))
_suffix = f"_w{WORKER_ID}" if N_WORKERS > 1 else ""
# RUN_TAG namespaces the output files by model (default "qwen" for back-compat;
# set RUN_TAG=nemotron for the Nemotron suite so it writes to its own files).
TAG = os.getenv("RUN_TAG", "qwen")
OUT = ROOT / "benchmark" / "results" / f"{TAG}_trace_ablation{_suffix}.json"
TRACE_DIR = ROOT / "benchmark" / "results" / f"{TAG}_traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# additive family (routing held 0); feature bits (kg, gate, critic)
ARMS: dict[str, dict] = {
    "base_agentic":     dict(kg=0, gate=0, critic=0),
    "base_plus_kg":     dict(kg=1, gate=0, critic=0),
    "base_plus_gate":   dict(kg=0, gate=1, critic=0),
    "base_plus_critic": dict(kg=0, gate=0, critic=1),
    "full":             dict(kg=1, gate=1, critic=1),
}
# ARM_IDS=comma,sep restricts the grid (e.g. scaffold-first: ARM_IDS=base_agentic)
if os.getenv("ARM_IDS"):
    _want = set(os.getenv("ARM_IDS").split(","))
    ARMS = {k: v for k, v in ARMS.items() if k in _want}

# Statuses that are infrastructure faults, not model behaviour: on resume these reps are
# DROPPED and re-run rather than counted as done (fixes the status-blind-resume gotcha).
RETRYABLE = tuple((os.getenv("RETRY_STATUSES",
                             "error:EmptyResponseError,error:APIConnectionError,"
                             "error:InternalServerError,error:APITimeoutError,"
                             "error:RateLimitError")).split(","))
DISK_FLOOR_GB = float(os.getenv("DISK_FLOOR_GB", "30"))
# The easy set (high gpt-5.4 success) already ran; default is now the HARDER 12
# (the gpt-5.4-low half), which completes the full 24-prompt gold on Qwen.
# Override with PROMPT_IDS=comma,sep for a cost probe — same shard files, so the
# probe's reps are reused (resume-skipped) by the full sweep, no wasted credits.
_HARD12 = ["L3_03", "L3_05", "L3_06", "L3_08", "L3_11",
           "L4_01", "L4_02", "L4_03", "L4_04", "L4_05", "L4_06", "L4_07"]
PROMPTS = os.getenv("PROMPT_IDS", ",".join(_HARD12)).split(",")

GOLD = {e["id"]: e for e in json.load(open(ROOT / "benchmark" / "b3_gold_v2.json"))["gold"]}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def run_one(prompt: str, bits: dict) -> tuple[dict, list]:
    """One agentic run; returns (record, trace). Never raises."""
    llm_client.trace_reset()
    _c0 = dict(llm_client.USAGE)  # snapshot to attribute cost/tokens to THIS run
    t0 = time.time()
    signal.alarm(TIMEOUT)
    try:
        topo, status, dot = SC._run_agentic_cfg(
            prompt, kg=bool(bits["kg"]), gate=bool(bits["gate"]),
            critic=bool(bits["critic"]), disable_enforcement=True)
        rec = {"status": status,
               "nodes": dict(getattr(topo, "nodes", {}) or {}),
               "edges": list(getattr(topo, "edges", []) or []),
               "dot_string": dot}
    except _Timeout:
        rec = {"status": "timeout"}
    except Exception as e:  # noqa: BLE001 — keep the grid alive
        rec = {"status": f"error:{type(e).__name__}", "error": str(e)[:200]}
    finally:
        signal.alarm(0)
    trace = llm_client.trace_dump()
    rec["latency_s"] = round(time.time() - t0, 1)
    rec["n_llm_calls"] = len(trace)
    rec["cost_usd"] = round(llm_client.USAGE["cost"] - _c0["cost"], 4)
    rec["tok_in"] = llm_client.USAGE["in"] - _c0["in"]
    rec["tok_out"] = llm_client.USAGE["out"] - _c0["out"]
    return rec, trace


def main() -> None:
    # round-robin slice of the (arm, prompt) grid for this worker
    grid = [(arm, pid) for arm in ARMS for pid in PROMPTS]
    tasks = [t for i, t in enumerate(grid) if i % N_WORKERS == WORKER_ID]
    if os.getenv("DRY") == "1":
        print(f"[DRY] model={MODEL} K={K} timeout={TIMEOUT}s worker {WORKER_ID}/{N_WORKERS} "
              f"-> {len(tasks)} (arm,prompt) pairs x K={K}; OUT={OUT.name}")
        return
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    # status-aware resume: purge retryable-infra reps so they re-run instead of counting done
    purged = 0
    for arm in list(store):
        for pid in list(store[arm]):
            keep = [r for r in store[arm][pid]
                    if not str(r.get("status", "")).startswith(RETRYABLE)]
            purged += len(store[arm][pid]) - len(keep)
            store[arm][pid] = keep
    if purged:
        print(f"[w{WORKER_ID}] purged {purged} retryable-infra rep(s) for re-run", flush=True)
    # Per-worker cost ceiling = total cap / N_WORKERS (each process spends independently).
    cap = float(os.getenv("COST_CAP_USD", "0") or 0)
    share = cap / N_WORKERS if cap else 0.0
    spent = 0.0
    done_ct = 0
    for arm, pid in tasks:
        bits = ARMS[arm]
        have = len(store.get(arm, {}).get(pid, []))
        for rep in range(have, K):
            if share and spent >= share:
                print(f"[w{WORKER_ID}] COST CAP hit: spent ${spent:.2f} >= "
                      f"${share:.2f} (share of ${cap:.0f}); stopping.", flush=True)
                OUT.write_text(json.dumps(store, indent=1, default=str))
                return
            # disk guard: this shared box fills fast and ENOSPC wedges workers — pause, don't die
            import shutil as _sh
            while _sh.disk_usage("/").free < DISK_FLOOR_GB * 2**30:
                print(f"[w{WORKER_ID}] disk below {DISK_FLOOR_GB} GiB free — pausing 10 min",
                      flush=True)
                time.sleep(600)
            done_ct += 1
            rec, trace = run_one(GOLD[pid]["prompt"], bits)
            spent += rec.get("cost_usd", 0) or 0
            # next free trace index (purged reps may leave gaps; never overwrite an old trace)
            taken = {int(p.stem.rsplit("rep", 1)[1]) for p in
                     TRACE_DIR.glob(f"{arm}__{pid}__rep*.json")}
            tidx = max(taken, default=0) + 1
            rec["trace_idx"] = tidx  # explicit store<->trace join (order breaks after purges)
            (TRACE_DIR / f"{arm}__{pid}__rep{tidx}.json").write_text(
                json.dumps({"arm": arm, "prompt_id": pid, "rep": tidx,
                            "model": MODEL, "trace": trace}, indent=1, default=str))
            store.setdefault(arm, {}).setdefault(pid, []).append(rec)
            OUT.write_text(json.dumps(store, indent=1, default=str))
            print(f"[w{WORKER_ID} {arm:16} {pid} rep{rep + 1}/{K}] {rec['status']:14} "
                  f"nodes={len(rec.get('nodes', {}))} edges={len(rec.get('edges', []))} "
                  f"calls={rec['n_llm_calls']} {rec['latency_s']}s "
                  f"${rec.get('cost_usd', 0):.3f} (worker spent ${spent:.2f})", flush=True)
    print(f"\nDONE worker {WORKER_ID}/{N_WORKERS}: {len(tasks)} pairs; shard -> {OUT}")


if __name__ == "__main__":
    main()

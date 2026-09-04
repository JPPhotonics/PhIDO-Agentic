"""Qwen3.6-27B RIGID baseline (pure Qwen, no OpenAI) for the uplift-vs-baseline comparison.

Runs the ORIGINAL rigid pipeline (`baseline_runner.build_rigid_netlist`) UNMODIFIED. The only
change is a runtime monkeypatch of the two low-level `llm_api` entry points the baseline uses —
`callgpt_pydantic` (structured) and `call_llm` (text) — rerouting them to Qwen via OpenRouter with
reasoning on. No pipeline source is edited; no OpenAI endpoint is touched. Structured calls use JSON
mode + schema-in-prompt + a validation-repair loop (Qwen doesn't honour OpenAI strict `.parse`).

Writes to a SEPARATE file (`qwen_rigid_baseline.json`) so it never races the running agentic driver
(`qwen_trace_ablation.json`); the scorer merges the six arms. The rigid baseline touches no Neo4j /
embeddings, so it is safe to run concurrently with the agentic grid.

Run:
  CUDA_VISIBLE_DEVICES="" E2_MODEL=qwen/qwen3.6-27b K_REPEATS=1 \
  PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_qwen_rigid_baseline.py
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
os.environ.setdefault("E2_GOLD", str(ROOT / "benchmark" / "b3_gold_v2.json"))
os.environ.setdefault("E2_PROMPTS", str(ROOT / "benchmark" / "e2_prompts_v2.json"))

import _embed_proxy

print(f"[embed] mode: {_embed_proxy.install()}", flush=True)

from openai import OpenAI
from PhotonicsAI.Photon import llm_api

MODEL = os.getenv("E2_MODEL", "qwen/qwen3.6-27b")  # any OpenRouter slug (e.g. nvidia/nemotron-...)
_or = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))


class EmptyResponseError(RuntimeError):
    """2xx with no choices (free-tier load shed) that outlasted the patient retry."""


def _create(**kw):
    """chat.completions.create with a patient retry on provider load-shed (2xx, no choices).

    This driver bypasses llm_client (it monkeypatches llm_api directly), so it needs its
    own guard: OpenRouter :free answers load-shed with HTTP 200 and ``choices: null``.
    """
    shed = 0
    while True:
        r = _or.chat.completions.create(**kw)
        if getattr(r, "choices", None):
            return r
        shed += 1
        if shed >= int(os.getenv("EMPTY_RESPONSE_MAX_ATTEMPTS", "10")):
            raise EmptyResponseError("provider returned no choices after patient retries")
        time.sleep(min(15 * shed, 120))


# ── monkeypatch the baseline's two LLM entry points → Qwen/OpenRouter (source untouched) ──
def _qwen_structured(prompt, sys_prompt, pydantic_model):
    schema = json.dumps(pydantic_model.model_json_schema())
    msgs = [{"role": "system", "content": sys_prompt or ""},
            {"role": "user", "content": (prompt or "") +
             "\n\nRespond with ONLY a JSON object matching this schema; include every required "
             "field (use empty arrays/objects where you have no value, never omit a field):\n" + schema}]
    last = None
    for _ in range(3):
        r = _create(model=MODEL, messages=msgs,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning": {"enabled": True}})
        c = r.choices[0].message.content or ""
        try:
            return pydantic_model.model_validate_json(c)
        except Exception as e:  # noqa: BLE001 — repair loop
            last = e
            msgs += [{"role": "assistant", "content": c},
                     {"role": "user", "content":
                      f"That JSON failed validation: {str(e)[:300]}. Return corrected JSON with ALL required fields."}]
    raise ValueError(f"qwen structured failed: {last}")


def _qwen_text(prompt, sys_prompt="", llm_api_selection=MODEL):
    r = _create(
        model=MODEL,
        messages=[{"role": "system", "content": sys_prompt or ""},
                  {"role": "user", "content": prompt or ""}],
        extra_body={"reasoning": {"enabled": True}})
    return r.choices[0].message.content


llm_api.callgpt_pydantic = _qwen_structured
llm_api.call_llm = _qwen_text

import baseline_runner as B  # noqa: E402  (resolves llm_api.* at call time → sees the patches)
from _score_correctness import topo_from_dsl  # noqa: E402

# default = all 24 gold prompts (resume skips any already done, e.g. Qwen's easy 12);
# override with PROMPT_IDS=comma,sep. TAG namespaces the output per model.
_ALL24 = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
PROMPTS = os.getenv("PROMPT_IDS", ",".join(_ALL24)).split(",")
K = int(os.getenv("K_REPEATS", "1"))
TIMEOUT = int(os.getenv("RUN_TIMEOUT", "1800"))
TAG = os.getenv("RUN_TAG", "qwen")
# process-level sharding (round-robin over PROMPTS), same pattern as the agentic driver
N_WORKERS = int(os.getenv("N_WORKERS", "1"))
WORKER_ID = int(os.getenv("WORKER_ID", "0"))
if N_WORKERS > 1:
    PROMPTS = [p for i, p in enumerate(PROMPTS) if i % N_WORKERS == WORKER_ID]
_suffix = f"_w{WORKER_ID}" if N_WORKERS > 1 else ""
OUT = ROOT / "benchmark" / "results" / f"{TAG}_rigid_baseline{_suffix}.json"
GOLD = {e["id"]: e for e in json.load(open(ROOT / "benchmark" / "b3_gold_v2.json"))["gold"]}


class _TO(Exception):
    pass


def _alarm(signum, frame):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)


def run_one(prompt: str) -> dict:
    t0 = time.time()
    signal.alarm(TIMEOUT)
    try:
        topo = topo_from_dsl(B.build_rigid_netlist(prompt, MODEL))
        rec = {"status": "ok" if topo.nodes else "empty_netlist",
               "nodes": dict(getattr(topo, "nodes", {}) or {}),
               "edges": list(getattr(topo, "edges", []) or [])}
    except _TO:
        rec = {"status": "timeout"}
    except Exception as e:  # noqa: BLE001
        rec = {"status": f"error:{type(e).__name__}", "error": str(e)[:200]}
    finally:
        signal.alarm(0)
    rec["latency_s"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    retryable = tuple((os.getenv("RETRY_STATUSES",
                                 "error:EmptyResponseError,error:APIConnectionError,"
                                 "error:InternalServerError,error:APITimeoutError,"
                                 "error:RateLimitError")).split(","))
    purged = 0
    for a_ in list(store):
        for p_ in list(store[a_]):
            keep = [r for r in store[a_][p_]
                    if not str(r.get("status", "")).startswith(retryable)]
            purged += len(store[a_][p_]) - len(keep)
            store[a_][p_] = keep
    if purged:
        print(f"[rigid] purged {purged} retryable-infra rep(s) for re-run", flush=True)
    if os.getenv("DRY") == "1":
        print(f"[DRY] rigid_baseline qwen K={K} prompts={len(PROMPTS)} -> {len(PROMPTS) * K} runs")
        return
    arm = "rigid_baseline"
    for pid in PROMPTS:
        have = len(store.get(arm, {}).get(pid, []))
        for rep in range(have, K):
            import shutil as _sh
            while _sh.disk_usage("/").free < float(os.getenv("DISK_FLOOR_GB", "30")) * 2**30:
                print("[rigid] disk below floor — pausing 10 min", flush=True)
                time.sleep(600)
            rec = run_one(GOLD[pid]["prompt"])
            store.setdefault(arm, {}).setdefault(pid, []).append(rec)
            OUT.write_text(json.dumps(store, indent=1, default=str))
            print(f"[rigid_baseline {pid} rep{rep + 1}/{K}] {rec['status']:14} "
                  f"nodes={len(rec.get('nodes', {}))} edges={len(rec.get('edges', []))} "
                  f"{rec['latency_s']}s", flush=True)
    print(f"\nDONE rigid_baseline -> {OUT}")


if __name__ == "__main__":
    main()

"""Pilot E2 ablation: single-shot vs iterative builder on N Testbench prompts.

Runs each prompt through the GraphRAG pipeline twice (extraction_mode single_shot vs
iterative), topology gate ON, AR off (no AWS), KB-grounding off (Neo4j down), and scores the
staged funnel + the paired arm comparison. Incremental-saves results so a long run is
recoverable. DRY=1 just prints the selected prompts and exits.

Run:  PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_pilot_e2.py
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path("/home/tony/PhIDOv1/wt-graphrag")
sys.path.insert(0, str(ROOT))                 # mcp_servers
sys.path.insert(0, str(ROOT / "benchmark"))   # e2_runner / e2_funnel

from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))

import e2_funnel as F
import e2_runner as R

N = int(os.getenv("N_PROMPTS", "10"))
MODEL = os.getenv("PILOT_MODEL", "o3-mini")
OUT = ROOT / "benchmark" / "results"
OUT.mkdir(exist_ok=True)
OUT_JSON = OUT / "pilot_builder_ablation.json"


def load_prompts(n: int) -> list[dict]:
    rows = [r[0].strip() for r in csv.reader(open(ROOT / "Testbench_modified.csv"))
            if r and r[0].strip()]
    idxs = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]   # evenly spaced, reproducible
    return [{"id": f"tb{j}", "prompt": rows[j], "level": "?"} for j in idxs]


class NoGroundOrch(R.RealOrchestrator):
    def explore(self, prompt, model, grounding_rounds=0):
        from mcp_servers.pipeline_orchestrator import explore_and_ask
        return explore_and_ask(prompt, model, 15, 0)        # grounding off (ignores arg)


def main():
    prompts = load_prompts(N)
    if os.getenv("DRY") == "1":
        print(f"[DRY] model={MODEL}, {len(prompts)} prompts:")
        for p in prompts:
            print(f"  {p['id']}: {p['prompt'][:90]}")
        return

    common = dict(model=MODEL, max_critic_rounds=1, enable_topology_gate=True, enable_ar_gate=False)
    arms = {"single_shot": R.RunConfig(extraction_mode="single_shot", **common),
            "iterative": R.RunConfig(extraction_mode="iterative", **common)}
    results: dict[str, list] = {"single_shot": [], "iterative": []}
    orch = NoGroundOrch()

    for p in prompts:
        for arm, cfg in arms.items():
            t = time.time()
            try:
                r = R.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], arm=arm, config=cfg, orch=orch)
            except Exception as e:  # noqa: BLE001 — keep the batch alive
                r = {"prompt_id": p["id"], "level": p["level"], "arm": arm,
                     "stages": {s: False for s in R.STAGES},
                     "cost": {"latency_s": time.time() - t, "failed_at": f"exc:{type(e).__name__}"},
                     "error": str(e)[:200]}
            results[arm].append(r)
            json.dump(results, open(OUT_JSON, "w"), indent=2)        # incremental save
            print(f"[{arm:11}] {p['id']:6} stages={r['stages']} {r['cost'].get('failed_at')} "
                  f"{time.time() - t:.0f}s", flush=True)

    print("\n=== SINGLE-SHOT funnel ===")
    print(json.dumps(F.funnel_summary(results["single_shot"]), indent=2))
    print("\n=== ITERATIVE funnel ===")
    print(json.dumps(F.funnel_summary(results["iterative"]), indent=2))
    print("\n=== ABLATION: single_shot (base) vs iterative (treat) ===")
    print(json.dumps(F.compare_arms(results["single_shot"], results["iterative"]), indent=2))
    print(f"\n[ok] results -> {OUT_JSON}")


if __name__ == "__main__":
    main()

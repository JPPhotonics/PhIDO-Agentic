"""Headline E2: rigid single-pass (main) vs agentic orchestrator+MCP (graphRAG).

Runs the RIGID baseline on the same N Testbench prompts as the pilot, then pairs it against
the pilot's stored AGENTIC single-shot arm (same prompts, same model, same layout/sim/DRC tail)
and reports compare_arms — isolating the LLM-integration architecture.

Reuses benchmark/results/pilot_builder_ablation.json["single_shot"] as the agentic arm, so only
the 10 rigid runs are new. Run AFTER the pilot has produced that file.

Run: PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_baseline_vs_agentic.py
"""

from __future__ import annotations

import json
import os
import time

import baseline_runner as B
import e2_funnel as F
from run_pilot_e2 import ROOT, load_prompts

N = int(os.getenv("N_PROMPTS", "10"))
MODEL = os.getenv("PILOT_MODEL", "o3-mini")
OUT = ROOT / "benchmark" / "results"
PILOT = OUT / "pilot_builder_ablation.json"
OUT_JSON = OUT / "baseline_vs_agentic.json"


def main():
    prompts = load_prompts(N)
    if os.getenv("DRY") == "1":
        print(f"[DRY] rigid baseline, model={MODEL}, {len(prompts)} prompts; "
              f"agentic arm from {PILOT.name}")
        return

    agentic = {r["prompt_id"]: r for r in json.loads(PILOT.read_text())["single_shot"]}

    rigid = []
    for p in prompts:
        t = time.time()
        try:
            r = B.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], model=MODEL)
        except Exception as e:  # noqa: BLE001
            r = {"prompt_id": p["id"], "level": p["level"], "arm": "rigid_baseline",
                 "stages": {s: False for s in B.R.STAGES},
                 "cost": {"latency_s": time.time() - t, "failed_at": f"exc:{type(e).__name__}"},
                 "error": str(e)[:200]}
        rigid.append(r)
        json.dump({"rigid": rigid}, open(OUT_JSON, "w"), indent=2, default=str)
        print(f"[rigid] {p['id']:6} {r['stages']} {r['cost'].get('failed_at')} {time.time()-t:.0f}s", flush=True)

    # pair against the agentic single-shot arm on shared prompt ids
    agentic_arm = [agentic[r["prompt_id"]] for r in rigid if r["prompt_id"] in agentic]
    print("\n=== RIGID baseline funnel ===")
    print(json.dumps(F.funnel_summary(rigid), indent=2))
    print("\n=== AGENTIC single-shot funnel (from pilot) ===")
    print(json.dumps(F.funnel_summary(agentic_arm), indent=2))
    print("\n=== compare_arms: baseline=RIGID, treat=AGENTIC (delta = agentic - rigid) ===")
    print(json.dumps(F.compare_arms(rigid, agentic_arm), indent=2))
    print(f"\n[ok] -> {OUT_JSON}")


if __name__ == "__main__":
    main()

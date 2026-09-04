"""E2 ablation sweep: a shared baseline (everything ON) vs each toggle flipped OFF.

Reuses the pilot's prompt set + grounding-off orchestrator. Runs the baseline once, then one
arm per ablation, and reports each feature's *contribution* = (ON rate) − (OFF rate) per funnel
stage, paired (McNemar + bootstrap CI). KB-independent.

Ablations:
  topology_gate (Clingo B4), critic (infra-C), ar_gate (B5 — only if a Bedrock guardrail is set).

Select via ABLATIONS env (default "topology_gate,critic"). DRY=1 lists prompts.
Run: PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_ablation_sweep.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import e2_funnel as F
import e2_runner as R
from run_pilot_e2 import ROOT, NoGroundOrch, load_prompts

N = int(os.getenv("N_PROMPTS", "10"))
MODEL = os.getenv("PILOT_MODEL", "o3-mini")
OUT = ROOT / "benchmark" / "results"
OUT.mkdir(exist_ok=True)
OUT_JSON = OUT / "ablation_sweep.json"

BASELINE = R.RunConfig(model=MODEL, max_critic_rounds=2, enable_topology_gate=True,
                       enable_ar_gate=False, extraction_mode="single_shot")
ABLATIONS = {
    "topology_gate": {"enable_topology_gate": False},
    "critic": {"max_critic_rounds": 0},
}
# AR gate ablation only makes sense if a Bedrock guardrail is configured (else AR is a no-op).
if os.getenv("AR_GUARDRAIL_ID"):
    BASELINE = replace(BASELINE, enable_ar_gate=True)
    ABLATIONS["ar_gate"] = {"enable_ar_gate": False}

SELECTED = [a for a in os.getenv("ABLATIONS", "topology_gate,critic").split(",") if a in ABLATIONS]


def run_arm(prompts, cfg, label, orch):
    out = []
    for p in prompts:
        t = time.time()
        try:
            r = R.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], arm=label, config=cfg, orch=orch)
        except Exception as e:  # noqa: BLE001
            r = {"prompt_id": p["id"], "level": p["level"], "arm": label,
                 "stages": {s: False for s in R.STAGES},
                 "cost": {"latency_s": time.time() - t, "failed_at": f"exc:{type(e).__name__}"},
                 "error": str(e)[:200]}
        out.append(r)
        print(f"[{label:16}] {p['id']:6} {r['stages']} {r['cost'].get('failed_at')} {time.time()-t:.0f}s", flush=True)
    return out


def main():
    prompts = load_prompts(N)
    if os.getenv("DRY") == "1":
        print(f"[DRY] model={MODEL} baseline={BASELINE}")
        print(f"[DRY] ablations={SELECTED}, {len(prompts)} prompts")
        return

    orch = NoGroundOrch()
    results = {}
    print(f"baseline (all ON): {BASELINE}\nablations: {SELECTED}\n")
    baseline = run_arm(prompts, BASELINE, "baseline_on", orch)
    results["baseline_on"] = baseline
    json.dump(results, open(OUT_JSON, "w"), indent=2, default=str)

    for name in SELECTED:
        off = run_arm(prompts, replace(BASELINE, **ABLATIONS[name]), f"{name}_off", orch)
        results[f"{name}_off"] = off
        json.dump(results, open(OUT_JSON, "w"), indent=2, default=str)
        cmp = F.compare_arms(off, baseline)   # baseline_arg=OFF, graphrag_arg=ON -> delta = ON - OFF
        print(f"\n=== {name}: contribution per stage = (ON) - (OFF), paired N={cmp['n_paired']} ===")
        for s, st in cmp["by_stage"].items():
            print(f"  {s:13} ON={st['graphrag_rate']:.2f}  OFF={st['baseline_rate']:.2f}  "
                  f"Δ(ON-OFF)={st['delta']:+.2f}  McNemar p={st['mcnemar']['p_value']:.3f}")

    print(f"\n[ok] results -> {OUT_JSON}")


if __name__ == "__main__":
    main()

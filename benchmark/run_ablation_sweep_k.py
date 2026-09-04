"""Powered ablation sweep: N prompts x K repeats per arm (handles the no-seed variance).

Baseline (topology gate + critic ON) vs each toggle OFF, single-shot, AR off, grounding off.
Because the pipeline has no seed, K repeats per prompt = the "multiple seeds" — and the
flip-rate across repeats directly quantifies the run-to-run variance the pilot was confounded
by. Reports: pooled per-stage rates (N*K), per-prompt MAJORITY outcomes, paired McNemar on the
majorities, and the repeat flip-rate. Defaults N=20, K=3. Incremental-saved (multi-hour run).

Run: PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_ablation_sweep_k.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import e2_funnel as F
import e2_runner as R
from report import Reporter
from run_pilot_e2 import ROOT, NoGroundOrch, load_prompts

N = int(os.getenv("N_PROMPTS", "20"))
K = int(os.getenv("K_REPEATS", "3"))
MODEL = os.getenv("PILOT_MODEL", "o3-mini")
OUT = ROOT / "benchmark" / "results" / "ablation_sweep_k.json"

BASELINE = R.RunConfig(
    model=MODEL,
    max_critic_rounds=2,
    enable_topology_gate=True,
    enable_ar_gate=False,
    extraction_mode="single_shot",
)
ABLATIONS = {
    "topology_gate": {"enable_topology_gate": False},
    "critic": {"max_critic_rounds": 0},
}


def run_arm(cfg, label, prompts, orch, store):
    for p in prompts:
        for rep in range(K):
            t = time.time()
            try:
                r = R.run_one(
                    p["prompt"],
                    prompt_id=p["id"],
                    level=p["level"],
                    arm=label,
                    config=cfg,
                    orch=orch,
                )
            except Exception as e:  # noqa: BLE001
                r = {
                    "prompt_id": p["id"],
                    "level": p["level"],
                    "arm": label,
                    "stages": dict.fromkeys(R.STAGES, False),
                    "cost": {
                        "latency_s": time.time() - t,
                        "failed_at": f"exc:{type(e).__name__}",
                    },
                }
            store.setdefault(label, {}).setdefault(p["id"], []).append(r)
            json.dump(store, open(OUT, "w"), indent=2, default=str)
            print(
                f"[{label:16}] {p['id']:6} rep{rep + 1}/{K} {r['stages']} {time.time() - t:.0f}s",
                flush=True,
            )


def _majority(runs):
    norm = [F.normalize(r["stages"]) for r in runs]
    return {
        "prompt_id": runs[0]["prompt_id"],
        "stages": {s: sum(n[s] for n in norm) > len(norm) / 2 for s in F.STAGES},
    }


def _pooled(runs_flat):
    return {
        s: round(v["rate"], 2)
        for s, v in F.funnel_summary(runs_flat)["by_stage"].items()
    }


def main():
    prompts = load_prompts(N)
    if os.getenv("DRY") == "1":
        print(
            f"[DRY] N={N} K={K} model={MODEL}; arms: baseline_on + {list(ABLATIONS)}; "
            f"total runs = {(1 + len(ABLATIONS)) * N * K}"
        )
        return

    orch = NoGroundOrch()
    store = {}
    run_arm(BASELINE, "baseline_on", prompts, orch, store)
    for name, ov in ABLATIONS.items():
        run_arm(replace(BASELINE, **ov), f"{name}_off", prompts, orch, store)

    base_runs = [r for runs in store["baseline_on"].values() for r in runs]
    base_maj = [_majority(store["baseline_on"][pid]) for pid in store["baseline_on"]]
    rep = Reporter(
        OUT.with_suffix(".md"),
        "Powered ablation sweep (N×K, post-fix)",
        meta={
            "model": MODEL,
            "N_prompts": N,
            "K_repeats": K,
            "arms": "baseline_on + " + ", ".join(f"{a}_off" for a in ABLATIONS),
            "raw_json": OUT.name,
        },
    )

    rep.h("Pooled per-stage rates (N×K runs)")
    rep.table(
        ["arm"] + list(F.STAGES),
        [["baseline_on"] + [_pooled(base_runs).get(s) for s in F.STAGES]]
        + [
            [f"{name}_off"]
            + [
                _pooled(
                    [r for runs in store[f"{name}_off"].values() for r in runs]
                ).get(s)
                for s in F.STAGES
            ]
            for name in ABLATIONS
        ],
    )

    for name in ABLATIONS:
        off = store[f"{name}_off"]
        cmp = F.compare_arms(
            [_majority(off[pid]) for pid in off], base_maj
        )  # base=OFF, treat=ON
        rep.h(
            f"{name}: contribution (ON − OFF), per-prompt majority, N={cmp['n_paired']}"
        )
        rep.table(
            ["stage", "ON", "OFF", "Δ", "McNemar p"],
            [
                [
                    s,
                    f"{st['graphrag_rate']:.2f}",
                    f"{st['baseline_rate']:.2f}",
                    f"{st['delta']:+.2f}",
                    f"{st['mcnemar']['p_value']:.3f}",
                ]
                for s, st in cmp["by_stage"].items()
            ],
        )

    rep.h(
        "Repeat variance: fraction of prompts with NON-unanimous repeats (no-seed noise)"
    )
    vrows = []
    for label, byp in store.items():
        flips, n = dict.fromkeys(F.STAGES, 0), len(byp)
        for runs in byp.values():
            norm = [F.normalize(r["stages"]) for r in runs]
            for s in F.STAGES:
                if len({nn[s] for nn in norm}) > 1:
                    flips[s] += 1
        vrows.append([label] + [round(flips[s] / n, 2) for s in F.STAGES])
    rep.table(["arm"] + list(F.STAGES), vrows)

    rep.line(f"\n[ok] raw -> {OUT}")
    rep.save()


if __name__ == "__main__":
    main()

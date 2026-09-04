"""C — cost A/B: agentic GraphRAG vs rigid baseline, tokens + latency + rounds, metered uniformly.

Runs the SAME prompts through both arms with the OpenAI `TokenMeter` active, so token counts are
measured by one instrument on both (apples-to-apples). Reports the cost comparison AND the
cost-effectiveness number — tokens per DRC-clean design — so the agentic overhead is weighed
against the extrinsic lift it buys, not reported in isolation.

K repeats per prompt absorb the pipeline's no-seed variance (same rationale as the ablation).
This is the expensive arm of the suite (real LLM calls on both pipelines) — N defaults small.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_cost_ab.py
"""

from __future__ import annotations

import json
import os
import time

import baseline_runner as BL
import cost_aggregator as C
import e2_runner as R
from report import Reporter
from run_pilot_e2 import ROOT, NoGroundOrch, load_prompts

N = int(os.getenv("N_PROMPTS", "8"))
K = int(os.getenv("K_REPEATS", "1"))
MODEL = os.getenv("PILOT_MODEL", "o3-mini")
OUT = ROOT / "benchmark" / "results" / "cost_ab.json"

AGENTIC = R.RunConfig(
    model=MODEL,
    max_critic_rounds=2,
    enable_topology_gate=True,
    enable_ar_gate=False,
    extraction_mode="single_shot",
)


def main() -> None:
    prompts = load_prompts(N)
    if os.getenv("DRY") == "1":
        print(
            f"[DRY] N={N} K={K} model={MODEL}; arms: agentic + rigid; runs = {2 * N * K}"
        )
        return

    orch = NoGroundOrch()
    agentic, rigid, store = [], [], {"agentic": [], "rigid": []}
    for p in prompts:
        for _ in range(K):
            t = time.time()
            ra = R.run_one(
                p["prompt"],
                prompt_id=p["id"],
                level=p["level"],
                arm="agentic",
                config=AGENTIC,
                orch=orch,
            )
            agentic.append(ra)
            store["agentic"].append(ra)
            print(
                f"[agentic] {p['id']:6} {ra['cost'].get('tokens_in', 0)}+{ra['cost'].get('tokens_out', 0)}tok "
                f"drc={ra['stages'].get('drc_clean')} {time.time() - t:.0f}s",
                flush=True,
            )

            t = time.time()
            rb = BL.run_one(
                p["prompt"], prompt_id=p["id"], level=p["level"], model=MODEL
            )
            rigid.append(rb)
            store["rigid"].append(rb)
            json.dump(store, open(OUT, "w"), indent=2, default=str)
            print(
                f"[rigid  ] {p['id']:6} {rb['cost'].get('tokens_in', 0)}+{rb['cost'].get('tokens_out', 0)}tok "
                f"drc={rb['stages'].get('drc_clean')} {time.time() - t:.0f}s",
                flush=True,
            )

    cmp = C.compare(
        [C.to_cost_log(r) for r in agentic], [C.to_cost_log(r) for r in rigid]
    )
    a, lin, ratios = cmp["agentic"], cmp["linear"], cmp["ratios"]

    rep = Reporter(
        OUT.with_suffix(".md"),
        "C — cost A/B (agentic vs rigid baseline)",
        meta={
            "model": MODEL,
            "N_prompts": N,
            "K_repeats": K,
            "raw_json": OUT.name,
            "metered": "OpenAI SDK (both arms, same instrument)",
        },
    )
    rep.h("Cost per arm")
    rep.table(
        ["metric", "agentic", "rigid"],
        [
            ["runs", a["n"], lin["n"]],
            ["DRC-clean", a["n_drc_clean"], lin["n_drc_clean"]],
            ["tokens in (total)", a["tokens_in_total"], lin["tokens_in_total"]],
            ["tokens out (total)", a["tokens_out_total"], lin["tokens_out_total"]],
            [
                "tokens/run (mean)",
                round(a["tokens_per_run_mean"]),
                round(lin["tokens_per_run_mean"]),
            ],
            [
                "tokens/SUCCESS",
                round(a["tokens_per_success"], 1),
                round(lin["tokens_per_success"], 1),
            ],
            [
                "latency/run s (mean)",
                round(a["latency_s_mean"], 1),
                round(lin["latency_s_mean"], 1),
            ],
            ["rounds (mean)", round(a["rounds_mean"], 2), round(lin["rounds_mean"], 2)],
        ],
    )

    rep.h("Agentic / rigid ratios")
    rep.table(
        ["dimension", "ratio (agentic ÷ rigid)"],
        [[k, f"{v:.2f}" if v != float("inf") else "∞"] for k, v in ratios.items()],
    )
    rep.line("")
    rep.line(f"_{cmp['note']}_")
    rep.save()


if __name__ == "__main__":
    main()

"""E2 HEADLINE — rigid baseline vs agentic GraphRAG, paired funnel (k-repeat).

The headline end-to-end result: does the full agentic+KG pipeline yield more
manufacturable/simulatable designs than the rigid main-branch pipeline, and where do designs die
(instantiate -> routing_ok -> models -> sim_success -> drc_clean)?

Both arms share the SAME model (o3-mini), the SAME DemoPDK, the SAME prompts, and the SAME
layout/sim/DRC tail (run_layout_simulation) -> the LLM-integration architecture + KG is the only
variable. The agentic arm runs with KB grounding ON (RealOrchestrator) so the KG is exercised; the
AR gate is OFF (no AWS). Because the pipeline has no seed, every prompt is run K times and we report
per-prompt MAJORITY + the repeat-variance (the no-seed noise).

Arms:
  baseline  = benchmark/baseline_runner.run_one  (rigid p100->p300, headless)
  graphrag  = benchmark/e2_runner.run_one        (agentic, RealOrchestrator, grounding ON)

Reuses e2_funnel (funnel_summary/compare_arms/normalize) and the k-repeat majority pattern from
run_ablation_sweep_k. Incremental-saves so a long run is recoverable.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python benchmark/run_e2_headline.py
Env: N_PROMPTS (default all), K_REPEATS (default 3), E2_MODEL (default o3-mini), DRY=1.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path("/home/tony/PhIDOv1/wt-graphrag")
sys.path.insert(0, str(ROOT))                # mcp_servers / PhotonicsAI
sys.path.insert(0, str(ROOT / "benchmark"))  # e2_runner / e2_funnel / baseline_runner

from dotenv import load_dotenv

load_dotenv(str(ROOT / ".env"))

import baseline_runner as B  # noqa: E402
import e2_funnel as F  # noqa: E402
import e2_runner as R  # noqa: E402
from report import Reporter  # noqa: E402

K = int(os.getenv("K_REPEATS", "3"))
MODEL = os.getenv("E2_MODEL", "o3-mini")
PROMPTS_FILE = ROOT / "benchmark" / "e2_prompts.json"
OUT = ROOT / "benchmark" / "results" / "e2_headline.json"
OUT.parent.mkdir(exist_ok=True)


def load_prompts() -> list[dict]:
    """Tier-tagged subset; N_PROMPTS (if set) subsamples evenly across the list (tier spread)."""
    allp = json.loads(PROMPTS_FILE.read_text())["prompts"]
    n = int(os.getenv("N_PROMPTS", str(len(allp))))
    if n >= len(allp):
        return allp
    idxs = [round(i * (len(allp) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [allp[i] for i in idxs]


def _run_baseline(p):
    return B.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], model=MODEL)


def _run_graphrag(p, orch, cfg):
    return R.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], arm="graphrag",
                     config=cfg, orch=orch)


def run_arm(label, runfn, prompts, store):
    """K repeats per prompt; incremental save; one failure -> all-False, batch survives."""
    for p in prompts:
        for rep in range(K):
            t = time.time()
            try:
                r = runfn(p)
            except Exception as e:  # noqa: BLE001 — keep the batch alive
                r = {"prompt_id": p["id"], "level": p["level"], "arm": label,
                     "stages": dict.fromkeys(R.STAGES, False),
                     "cost": {"latency_s": time.time() - t, "failed_at": f"exc:{type(e).__name__}"},
                     "error": str(e)[:200]}
            r["level"] = p["level"]  # ensure tier present for stratification
            store.setdefault(label, {}).setdefault(p["id"], []).append(r)
            json.dump(store, open(OUT, "w"), indent=2, default=str)
            print(f"[{label:9}] {p['id']:6} L{p['level']} rep{rep + 1}/{K} "
                  f"{r['stages']} {r['cost'].get('failed_at')} {time.time() - t:.0f}s", flush=True)


def _majority(runs):
    norm = [F.normalize(r["stages"]) for r in runs]
    return {"prompt_id": runs[0]["prompt_id"], "level": runs[0].get("level", "?"),
            "stages": {s: sum(n[s] for n in norm) > len(norm) / 2 for s in F.STAGES}}


def _variance_row(store_arm):
    """Fraction of prompts with NON-unanimous repeats per stage (no-seed noise)."""
    row = {}
    for s in F.STAGES:
        nonunanimous = 0
        for runs in store_arm.values():
            vals = {F.normalize(r["stages"])[s] for r in runs}
            nonunanimous += len(vals) > 1
        row[s] = round(nonunanimous / len(store_arm), 2) if store_arm else float("nan")
    return row


def _cost_summary(store_arm):
    flat = [r for runs in store_arm.values() for r in runs]
    lat = [r["cost"].get("latency_s", 0) for r in flat]
    ti = [r["cost"].get("tokens_in", 0) or 0 for r in flat]
    to = [r["cost"].get("tokens_out", 0) or 0 for r in flat]
    n = len(flat) or 1
    return {"mean_latency_s": round(sum(lat) / n, 1),
            "mean_tokens_in": round(sum(ti) / n), "mean_tokens_out": round(sum(to) / n)}


def main():
    prompts = load_prompts()
    if os.getenv("DRY") == "1":
        print(f"[DRY] N={len(prompts)} K={K} model={MODEL}; arms: baseline + graphrag; "
              f"total runs = {2 * len(prompts) * K}")
        for p in prompts:
            print(f"  {p['id']:6} L{p['level']} buildable={p.get('buildable')}: {p['prompt'][:90]}")
        return

    store: dict = {}
    # baseline: rigid pipeline at the shared model
    run_arm("baseline", _run_baseline, prompts, store)
    # graphrag: agentic, KB grounding ON (RealOrchestrator), AR gate off (no AWS)
    orch = R.RealOrchestrator()
    cfg = R.RunConfig(model=MODEL, enable_ar_gate=False, enable_topology_gate=True,
                      max_critic_rounds=2)
    run_arm("graphrag", lambda p: _run_graphrag(p, orch, cfg), prompts, store)

    base_maj = [_majority(store["baseline"][pid]) for pid in store["baseline"]]
    gr_maj = [_majority(store["graphrag"][pid]) for pid in store["graphrag"]]
    base_flat = [r for runs in store["baseline"].values() for r in runs]
    gr_flat = [r for runs in store["graphrag"].values() for r in runs]

    rep = Reporter(OUT.with_suffix(".md"), "E2 headline — rigid baseline vs agentic GraphRAG",
                   meta={"model": MODEL, "N_prompts": len(prompts), "K_repeats": K,
                         "arms": "baseline (rigid) vs graphrag (agentic, KB-grounded)",
                         "raw_json": OUT.name})

    rep.h("Funnel — per-stage pass rate (per-prompt majority over K)")
    base_sum, gr_sum = F.funnel_summary(base_maj), F.funnel_summary(gr_maj)
    rep.table(["arm", "n"] + list(F.STAGES),
              [["baseline", base_sum["n"]] + [f"{base_sum['by_stage'][s]['rate']:.2f}" for s in F.STAGES],
               ["graphrag", gr_sum["n"]] + [f"{gr_sum['by_stage'][s]['rate']:.2f}" for s in F.STAGES]])

    rep.h("Died-at histogram (where designs die)")
    rep.table(["arm"] + [f"died_{s}" for s in F.STAGES] + ["passed_all"],
              [["baseline"] + [base_sum["died_at"].get(f"died_at_{s}", 0) for s in F.STAGES]
               + [base_sum["died_at"].get("passed_all", 0)],
               ["graphrag"] + [gr_sum["died_at"].get(f"died_at_{s}", 0) for s in F.STAGES]
               + [gr_sum["died_at"].get("passed_all", 0)]])

    rep.h("Paired comparison (Δ = graphrag − baseline; McNemar p)")
    cmp = F.compare_arms(base_maj, gr_maj)
    rep.line(f"_n_paired = {cmp['n_paired']}_")
    rep.line("")
    rep.table(["stage", "baseline", "graphrag", "Δ", "McNemar p"],
              [[s, f"{cmp['by_stage'][s]['baseline_rate']:.2f}", f"{cmp['by_stage'][s]['graphrag_rate']:.2f}",
                f"{cmp['by_stage'][s]['delta']:+.2f}", f"{cmp['by_stage'][s]['mcnemar'].get('p_value', float('nan')):.3f}"]
               for s in F.STAGES])

    rep.h("Per-tier funnel (Level 1–4)")
    for arm, summ in (("baseline", base_sum), ("graphrag", gr_sum)):
        rep.line(f"**{arm}**")
        rep.line("")
        rep.table(["level", "n"] + list(F.STAGES),
                  [[lvl, blk["n"]] + [f"{blk['by_stage'][s]['rate']:.2f}" for s in F.STAGES]
                   for lvl, blk in summ["by_level"].items()])
        rep.line("")

    rep.h("Repeat variance (fraction of prompts with non-unanimous K repeats)")
    rep.table(["arm"] + list(F.STAGES),
              [["baseline"] + [_variance_row(store["baseline"])[s] for s in F.STAGES],
               ["graphrag"] + [_variance_row(store["graphrag"])[s] for s in F.STAGES]])

    rep.h("Cost (per-run mean; infra-C)")
    bc, gc = _cost_summary(store["baseline"]), _cost_summary(store["graphrag"])
    rep.table(["arm", "mean_latency_s", "mean_tokens_in", "mean_tokens_out"],
              [["baseline", bc["mean_latency_s"], bc["mean_tokens_in"], bc["mean_tokens_out"]],
               ["graphrag", gc["mean_latency_s"], gc["mean_tokens_in"], gc["mean_tokens_out"]]])

    rep.h("Fairness controls & caveats")
    for c in [
        f"Same model ({MODEL}) both arms; same DemoPDK; same prompts; same layout/sim/DRC tail "
        "(run_layout_simulation) → the LLM-integration architecture + KG is the only variable.",
        "The agentic arm bundles KG grounding + topology gate + critic; the rigid baseline has none. "
        "This is the WHOLE-SYSTEM delta; component attribution is the (separate) ablations.",
        "Agentic: KB grounding ON (RealOrchestrator); AR gate OFF (no AWS); clarify OFF (baseline has "
        "no clarification step — symmetric).",
        "Passing = manufacturable & simulatable, NOT functionally correct. Prompts needing unmodeled "
        "parts die at 'models' — a real funnel signal.",
        "Tier labels are heuristic (complexity), pending Poon-group review; they drive stratification "
        "only, not pass/fail.",
    ]:
        rep.line(f"- {c}")
    rep.save()
    print(f"\n[ok] raw -> {OUT}")


if __name__ == "__main__":
    main()

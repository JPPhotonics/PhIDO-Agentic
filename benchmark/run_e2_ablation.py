"""E2 ablation grid — reframes the two-arm baseline-vs-graphrag funnel into a feature
ablation over the four "new" agentic features, plus the two anchors.

Four toggleable features (each a RunConfig axis):
    kg       — KG grounding (explore grounding_rounds>0) + hybrid RRF retrieval backend
    routing  — iterative tool-calling builder (vs single-shot extraction)
    gate     — Clingo topology gate
    critic   — critic review loop (now orthogonal to routing; runs under iterative too)

AR-gate is held OFF for every arm (AWS Bedrock unavailable → no-op); the `full-AR`/`base+AR`
slots are documented in the design but not runnable here.

Arms (feature bits = kg,routing,gate,critic):
    rigid_baseline .... main-branch rigid pipeline (LLM top-1 search, no orchestrator)
    base_agentic 0000 . orchestrator + MCP floor (all four features off)
    full         1111 . all four on
    full_minus_* ..... leave-one-out from full (marginal contribution IN CONTEXT)
    base_plus_*  ..... additive from base_agentic (STANDALONE contribution)

Report: per-arm funnel (per-prompt majority over K), anchor comparisons, and per-feature
additive (base+X − base) and leave-one-out (full − full∖X) marginals with paired McNemar.

Run (K=1 pilot over the 103-prompt testbench set):
    K_REPEATS=1 PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_e2_ablation.py
"""

from __future__ import annotations

import json
import os
import time

import baseline_runner as B
import e2_funnel as F
import e2_runner as R
from dotenv import load_dotenv
from report import Reporter
from run_e2_testbench import PROMPTS_FILE, ROOT, build_prompts_file

load_dotenv(str(ROOT / ".env"))

K = int(os.getenv("K_REPEATS", "1"))
MODEL = os.getenv("E2_MODEL", "o3-mini")
RIGID_MODEL = os.getenv("RIGID_MODEL", "o1")  # baseline_runner default is the main-branch "o1"
N = int(os.getenv("N_PROMPTS", "0"))  # 0 = all testbench prompts
OUT = ROOT / "benchmark" / "results" / "e2_ablation.json"

# ROUTING (iterative extraction) is treated as UNIMPLEMENTED: its bit is held 0 across every arm,
# so there are no routing arms and "full" is kg+gate+critic (== app default).
FEATURES = ("kg", "gate", "critic")

# feature bits: (kg, routing, gate, critic) — routing always 0
ARMS: dict[str, tuple[int, int, int, int]] = {
    "base_agentic": (0, 0, 0, 0),
    "full": (1, 0, 1, 1),
    "full_minus_kg": (0, 0, 1, 1),
    "full_minus_gate": (1, 0, 0, 1),
    "full_minus_critic": (1, 0, 1, 0),
    "base_plus_kg": (1, 0, 0, 0),
    "base_plus_gate": (0, 0, 1, 0),
    "base_plus_critic": (0, 0, 0, 1),
}
# run anchors first for early signal, then LOO, then additive
RUN_ORDER = [
    "base_agentic", "full",
    "full_minus_kg", "full_minus_gate", "full_minus_critic",
    "base_plus_kg", "base_plus_gate", "base_plus_critic",
]
# feature -> (additive arm vs base_agentic, leave-one-out arm vs full)
FEATMAP = {
    "kg": ("base_plus_kg", "full_minus_kg"),
    "gate": ("base_plus_gate", "full_minus_gate"),
    "critic": ("base_plus_critic", "full_minus_critic"),
}


def cfg_for(bits: tuple[int, int, int, int]) -> R.RunConfig:
    kg, routing, gate, critic = bits
    return R.RunConfig(
        model=MODEL,
        enable_topology_gate=bool(gate),
        enable_ar_gate=False,  # Bedrock unavailable — held off for all arms
        max_critic_rounds=2 if critic else 0,
        extraction_mode="iterative" if routing else "single_shot",
        retrieval_backend="hybrid" if kg else "lexical",
        grounding_rounds=5 if kg else 0,
    )


def _stratified(prompts: list[dict], per_level: int) -> list[dict]:
    """Evenly-spaced `per_level` prompts from each complexity level (sorted by id),
    so a small pilot subset stays representative of the level mix. Deterministic."""
    from collections import defaultdict
    by: dict = defaultdict(list)
    for p in prompts:
        by[p["level"]].append(p)
    out = []
    for lvl in sorted(by):
        grp = sorted(by[lvl], key=lambda p: p["id"])
        m, k = len(grp), min(per_level, len(by[lvl]))
        if k <= 0:
            continue
        idx = [0] if k == 1 else sorted({round(i * (m - 1) / (k - 1)) for i in range(k)})
        out.extend(grp[j] for j in idx)
    return out


def _apply_paper_levels(prompts: list[dict]) -> None:
    """Overwrite each prompt's ``level`` with the paper Table-1 (component-count) level from the
    retag sidecar (``results/level_retag_testbench.json``), so stratification AND the per-level
    report use the corrected axis instead of the repo's heuristic tags
    ([[e2-prompt-level-mislabeling-2026-07-09]]). Prompts absent from the sidecar keep their repo
    level (logged), so the run never silently drops a prompt."""
    side = ROOT / "benchmark" / "results" / "level_retag_testbench.json"
    if not side.exists():
        print(f"[paper-levels] sidecar not found ({side}); keeping repo levels", flush=True)
        return
    lp = {r["id"]: r["level_paper"] for r in json.loads(side.read_text())
          if r.get("level_paper") is not None}
    miss = [p["id"] for p in prompts if p["id"] not in lp]
    for p in prompts:
        if p["id"] in lp:
            p["level"] = lp[p["id"]]
    print(f"[paper-levels] remapped {len(prompts) - len(miss)}/{len(prompts)} prompts"
          + (f"; {len(miss)} missing kept repo levels: {miss[:10]}" if miss else ""), flush=True)


def load_prompts() -> list[dict]:
    build_prompts_file()
    prompts = json.loads(PROMPTS_FILE.read_text())["prompts"]
    if os.getenv("PHIDO_USE_PAPER_LEVELS") == "1":
        _apply_paper_levels(prompts)
    per_level = int(os.getenv("N_PER_LEVEL", "0"))
    if per_level:
        return _stratified(prompts, per_level)
    return prompts[:N] if N else prompts


def _fail_result(pid, level, label, t, exc):
    return {
        "prompt_id": pid, "level": level, "arm": label,
        "stages": dict.fromkeys(R.STAGES, False),
        "cost": {"latency_s": time.time() - t, "failed_at": f"exc:{type(exc).__name__}"},
        "diag": {"last_error": f"{type(exc).__name__}: {exc}"[:300]},
    }


def run_arm(label, prompts, store, *, rigid=False, cfg=None, orch=None):
    for p in prompts:
        done = len(store.get(label, {}).get(p["id"], []))
        if done >= K:  # resumable: skip (arm, prompt) pairs already at K reps
            print(f"[{label:20}] {p['id']:6} skip (have {done}/{K})", flush=True)
            continue
        for rep in range(done, K):
            t = time.time()
            try:
                if rigid:
                    r = B.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], model=RIGID_MODEL)
                else:
                    r = R.run_one(p["prompt"], prompt_id=p["id"], level=p["level"],
                                  arm=label, config=cfg, orch=orch)
            except Exception as e:  # noqa: BLE001 — keep the batch alive
                r = _fail_result(p["id"], p["level"], label, t, e)
            store.setdefault(label, {}).setdefault(p["id"], []).append(r)
            json.dump(store, open(OUT, "w"), indent=2, default=str)
            print(f"[{label:20}] {p['id']:6} rep{rep + 1}/{K} {r['stages']} "
                  f"{time.time() - t:.0f}s", flush=True)


def _majority(runs):
    norm = [F.normalize(r["stages"]) for r in runs]
    return {"prompt_id": runs[0]["prompt_id"], "level": runs[0].get("level", "?"),
            "stages": {s: sum(n[s] for n in norm) > len(norm) / 2 for s in F.STAGES}}


def _majorities(store):
    return {label: [_majority(byp[pid]) for pid in byp] for label, byp in store.items()}


def _marginal_table(rep, title, ref_maj, treat_maj):
    """Δ = treat − ref, per stage, per-prompt majority, with McNemar."""
    cmp = F.compare_arms(ref_maj, treat_maj)  # baseline=ref(OFF), graphrag=treat(ON)
    rep.h(f"{title} (n_paired={cmp['n_paired']})")
    rep.table(["stage", "OFF", "ON", "Δ", "McNemar p"],
              [[s, f"{st['baseline_rate']:.2f}", f"{st['graphrag_rate']:.2f}",
                f"{st['delta']:+.2f}", f"{st['mcnemar']['p_value']:.3f}"]
               for s, st in cmp["by_stage"].items()])
    return cmp


def main():
    prompts = load_prompts()
    from collections import Counter
    lvl_mix = dict(sorted(Counter(p["level"] for p in prompts).items()))
    if os.getenv("DRY") == "1":
        print(f"[DRY] N={len(prompts)} level_mix={lvl_mix} K={K} model={MODEL} rigid_model={RIGID_MODEL}")
        print(f"      arms: rigid_baseline + {len(ARMS)} agentic = {1 + len(ARMS)}")
        print(f"      total runs = {(1 + len(ARMS)) * len(prompts) * K}")
        print(f"      selected ids: {[p['id'] for p in prompts]}")
        for label in RUN_ORDER:
            print(f"        {label:20} bits(kg,routing,gate,critic)={ARMS[label]}")
        return

    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    orch = R.RealOrchestrator()

    only = [a.strip() for a in os.getenv("ARMS_ONLY", "").split(",") if a.strip()]
    run_labels = [a for a in RUN_ORDER if a in only] if only else RUN_ORDER
    if os.getenv("SKIP_RIGID") != "1" and (not only or "rigid_baseline" in only):
        run_arm("rigid_baseline", prompts, store, rigid=True)
    for label in run_labels:
        run_arm(label, prompts, store, cfg=cfg_for(ARMS[label]), orch=orch)

    # ── report ────────────────────────────────────────────────────────
    maj = _majorities(store)
    rep = Reporter(OUT.with_suffix(".md"),
                   "E2 ablation grid — feature LOO + additive (no gold; correctness = manual)",
                   meta={"model": MODEL, "rigid_model": RIGID_MODEL, "N_prompts": len(prompts),
                         "level_mix": lvl_mix, "K_repeats": K,
                         "AR_gate": "OFF for all arms (Bedrock unavailable)",
                         "raw_json": OUT.name, "artifacts": os.getenv("E2_OUTPUT_DIR", "(unset)")})

    all_arms = ["rigid_baseline"] + RUN_ORDER
    rep.h("Per-arm funnel — per-prompt majority over K")
    rows = []
    for label in all_arms:
        if label not in maj:
            continue
        summ = F.funnel_summary(maj[label])
        bits = "-" if label == "rigid_baseline" else "".join(str(b) for b in ARMS[label])
        rows.append([label, bits, summ["n"]] + [f"{summ['by_stage'][s]['rate']:.2f}" for s in F.STAGES])
    rep.table(["arm", "kg·rt·gt·cr", "n"] + list(F.STAGES), rows)

    # anchor comparisons
    if "full" in maj and "rigid_baseline" in maj:
        _marginal_table(rep, "Anchor: full − rigid_baseline", maj["rigid_baseline"], maj["full"])
    if "base_agentic" in maj and "rigid_baseline" in maj:
        _marginal_table(rep, "Anchor: base_agentic − rigid_baseline", maj["rigid_baseline"], maj["base_agentic"])
    if "full" in maj and "base_agentic" in maj:
        _marginal_table(rep, "Anchor: full − base_agentic", maj["base_agentic"], maj["full"])

    # per-feature marginals + a drc_clean focus summary
    focus = []  # feature, additive Δ drc, additive p, LOO Δ drc, LOO p
    for feat, (add_arm, loo_arm) in FEATMAP.items():
        add_cmp = loo_cmp = None
        if add_arm in maj and "base_agentic" in maj:
            add_cmp = _marginal_table(rep, f"[{feat}] additive: {add_arm} − base_agentic",
                                      maj["base_agentic"], maj[add_arm])
        if loo_arm in maj and "full" in maj:
            loo_cmp = _marginal_table(rep, f"[{feat}] leave-one-out: full − {loo_arm}",
                                      maj[loo_arm], maj["full"])
        def _d(cmp): return f"{cmp['by_stage']['drc_clean']['delta']:+.2f}" if cmp else "—"
        def _p(cmp): return f"{cmp['by_stage']['drc_clean']['mcnemar']['p_value']:.3f}" if cmp else "—"
        focus.append([feat, _d(add_cmp), _p(add_cmp), _d(loo_cmp), _p(loo_cmp)])

    rep.h("drc_clean marginal summary (Δ = ON − OFF)")
    rep.table(["feature", "additive Δ", "additive p", "LOO Δ", "LOO p"], focus)

    if K > 1:
        rep.h("Repeat variance: fraction of prompts with non-unanimous repeats")
        vrows = []
        for label in all_arms:
            byp = store.get(label, {})
            if not byp:
                continue
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
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

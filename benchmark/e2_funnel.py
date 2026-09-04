"""E2 / B6 — end-to-end staged success funnel + paired arm comparison.

A design dies somewhere in:

    instantiate -> routing_ok -> models -> sim_success -> drc_clean

so we report a **funnel** (where designs die), not one pass/fail, complexity-stratified by
Level 1-4 and self-labeled (DRC + SAX grade themselves). Arms (baseline vs GraphRAG) are
compared **paired** on the same prompts: per-stage McNemar + bootstrap CIs.

Caveat (state it): passing the funnel = manufacturable & simulatable, NOT functionally
correct. A run result is ``{"prompt_id", "level", "arm", "stages": {stage: bool}}``.
"""

from __future__ import annotations

from collections import defaultdict

from stats import bootstrap_ci, paired_mcnemar, wilson

STAGES = ["instantiate", "routing_ok", "models", "sim_success", "drc_clean"]


def normalize(stages: dict) -> dict:
    """Enforce monotonicity: once a stage fails, every later stage is False."""
    out, alive = {}, True
    for s in STAGES:
        alive = alive and bool(stages.get(s, False))
        out[s] = alive
    return out


def furthest_stage(stages: dict) -> str:
    """Name of the last stage reached ('none' if it didn't even instantiate)."""
    norm = normalize(stages)
    reached = [s for s in STAGES if norm[s]]
    return reached[-1] if reached else "none"


def funnel_summary(results: list[dict]) -> dict:
    """Per-stage pass counts/rates (+ Wilson CI), overall and stratified by level."""
    def summarize(rs: list[dict]) -> dict:
        n = len(rs)
        block = {"n": n, "by_stage": {}}
        for s in STAGES:
            passed = sum(1 for r in rs if normalize(r["stages"])[s])
            block["by_stage"][s] = {"pass": passed, "rate": (passed / n) if n else float("nan"),
                                    "ci95": wilson(passed, n) if n else None}
        block["died_at"] = _died_at_histogram(rs)
        return block

    out = summarize(results)
    by_level = defaultdict(list)
    for r in results:
        by_level[r.get("level", "?")].append(r)
    out["by_level"] = {lvl: summarize(rs) for lvl, rs in sorted(by_level.items(), key=lambda kv: str(kv[0]))}
    return out


def _died_at_histogram(results: list[dict]) -> dict:
    hist: dict[str, int] = defaultdict(int)
    for r in results:
        norm = normalize(r["stages"])
        if all(norm[s] for s in STAGES):
            hist["passed_all"] += 1
        else:
            first_fail = next(s for s in STAGES if not norm[s])
            hist[f"died_at_{first_fail}"] += 1
    return dict(hist)


def compare_arms(baseline: list[dict], graphrag: list[dict]) -> dict:
    """Paired per-stage comparison on the shared prompt set: rates, Δ, McNemar, bootstrap CI."""
    b = {r["prompt_id"]: normalize(r["stages"]) for r in baseline}
    g = {r["prompt_id"]: normalize(r["stages"]) for r in graphrag}
    shared = sorted(set(b) & set(g))
    out = {"n_paired": len(shared), "by_stage": {}}
    for s in STAGES:
        bv = [int(b[p][s]) for p in shared]
        gv = [int(g[p][s]) for p in shared]
        br = sum(bv) / len(bv) if bv else float("nan")
        gr = sum(gv) / len(gv) if gv else float("nan")
        diff_ci = bootstrap_ci(list(zip(bv, gv)),
                               lambda pairs: (sum(y for _, y in pairs) - sum(x for x, _ in pairs)) / len(pairs),
                               n_boot=1000, seed=0) if shared else {}
        out["by_stage"][s] = {"baseline_rate": br, "graphrag_rate": gr, "delta": gr - br,
                              "mcnemar": paired_mcnemar(gv, bv), "delta_ci95": diff_ci}
    return out

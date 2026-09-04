"""C (infra) — cost A/B aggregator: is the agentic overhead worth the extrinsic lift?

Aggregates per-run logs into the cost comparison the benchmark needs: tokens / latency /
rounds / cap-hits for the agentic GraphRAG pipeline vs the linear baseline, plus the human
**review-queue burden** (auto / queue / reject split — ties to the confidence calibration).
The critic on/off ablation reuses ``e2_funnel.compare_arms`` for the E2 effect; this module
supplies its cost side.

A run log is ``{"arm", "prompt_id", "tokens_in", "tokens_out", "latency_s", "rounds",
"cap_hit": bool, "review_outcome": "auto"|"queue"|"reject"}``.
"""

from __future__ import annotations

import statistics
from collections import Counter


def to_cost_log(run: dict) -> dict:
    """Flatten an e2_runner / baseline_runner result (tokens nested under `cost`) into the flat
    cost-log shape this module consumes. cap_hit / review_outcome default when not instrumented."""
    cost = run.get("cost", {})
    return {
        "arm": run.get("arm"),
        "prompt_id": run.get("prompt_id"),
        "tokens_in": cost.get("tokens_in", 0),
        "tokens_out": cost.get("tokens_out", 0),
        "latency_s": cost.get("latency_s"),
        "rounds": cost.get("rounds", 0),
        "cap_hit": bool(cost.get("cap_hit", False)),
        "review_outcome": cost.get("review_outcome", "auto"),
        "drc_clean": bool(run.get("stages", {}).get("drc_clean")),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def aggregate(logs: list[dict]) -> dict:
    n = len(logs)
    if n == 0:
        return {"n": 0}
    tin = [g.get("tokens_in", 0) for g in logs]
    tout = [g.get("tokens_out", 0) for g in logs]
    total_tokens = sum(tin) + sum(tout)
    n_success = sum(1 for g in logs if g.get("drc_clean"))
    return {
        "n": n,
        "tokens_in_total": sum(tin),
        "tokens_out_total": sum(tout),
        "tokens_total": total_tokens,
        "tokens_per_run_mean": total_tokens / n,
        "n_drc_clean": n_success,
        # cost-effectiveness: tokens spent per SUCCESSFUL (DRC-clean) design — the defensible denominator
        "tokens_per_success": (total_tokens / n_success) if n_success else float("inf"),
        "latency_s_mean": _mean([g.get("latency_s") for g in logs]),
        "latency_s_total": sum(g.get("latency_s", 0) for g in logs),
        "rounds_mean": _mean([g.get("rounds") for g in logs]),
        "cap_hit_rate": sum(1 for g in logs if g.get("cap_hit")) / n,
        "review_burden": review_burden(logs),
    }


def review_burden(logs: list[dict]) -> dict:
    """Auto / queue / reject split — the human review load (ties to calibration)."""
    n = len(logs)
    c = Counter(g.get("review_outcome", "auto") for g in logs)
    return {
        k: {"n": c.get(k, 0), "frac": (c.get(k, 0) / n) if n else 0.0}
        for k in ("auto", "queue", "reject")
    }


def _ratio(a: float, b: float) -> float:
    return (a / b) if b else float("inf")


def compare(agentic: list[dict], linear: list[dict]) -> dict:
    """Cost A/B: the agentic overhead the extrinsic lift must justify."""
    a, lin = aggregate(agentic), aggregate(linear)
    ratios = {
        "tokens_total": _ratio(a.get("tokens_total", 0), lin.get("tokens_total", 0)),
        "tokens_per_run": _ratio(
            a.get("tokens_per_run_mean", 0), lin.get("tokens_per_run_mean", 0)
        ),
        "tokens_per_success": _ratio(
            a.get("tokens_per_success", 0), lin.get("tokens_per_success", 0)
        ),
        "latency_mean": _ratio(a.get("latency_s_mean", 0), lin.get("latency_s_mean", 0)),
        "rounds_mean": _ratio(a.get("rounds_mean", 0), lin.get("rounds_mean", 0)),
    }
    return {
        "agentic": a,
        "linear": lin,
        "ratios": ratios,
        "note": "ratios > 1 = agentic costs more; tokens_per_success is the cost-effectiveness "
        "number (tokens per DRC-clean design) — the honest denominator",
    }

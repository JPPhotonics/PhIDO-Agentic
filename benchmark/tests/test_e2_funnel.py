"""Tests for the E2 funnel scorer + paired arm comparison."""

from __future__ import annotations

import e2_funnel as F


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def mk(pid, level, last_ok):
    """Build a run result that passes up to and including `last_ok`."""
    idx = F.STAGES.index(last_ok) if last_ok != "none" else -1
    return {"prompt_id": pid, "level": level, "stages": {s: (i <= idx) for i, s in enumerate(F.STAGES)}}


def test_monotonicity_and_furthest():
    # a non-monotone input gets clamped: models False kills sim/drc even if marked True
    stages = {"instantiate": True, "routing_ok": True, "models": False, "sim_success": True, "drc_clean": True}
    norm = F.normalize(stages)
    assert norm["sim_success"] is False and norm["drc_clean"] is False
    assert F.furthest_stage(stages) == "routing_ok"
    assert F.furthest_stage({s: True for s in F.STAGES}) == "drc_clean"
    assert F.furthest_stage({}) == "none"


def test_funnel_summary():
    results = [mk("p1", "L1", "drc_clean"), mk("p2", "L1", "sim_success"),
               mk("p3", "L2", "routing_ok"), mk("p4", "L2", "none")]
    s = F.funnel_summary(results)
    assert s["by_stage"]["instantiate"]["pass"] == 3        # p4 failed to instantiate
    assert s["by_stage"]["drc_clean"]["pass"] == 1          # only p1
    assert s["died_at"]["died_at_instantiate"] == 1 and s["died_at"]["passed_all"] == 1
    assert set(s["by_level"]) == {"L1", "L2"}


def test_compare_arms():
    base = [mk("p1", "L1", "routing_ok"), mk("p2", "L1", "models"), mk("p3", "L1", "none")]
    graph = [mk("p1", "L1", "drc_clean"), mk("p2", "L1", "drc_clean"), mk("p3", "L1", "routing_ok")]
    cmp = F.compare_arms(base, graph)
    assert cmp["n_paired"] == 3
    drc = cmp["by_stage"]["drc_clean"]
    assert approx(drc["baseline_rate"], 0.0) and approx(drc["graphrag_rate"], 2 / 3)
    assert drc["delta"] > 0                                  # GraphRAG improves DRC-clean
    assert "p_value" in drc["mcnemar"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} e2-funnel tests passed.")


if __name__ == "__main__":
    main()

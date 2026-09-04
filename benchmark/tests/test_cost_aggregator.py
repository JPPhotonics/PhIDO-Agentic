"""Tests for the infra cost aggregator (mock run logs)."""

from __future__ import annotations

import cost_aggregator as C


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


AGENTIC = [
    {"arm": "graphrag", "prompt_id": "p1", "tokens_in": 4000, "tokens_out": 2000, "latency_s": 30, "rounds": 4, "cap_hit": False, "review_outcome": "auto"},
    {"arm": "graphrag", "prompt_id": "p2", "tokens_in": 6000, "tokens_out": 3000, "latency_s": 50, "rounds": 6, "cap_hit": True, "review_outcome": "queue"},
    {"arm": "graphrag", "prompt_id": "p3", "tokens_in": 5000, "tokens_out": 2500, "latency_s": 40, "rounds": 5, "cap_hit": False, "review_outcome": "reject"},
]
LINEAR = [
    {"arm": "baseline", "prompt_id": "p1", "tokens_in": 1000, "tokens_out": 500, "latency_s": 8, "rounds": 1, "cap_hit": False, "review_outcome": "auto"},
    {"arm": "baseline", "prompt_id": "p2", "tokens_in": 1200, "tokens_out": 600, "latency_s": 9, "rounds": 1, "cap_hit": False, "review_outcome": "auto"},
    {"arm": "baseline", "prompt_id": "p3", "tokens_in": 1100, "tokens_out": 550, "latency_s": 8, "rounds": 1, "cap_hit": False, "review_outcome": "auto"},
]


def test_aggregate():
    a = C.aggregate(AGENTIC)
    assert a["n"] == 3 and a["tokens_total"] == 4000 + 2000 + 6000 + 3000 + 5000 + 2500
    assert approx(a["cap_hit_rate"], 1 / 3)
    assert approx(a["rounds_mean"], 5.0)


def test_review_burden():
    rb = C.review_burden(AGENTIC)
    assert rb["auto"]["n"] == 1 and rb["queue"]["n"] == 1 and rb["reject"]["n"] == 1
    assert approx(rb["queue"]["frac"], 1 / 3)


def test_compare_ratios():
    cmp = C.compare(AGENTIC, LINEAR)
    assert cmp["ratios"]["tokens_total"] > 1     # agentic costs more
    assert cmp["ratios"]["rounds_mean"] == 5.0   # 5 rounds vs 1
    assert cmp["ratios"]["latency_mean"] > 1


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} cost-aggregator tests passed.")


if __name__ == "__main__":
    main()

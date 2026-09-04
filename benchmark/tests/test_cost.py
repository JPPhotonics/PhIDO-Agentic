"""Hermetic tests for the cost instrumentation (token_meter + cost_aggregator).

No SDK / network: TokenMeter is tested via its `_record` tally on fake response objects, and the
aggregator on synthetic logs. The live SDK-patch interception is validated separately by a tiny
real call in the harness bring-up, not here.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python benchmark/tests/test_cost.py
"""

from __future__ import annotations

import types

import cost_aggregator as C
from token_meter import TokenMeter


def _resp(p, c):
    return types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=p, completion_tokens=c)
    )


def test_token_meter_records_usage():
    m = TokenMeter()
    m._record(_resp(100, 30))
    m._record(_resp(50, 20))
    assert (m.tokens_in, m.tokens_out, m.calls) == (150, 50, 2)


def test_token_meter_ignores_missing_usage():
    m = TokenMeter()
    m._record(types.SimpleNamespace())  # no .usage attr
    m._record(types.SimpleNamespace(usage=None))  # explicit None
    assert (m.tokens_in, m.tokens_out, m.calls) == (0, 0, 0)


def test_to_cost_log_flattens_nested_cost():
    run = {
        "arm": "agentic",
        "prompt_id": "p1",
        "stages": {"drc_clean": True},
        "cost": {"latency_s": 12.0, "rounds": 2, "tokens_in": 900, "tokens_out": 300},
    }
    log = C.to_cost_log(run)
    assert log["tokens_in"] == 900 and log["tokens_out"] == 300
    assert log["drc_clean"] is True and log["rounds"] == 2


def test_aggregate_cost_per_success():
    logs = [
        {
            "arm": "a",
            "tokens_in": 800,
            "tokens_out": 200,
            "latency_s": 10,
            "rounds": 1,
            "drc_clean": True,
        },
        {
            "arm": "a",
            "tokens_in": 600,
            "tokens_out": 400,
            "latency_s": 12,
            "rounds": 2,
            "drc_clean": False,
        },
    ]
    agg = C.aggregate(logs)
    assert agg["tokens_total"] == 2000
    assert agg["n_drc_clean"] == 1
    assert agg["tokens_per_success"] == 2000.0  # 1 success → all tokens / 1
    assert agg["tokens_per_run_mean"] == 1000.0


def test_compare_ratios():
    agentic = [
        {
            "tokens_in": 1500,
            "tokens_out": 500,
            "latency_s": 20,
            "rounds": 2,
            "drc_clean": True,
        }
    ]
    rigid = [
        {
            "tokens_in": 500,
            "tokens_out": 250,
            "latency_s": 10,
            "rounds": 0,
            "drc_clean": True,
        }
    ]
    cmp = C.compare(agentic, rigid)
    assert cmp["ratios"]["tokens_total"] == 2000 / 750
    assert cmp["ratios"]["tokens_per_success"] == 2000 / 750  # both 1 success


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} cost tests passed.")


if __name__ == "__main__":
    main()

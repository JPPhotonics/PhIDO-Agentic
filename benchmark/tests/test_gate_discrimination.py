"""Tests for the rule-aligned topology-gate discrimination eval (gate_discrimination.py).

Runs the REAL Clingo gate over valid + per-rule-violation DesignIntents and asserts the
gate is sound + complete over its ENCODED error classes — the intrinsic-quality result the
end-to-end ablation (confounded by DRC noise) cannot produce. Requires clingo + the
architecture_rules.lp policy (both present on this branch).

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python benchmark/tests/test_gate_discrimination.py
"""

from __future__ import annotations

import gate_discrimination as D


def test_valid_seeds_pass():
    # every curated valid seed must pass the real gate (no false rejects)
    for case in D.valid_cases():
        rejected, codes = D._run_gate(case["di"])
        assert not rejected, f"{case['id']} wrongly rejected: fired {codes}"


def test_every_invalid_caught_with_right_code():
    # each injected violation fires, AND fires its expected clingo error code
    for case in D.invalid_cases():
        rejected, codes = D._run_gate(case["di"])
        assert rejected, f"{case['id']} not caught (gate fired nothing)"
        assert case["expected_code"] in codes, (
            f"{case['id']} caught but wrong code: expected {case['expected_code']}, fired {codes}"
        )


def test_full_report_sound_and_complete():
    records = D.build_records()
    from gate_eval import formal_gate_eval

    rep = formal_gate_eval(records)
    assert rep["false_reject_rate"] == 0.0
    assert rep["catch_rate"] == 1.0
    assert rep["coverage_gap"] == []  # no encoded class is ever missed
    assert rep["soundness"]["n_invalid_accepted"] == 0
    assert rep["n_error_classes"] >= 15  # full encoded rule set exercised


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} gate-discrimination tests passed.")


if __name__ == "__main__":
    main()

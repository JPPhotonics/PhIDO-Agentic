"""Tests for the LLM faithfulness judge (deterministic mock llm_fn)."""

from __future__ import annotations

import judge as J


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def keyword_judge(prompt, sys):
    """Mock: SUPPORTED iff the evidence carries the 'match-ok' sentinel (order-insensitive)."""
    return "SUPPORTED" if "match-ok" in prompt.lower() else "NOT_SUPPORTED"


def test_supported_and_not():
    v_ok = J.judge_faithfulness(("MZI", "BASED_ON_PRINCIPLE", "interference"),
                                ["the device relies on two-beam match-ok interference"], keyword_judge, votes=3)
    assert v_ok.label == "supported" and v_ok.n_supported == 6 and v_ok.order_consistent
    v_no = J.judge_faithfulness(("ring", "PERFORMS_FUNCTION", "modulation"),
                                ["the ring filters wavelengths"], keyword_judge, votes=3)
    assert v_no.label == "not_supported" and v_no.n_supported == 0


def test_position_bias_detection():
    # an order-sensitive judge: SUPPORTED only when EVIDENCE appears before CLAIM
    def biased(prompt, sys):
        return "SUPPORTED" if prompt.find("EVIDENCE") < prompt.find("CLAIM") else "NOT_SUPPORTED"
    v = J.judge_faithfulness(("a", "r", "b"), ["q"], biased, votes=1)
    assert v.order_consistent is False                 # flipped with ordering -> position bias flagged


def test_calibration():
    cases = [
        {"id": "c1", "triple": ("MZI", "p", "interference"), "quotes": ["match-ok yes"], "gold": "supported"},
        {"id": "c2", "triple": ("ring", "p", "modulation"), "quotes": ["filters light"], "gold": "not_supported"},
        {"id": "c3", "triple": ("dc", "p", "coupling"), "quotes": ["match-ok couples"], "gold": "supported"},
    ]
    cal = J.calibrate_judge(cases, keyword_judge, votes=3)
    assert cal["n"] == 3 and approx(cal["agreement"], 1.0) and approx(cal["cohen_kappa"], 1.0)
    assert cal["misclassified"] == [] and approx(cal["order_bias_rate"], 0.0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} judge tests passed.")


if __name__ == "__main__":
    main()

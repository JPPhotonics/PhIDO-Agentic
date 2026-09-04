"""Tests for B1 clarification eval + user simulator (mock LLM)."""

from __future__ import annotations

import clarification_eval as C


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_seed_set():
    amb = [q for q in C.SEED_QUERIES if q.ambiguous]
    clear = [q for q in C.SEED_QUERIES if not q.ambiguous]
    assert len(amb) == 3 and len(clear) == 3
    assert all(q.hidden_spec for q in amb)            # ambiguous queries carry a hidden spec for the simulator


def test_should_clarify_scores():
    # perfect clarifier: clarifies exactly the ambiguous ones
    perfect = C.evaluate_clarifier(C.SEED_QUERIES, lambda p: p in {q.prompt for q in C.SEED_QUERIES if q.ambiguous})
    assert approx(perfect["precision"], 1.0) and approx(perfect["recall"], 1.0) and approx(perfect["accuracy"], 1.0)
    # over-asker: always clarifies -> recall 1, precision 0.5 (3 tp / (3 tp + 3 fp))
    over = C.evaluate_clarifier(C.SEED_QUERIES, lambda p: True)
    assert approx(over["recall"], 1.0) and approx(over["precision"], 0.5)
    assert over["confusion"]["fp"] == 3
    # never-asker: recall 0
    never = C.evaluate_clarifier(C.SEED_QUERIES, lambda p: False)
    assert approx(never["recall"], 0.0) and never["confusion"]["fn"] == 3


def test_intent_match():
    gold = {"component": "mzi", "ports": "2x2", "bandwidth_GHz": 10}
    assert approx(C.intent_match({"component": "MZI", "ports": "2x2", "bandwidth_GHz": 10}, gold)["accuracy"], 1.0)
    partial = C.intent_match({"component": "mzi", "ports": "1x2", "bandwidth_GHz": 10}, gold)
    assert partial["matched"] == 2 and partial["total"] == 3
    assert partial["per_field"]["ports"] is False


def test_user_simulator():
    # mock llm_fn echoes that it saw the spec; assert the spec is passed through
    seen = {}
    def mock(prompt, sys):
        seen["prompt"] = prompt
        return "1x4, C-band, equal split"
    sim = C.UserSimulator({"ports": "1x4", "band": "C"}, llm_fn=mock)
    ans = sim.answer("How many output ports do you want?")
    assert ans == "1x4, C-band, equal split"
    assert "1x4" in seen["prompt"] and "output ports" in seen["prompt"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} clarification-eval tests passed.")


if __name__ == "__main__":
    main()

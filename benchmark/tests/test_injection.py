"""Tests for the B4 topology and B5 parameter error injectors + gate_eval integration.

Run: PYTHONPATH=benchmark .venv/bin/python benchmark/tests/test_injection.py
"""

from __future__ import annotations

import gate_eval as G
import inject_parameter as P
import inject_topology as T


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def mock_gate(records, caught_classes, false_reject_ids=frozenset()):
    """A stand-in gate: rejects invalids whose class is encoded; passes valids unless listed."""
    out = []
    for r in records:
        if r["label"] == "invalid":
            rejected = r["error_class"] in caught_classes
        else:
            rejected = r["id"] in false_reject_ids
        out.append({**r, "rejected": rejected})
    return out


def test_topology_injector():
    inv = T.inject_invalids(seed=1)
    val = T.valid_records()
    assert len(val) == 2
    assert {r["error_class"] for r in inv} == set(T.TOPOLOGY_INJECTORS)  # all classes realized
    assert all(r["label"] == "invalid" for r in inv)
    # structural sanity of a few mutations
    floating = next(r for r in inv if r["error_class"] == "floating_instance")
    assert "orphan" in floating["topology"].instances
    ghost = next(r for r in inv if r["error_class"] == "nonexistent_ref")
    assert any("ghost." in a or "ghost." in b for a, b in ghost["topology"].connections)


def test_parameter_injector():
    inv = P.inject_invalids(seed=1)
    val = P.valid_records()
    assert len(val) == 3 and len(inv) > 0
    pairing = [r for r in inv if r["error_class"] == "incompatible_pairing"]
    assert {r["component"] for r in pairing} == {"ring"}              # only ring has the constraint
    grid = {r["component"] for r in inv if r["error_class"] == "grid_violation"}
    assert "straight" not in grid                                     # straight has no grid spec


def test_topology_gate_eval():
    caught = {"dangling_port", "nonexistent_ref", "self_loop", "type_mismatch"}
    records = mock_gate(T.inject_invalids(seed=1) + T.valid_records(), caught, false_reject_ids={"valid-0"})
    fg = G.formal_gate_eval(records)
    assert approx(fg["catch_rate"], 8 / 12)                           # 4 caught classes × 2 seeds / 12
    assert fg["coverage_gap"] == ["double_connect", "floating_instance"]
    assert approx(fg["false_reject_rate"], 0.5)                       # 1 of 2 valids falsely rejected


def test_parameter_gate_eval():
    caught = {"out_of_range", "type_mismatch", "missing_required", "incompatible_pairing", "grid_violation"}
    records = mock_gate(P.inject_invalids(seed=1) + P.valid_records(), caught)
    fg = G.formal_gate_eval(records)
    assert fg["coverage_gap"] == ["wrong_units"]
    assert approx(fg["false_reject_rate"], 0.0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} injection tests passed.")


if __name__ == "__main__":
    main()

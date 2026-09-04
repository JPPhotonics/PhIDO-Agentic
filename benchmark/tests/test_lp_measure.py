"""Tests for LP-Measure + de-leak (synthetic structured graph)."""

from __future__ import annotations

import lp_measure as LP


def structured_graph():
    """Consistent structure: every Ai 'is_a' device and 'band' Cband -> highly recoverable."""
    triples = []
    for i in range(12):
        triples.append((f"A{i}", "is_a", "device"))
        triples.append((f"A{i}", "band", "Cband"))
    return triples


def test_de_leak():
    triples = [("a", "r", "b"), ("a", "r", "b"),          # exact dup
               ("b", "r", "a"),                            # symmetric mirror of (a,r,b)
               ("c", "r", "d")]
    dl = LP.de_leak(triples)
    assert dl["removed_exact_duplicates"] == 1
    assert dl["removed_symmetric"] == 1
    assert dl["n_after"] == 2                              # (a,r,b) and (c,r,d)


def test_recovery_on_structured_graph():
    res = LP.lp_measure(structured_graph(), fraction=0.25, seed=0)
    # dominant tails per relation -> the true tail ranks at the top
    assert res["hit@1"] == 1.0 and res["mrr"] == 1.0


def test_corruption_separation():
    rep = LP.health_report(structured_graph(), fraction=0.25, seed=0)
    assert rep["clean"]["mrr"] > rep["corrupted_control"]["mrr"]   # LP-Measure notices corruption
    assert rep["separation_mrr"] > 0.2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} lp-measure tests passed.")


if __name__ == "__main__":
    main()

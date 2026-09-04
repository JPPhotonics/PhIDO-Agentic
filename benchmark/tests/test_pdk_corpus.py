"""Tests for the GDSFactory candidate-corpus builder.

Requires gdsfactory (the dedicated wt-graphrag venv). Skips gracefully if absent.
"""

from __future__ import annotations

import e1_retrieval as E
import pdk_corpus as P


def test_distractor_corpus():
    try:
        import gdsfactory  # noqa: F401
    except Exception:
        print("  skip test_distractor_corpus (gdsfactory not importable)")
        return
    cands = P.build_distractor_candidates()
    assert len(cands) > 150                              # generic PDK is ~260 cells
    ids = {c["id"] for c in cands}
    assert "straight" in ids and any("mmi" in i for i in ids)
    for c in cands:                                      # every candidate is well-formed
        assert c["id"] and c["name"] and c["description"] and c["tier"] == "distractor"
    # signature-grounded: a parametric cell exposes some params in its description
    s = next(c for c in cands if c["id"] == "straight")
    assert "straight" in s["description"].lower()


def test_corpus_feeds_e1():
    """The corpus must be directly consumable by the E1 lexical arm."""
    try:
        import gdsfactory  # noqa: F401
    except Exception:
        print("  skip test_corpus_feeds_e1 (gdsfactory not importable)")
        return
    cands = P.build_distractor_candidates()              # full pool (mmi cells sort late alphabetically)
    case = E.QueryCase("q", "splits or combines optical power via multimode interference", set(), "functional")
    ranked = E.lexical_arm(case.query, cands)
    assert len(ranked) == len(cands)                     # arm ranks the whole pool
    # an mmi (multimode interference splitter) should surface near the top for this query
    assert any("mmi" in cid for cid in ranked[:10])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} pdk-corpus tests passed.")


if __name__ == "__main__":
    main()

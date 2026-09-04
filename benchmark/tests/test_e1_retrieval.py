"""Tests for the E1 retrieval harness (deterministic lexical arm)."""

from __future__ import annotations

import e1_retrieval as E


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


CANDS = [
    {"id": "mmi1x2", "name": "mmi 1x2 splitter", "description": "power splitter", "function": "splits power"},
    {"id": "dc", "name": "directional coupler", "description": "evanescent coupler", "function": "couples light"},
    {"id": "ring", "name": "ring resonator", "description": "wavelength filter", "function": "filters wavelengths"},
    {"id": "mzi", "name": "mach zehnder", "description": "interferometer modulator", "function": "modulates light"},
    {"id": "gc", "name": "grating coupler", "description": "fiber io", "function": "couples fiber"},
]


def test_query_generation():
    cases = E.generate_queries({"id": "ring", "name": "ring resonator", "function": "filters wavelengths"})
    kinds = {c.kind for c in cases}
    assert kinds == {"paraphrase", "functional"}
    assert all(c.gold == {"ring"} for c in cases)


def test_lexical_scoring():
    cases = [
        E.QueryCase("q1", "ring resonator", {"ring"}, "paraphrase"),
        E.QueryCase("q2", "a component that splits power", {"mmi1x2"}, "functional"),
    ]
    res = E.run_e1({"lexical": E.lexical_arm}, cases, CANDS)["lexical"]
    assert approx(res["pass@1"], 1.0)                 # both top-1 correct via token overlap
    assert approx(res["mrr"], 1.0)
    assert "functional" in res["by_kind"] and "paraphrase" in res["by_kind"]


def test_kg_arm_stub():
    try:
        E.kg_arm()("q", CANDS)
        raise AssertionError("kg_arm stub should raise until a retriever is supplied")
    except NotImplementedError:
        pass
    # with a retriever it works
    arm = E.kg_arm(lambda q, cands: ["ring"])
    assert arm("q", CANDS) == ["ring"]


def test_scaling_includes_gold():
    cases = [E.QueryCase("q1", "ring resonator", {"ring"}, "paraphrase")]
    sc = E.run_scaling({"lexical": E.lexical_arm}, cases, CANDS, sizes=(3, 5))
    for size, block in sc.items():
        pool_ids = "ring"  # gold must always be present
        assert any(c["id"] == "ring" for c in E.sample_pool(CANDS, size, {"ring"}))
        assert block["pool_size"] <= 5
    assert approx(sc[3]["arms"]["lexical"]["pass@1"], 1.0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} e1-retrieval tests passed.")


if __name__ == "__main__":
    main()

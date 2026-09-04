"""Tests for B3 topology parsing + scoring against the real GETTING_STARTED gold.

Run: PYTHONPATH=<wt-graphrag>/benchmark <venv>/bin/python <wt-graphrag>/benchmark/tests/test_topology_eval.py
"""

from __future__ import annotations

import topology_eval as TE


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_parse_level2():
    gold = TE.load_gold_topologies()
    assert "Level 2" in gold
    t = gold["Level 2"]
    assert t.nodes == {"N1": "mzi_2x2_pn_diode", "N2": "mzi_2x2_pn_diode"}
    assert t.edges == [frozenset({"N1.o3", "N2.o2"})]
    assert len(t.external) == 6
    assert TE.validity(t)["all_refs_exist"] is True


def test_all_levels_parse():
    gold = TE.load_gold_topologies()
    assert {"Level 1", "Level 2", "Level 3", "Level 4"} <= set(gold)
    assert len(gold["Level 1"].nodes) == 1                    # single directional coupler
    assert all(c == "mzi_1x2_pindiode_cband" for c in gold["Level 3"].nodes.values())
    assert len(gold["Level 3"].nodes) == 7                    # 1+2+4 MZI tree
    assert len(gold["Level 4"].nodes) == 63                   # 15 MMI + 16 VOA + 16 heater + 16 GC


def test_score_perfect_and_perturbed():
    gold = TE.load_gold_topologies()["Level 2"]
    perfect = TE.score_topology(gold, gold)
    assert approx(perfect["typed_edge_prf"]["f1"], 1.0)
    assert approx(perfect["component_prf"]["f1"], 1.0)
    assert approx(perfect["approx_ged"]["normalized_ged"], 0.0)

    missing_edge = TE.Topology(nodes=dict(gold.nodes), edges=[], external=dict(gold.external))
    s = TE.score_topology(missing_edge, gold)
    assert approx(s["typed_edge_prf"]["recall"], 0.0)         # gold edge not recovered

    wrong_comp = TE.Topology(nodes={"N1": "mzi_2x2_pn_diode", "N2": "straight"},
                             edges=list(gold.edges), external=dict(gold.external))
    s2 = TE.score_topology(wrong_comp, gold)
    assert s2["component_prf"]["f1"] < 1.0                    # wrong component type penalized
    assert s2["typed_edge_prf"]["f1"] < 1.0                   # typed edge endpoint changed


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} topology-eval tests passed.")


if __name__ == "__main__":
    main()

"""Fixtures for the edge-verified MZI collapse. Constructed topologies (the o1 3-cond run stored
no edges), covering both plausible agentic wirings + negative cases. Scored against gold to
confirm a decomposition now ties the atomic-cell realisation."""
from collections import Counter

from b3_eval import _entry_to_topology, load_gold
from hierarchical_collapse import collapse
from topology_eval import Topology, score_topology

_, GOLD = load_gold()


def T(nodes, edges, external=None):
    return Topology(nodes=dict(nodes), edges=[frozenset(e) for e in edges], external=dict(external or {}))


def comp_counts(topo):
    return Counter(topo.nodes.values())


def test_inline_phase_2x2_mzi():
    """L3_1 style: MMI_2x2 -> phase(inline) -> MMI_2x2, two arms. Gold {MZI_2x2}."""
    g = T({"C1": "MMI_2x2", "P1": "_ELECTRICAL", "P2": "_ELECTRICAL", "C2": "MMI_2x2"},
          [("C1.o3", "P1.o1"), ("P1.o2", "C2.o1"),
           ("C1.o4", "P2.o1"), ("P2.o2", "C2.o2")],
          {"o1": "C1.o1", "o2": "C1.o2", "o3": "C2.o3", "o4": "C2.o4"})
    c = collapse(g)
    assert comp_counts(c) == Counter({"MZI_2x2": 1}), comp_counts(c)
    assert score_topology(c, _entry_to_topology(GOLD["L3_1"]))["component_prf"]["f1"] == 1.0


def test_real_o1_pilot_L3_1():
    """EXACT wiring captured from the o1 agentic pilot on L3_1 (agentic_nogate): two MMI_2x2
    couplers, an inline HEATER on each arm, all four OUTER ports DANGLING (netlist declares no
    top-level I/O). Regression-locks the variant bug where dangling outer ports mis-collapsed a
    2x2 MZI to MZI_1x1. Gold {MZI_2x2}."""
    g = T({"C1": "MMI_2x2", "C2": "HEATER", "C3": "HEATER", "C4": "MMI_2x2"},
          [("C1.o3", "C2.o1"), ("C1.o4", "C3.o1"), ("C2.o2", "C4.o2"), ("C3.o2", "C4.o1")])
    c = collapse(g)
    assert comp_counts(c) == Counter({"MZI_2x2": 1}), comp_counts(c)
    assert score_topology(c, _entry_to_topology(GOLD["L3_1"]))["component_prf"]["f1"] == 1.0


def test_chain_arm_2x2_mzi():
    """Each arm is a SERIES chain of inline 2-port elements: coupler -> straight -> heater ->
    coupler. Exercises multi-hop arm traversal. (Tap-style wiring where a HEATER hangs off an
    inline straight via a separate electrical port is a known uncovered case -> conservative
    non-collapse; calibrate against real pred_edges on the first Opus run.)"""
    g = T({"C1": "MMI_2x2", "C2": "MMI_2x2",
           "S1": "STRAIGHT", "H1": "HEATER", "S2": "STRAIGHT", "H2": "HEATER"},
          [("C1.o3", "S1.o1"), ("S1.o2", "H1.o1"), ("H1.o2", "C2.o1"),
           ("C1.o4", "S2.o1"), ("S2.o2", "H2.o1"), ("H2.o2", "C2.o2")],
          {"o1": "C1.o1", "o2": "C1.o2", "o3": "C2.o3", "o4": "C2.o4"})
    c = collapse(g)
    assert comp_counts(c) == Counter({"MZI_2x2": 1}), comp_counts(c)


def test_1x1_mzi_with_gcs():
    """L3_4 style: GC - [DC_2x2 + straight arms + DC_2x2] - GC. Gold {MZI_1x1, GC, GC}."""
    g = T({"GC1": "GC", "C1": "DC_2x2", "S1": "STRAIGHT", "S2": "STRAIGHT", "C2": "DC_2x2", "GC2": "GC"},
          [("GC1.o1", "C1.o1"),
           ("C1.o3", "S1.o1"), ("S1.o2", "C2.o1"),
           ("C1.o4", "S2.o1"), ("S2.o2", "C2.o2"),
           ("C2.o3", "GC2.o1")])
    c = collapse(g)
    assert comp_counts(c) == Counter({"MZI_1x1": 1, "GC": 2}), comp_counts(c)
    assert score_topology(c, _entry_to_topology(GOLD["L3_4"]))["component_prf"]["f1"] == 1.0


def test_mzi_plus_dc_l3_6():
    """L3_6 style: decomposed MZI_2x2 + a separate DC_2x2. Gold {MZI_2x2, DC_2x2}. The extra DC
    must NOT be swallowed into the MZI."""
    g = T({"C1": "MMI_2x2", "H1": "HEATER", "H2": "HEATER", "C2": "MMI_2x2", "D": "DC_2x2"},
          [("C1.o3", "H1.o1"), ("H1.o2", "C2.o1"),
           ("C1.o4", "H2.o1"), ("H2.o2", "C2.o2"),
           ("C2.o3", "D.o1"), ("C2.o4", "D.o2")],
          {"o1": "C1.o1", "o2": "C1.o2", "o3": "D.o3", "o4": "D.o4"})
    c = collapse(g)
    assert comp_counts(c) == Counter({"MZI_2x2": 1, "DC_2x2": 1}), comp_counts(c)
    assert score_topology(c, _entry_to_topology(GOLD["L3_6"]))["component_prf"]["f1"] == 1.0


def test_no_false_collapse_unrelated_couplers():
    """Two couplers joined by ONE arm + a stray heater elsewhere is NOT an MZI — must not collapse."""
    g = T({"C1": "MMI_2x2", "C2": "MMI_2x2", "H": "HEATER"},
          [("C1.o3", "C2.o1"), ("H.o1", "C1.o1")],
          {})
    c = collapse(g)
    assert comp_counts(c) == Counter({"MMI_2x2": 2, "HEATER": 1}), comp_counts(c)


def test_idempotent_atomic_mzi():
    """An already-atomic MZI (gold shape) is a fixed point."""
    g = _entry_to_topology(GOLD["L3_1"])
    assert comp_counts(collapse(g)) == comp_counts(g)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")

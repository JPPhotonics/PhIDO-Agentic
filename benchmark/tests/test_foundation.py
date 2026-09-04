"""Smoke + correctness tests for benchmark.metrics and benchmark.stats.

Run: PYTHONPATH=benchmark .venv/bin/python benchmark/tests/test_foundation.py
(plain asserts so it needs no pytest in the bare venv).
"""

from __future__ import annotations

import math

import gate_eval as G
import metrics as M
import stats as S


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_retrieval():
    assert M.pass_at_k(["a", "b", "c"], "b", 2) is True
    assert M.pass_at_k(["a", "b", "c"], "b", 1) is False
    assert M.pass_at_k(["a", "b", "c"], {"x", "c"}, 2) is False
    assert M.pass_at_k(["a", "b", "c"], {"x", "c"}, 3) is True
    assert approx(M.reciprocal_rank(["a", "b", "c"], "b"), 0.5)
    assert approx(M.mrr([(["a", "b"], "b"), (["c"], "c")]), 0.75)
    assert approx(M.average_rank([(["a", "b", "c"], "c")]), 3.0)
    p, r, f = M.set_prf({"a", "b"}, {"b", "c"})
    assert approx(p, 0.5) and approx(r, 0.5) and approx(f, 0.5)
    assert approx(M.retrievable_coverage_at_k([(["a", "b", "c"], {"a", "c", "z"})], 3), 2 / 3)


def test_classification_topology():
    p, r, f = M.prf(2, 1, 1)
    assert approx(p, 2 / 3) and approx(r, 2 / 3) and approx(f, 2 / 3)
    cm = M.confusion_matrix([("x", "x"), ("x", "y"), ("y", "y")])
    assert cm["x"]["x"] == 1 and cm["x"]["y"] == 1 and cm["y"]["y"] == 1
    p, r, f = M.edge_prf({(1, 2), (2, 3)}, {(1, 2)})
    assert approx(p, 0.5) and approx(r, 1.0)
    g = M.approx_ged({1: "A", 2: "B"}, {(1, 2)}, {1: "A", 2: "B", 3: "C"}, {(1, 2), (2, 3)})
    assert g["node_diff"] == 1 and g["edge_diff"] == 1 and g["ged"] == 2 and approx(g["normalized_ged"], 0.4)


def test_stats_proportion_paired():
    lo, hi = S.wilson(8, 10)
    assert 0.4 < lo < 0.8 and 0.9 < hi <= 1.0
    assert S.mcnemar(10, 2)["p_value"] < 0.05 and S.mcnemar(10, 2)["method"] == "exact"
    big = S.mcnemar(30, 10)
    assert big["method"] == "chi2_cc" and big["p_value"] < 0.01
    pm = S.paired_mcnemar([1, 1, 0, 1], [0, 1, 0, 0])
    assert pm["b"] == 2 and pm["c"] == 0
    bc = S.bootstrap_ci([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], lambda d: sum(d) / len(d), n_boot=500, seed=1)
    assert approx(bc["point"], 0.8) and 0.0 <= bc["lo"] <= bc["hi"] <= 1.0


def test_stats_cluster_agreement():
    icc, m0 = S.icc_oneway([[1, 1, 1], [0, 0, 1]])
    assert 0.0 <= icc <= 1.0 and approx(m0, 3.0)
    ct = S.cluster_t_interval([[1, 1], [0, 1], [1, 1]])
    assert ct["n_clusters"] == 3 and ct["df"] == 2 and 0.0 <= ct["lo"] <= ct["hi"] <= 1.0
    assert approx(S.cohen_kappa([(1, 1), (0, 0), (1, 0), (0, 1)]), 0.0)
    assert approx(S.design_effect(20, 0.1), 2.9)
    assert S.cluster_t_interval([[1, 1]]).get("n_clusters") == 1  # too few -> no CI


def test_gate_eval():
    records = (
        [{"id": f"o{i}", "label": "invalid", "error_class": "open_port", "rejected": True} for i in range(3)]
        + [{"id": f"f{i}", "label": "invalid", "error_class": "floating_node", "rejected": False} for i in range(2)]
        + [{"id": f"v{i}", "label": "valid", "rejected": (i == 0)} for i in range(5)]
    )
    base = G.evaluate_gate(records)
    assert base["n_invalid"] == 5 and base["n_valid"] == 5
    assert approx(base["catch_rate"], 0.6) and approx(base["false_reject_rate"], 0.2)
    fg = G.formal_gate_eval(records)
    assert approx(fg["per_error_class"]["open_port"]["catch_rate"], 1.0)
    assert approx(fg["per_error_class"]["floating_node"]["catch_rate"], 0.0)
    assert fg["coverage_gap"] == ["floating_node"]
    assert fg["soundness"]["n_invalid_accepted"] == 2
    ab = G.funnel_ablation_delta([1, 1, 1, 0], [0, 1, 0, 0])
    assert approx(ab["delta"], 0.5) and ab["mcnemar"]["b"] == 2 and ab["mcnemar"]["c"] == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} foundation tests passed.")


if __name__ == "__main__":
    main()

"""Hermetic tests for A-series scorer functions (no live KB required).

The KB-touching diagnostics (A1 duplication sweep, A3 consolidation, A4 cluster stats, A5/A6
queries) are validated by running the harnesses against the live KB; these tests cover the pure
scoring/logic functions with synthetic inputs so they run anywhere.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python benchmark/tests/test_a_series.py
"""

from __future__ import annotations

import a1_extraction as A1
import a3_crossdoc as A3
import a5_scoring as A5
import a6_integrity as A6


def test_conformance_violations():
    rules = {
        "PERFORMS_FUNCTION": {
            "src": {"Component", "PDK_Cell"},
            "tgt": {"Design_Function"},
        }
    }
    sigs = [
        {
            "src": "Component",
            "rel": "PERFORMS_FUNCTION",
            "tgt": "Design_Function",
            "n": 5,
        },  # ok
        {
            "src": "Component",
            "rel": "PERFORMS_FUNCTION",
            "tgt": "Physical_Principle",
            "n": 1,
        },  # bad tgt
        {
            "src": "Property",
            "rel": "PERFORMS_FUNCTION",
            "tgt": "Design_Function",
            "n": 2,
        },  # bad src
        {"src": "X", "rel": "RELATED_TO", "tgt": "Y", "n": 9},  # exempt
        {"src": "A", "rel": "MYSTERY_REL", "tgt": "B", "n": 3},  # unknown type
    ]
    v = A6.conformance_violations(sigs, rules)
    bad = {(x["src"], x["rel"], x["tgt"]) for x in v}
    assert ("Component", "PERFORMS_FUNCTION", "Physical_Principle") in bad
    assert ("Property", "PERFORMS_FUNCTION", "Design_Function") in bad
    assert ("A", "MYSTERY_REL", "B") in bad
    assert (
        "Component",
        "PERFORMS_FUNCTION",
        "Design_Function",
    ) not in bad  # conformant
    assert ("X", "RELATED_TO", "Y") not in bad  # exempt
    assert len(v) == 3


def test_cosine_clusters():
    # 'a' and 'a2' have cosine 0.8; 'b' is orthogonal
    items = [("a", [1.0, 0.0, 0.0]), ("a2", [0.8, 0.6, 0.0]), ("b", [0.0, 1.0, 0.0])]
    clusters = A1._cosine_clusters(items, thr=0.7)
    assert len(clusters) == 1
    assert set(clusters[0]) == {"a", "a2"}
    # raising the threshold past their similarity (0.8) dissolves the cluster
    assert A1._cosine_clusters(items, thr=0.9) == []


def test_score_extraction_types_and_nil_and_dup():
    gold = [
        {"name": "MZI", "type": "Architecture", "is_new": False},
        {"name": "ring", "type": "Component", "is_new": True},
    ]
    pred = [
        {"name": "MZI", "type": "Architecture", "is_new": False},
        {"name": "ring", "type": "Component", "is_new": True},
        {"name": "MZI", "type": "Architecture", "is_new": False},
    ]  # duplicate
    out = A1.score_extraction(pred, gold)
    assert out["duplication_rate"] > 0  # one repeated name
    # NIL: 'ring' correctly flagged new -> recall on new = 1.0
    _p, r, _f1 = out["nil_prf"]
    assert r == 1.0


def test_recall_at_k():
    gold = ["x", "y", "z"]
    retrieved = [["a", "x"], ["q", "r", "y"], ["nope"]]
    assert (
        A3.recall_at_k(gold, retrieved, k=2) == 1 / 3
    )  # x@2 hit; y@3 misses top-2; z miss
    assert A3.recall_at_k(gold, retrieved, k=3) == 2 / 3  # y now in top-3; z still miss


def test_score_dia_gate_template():
    records = [
        {"id": "1", "label": "invalid", "rejected": True},
        {"id": "2", "label": "invalid", "rejected": False},
        {"id": "3", "label": "valid", "rejected": False},
    ]
    out = A3.score_dia_gate(records)
    assert out["catch_rate"] == 0.5
    assert out["false_reject_rate"] == 0.0


# --------------------------------------------------------------------------- A5 scoring helpers
def test_normalize_label_strips_acronym_and_punctuation():
    # parenthetical acronym dropped, hyphen/whitespace collapsed, lowercased
    assert A5.normalize_label("Amplitude Modulation (AM)") == "amplitude modulation"
    assert A5.normalize_label("phase-modulation") == "phase modulation"
    assert A5.normalize_label("  Power   Splitter ") == "power splitter"
    assert A5.normalize_label("") == ""


def test_normalize_label_unifies_surface_variants_only():
    # surface variants unify ...
    assert A5.normalize_label("Mach-Zehnder Interferometer") == A5.normalize_label(
        "mach zehnder interferometer"
    )
    # ... but genuine synonyms are NOT unified (we measure vocab fragmentation, not hide it)
    assert A5.normalize_label("amplitude modulator") != A5.normalize_label(
        "amplitude modulation (AM)"
    )


def test_performs_function_gold_drops_skip_labels():
    # structural/port labels removed; real functions kept + normalized
    labels = ["modulator", "active", "amplitude modulation (AM)", "2x2"]
    gold = A5.performs_function_gold(labels)
    assert gold == {"modulator", "amplitude modulation"}
    # a purely-structural cell yields empty gold (correctly not scored)
    assert A5.performs_function_gold(["passive", "1x1"]) == set()


def test_cell_set_prf_and_unmatched():
    gold = {"modulator", "amplitude modulation"}
    pred = {"modulator", "switch"}  # 1 hit, 1 spurious, 1 missed
    tp, fp, fn = A5.cell_set_prf(pred, gold)
    assert (tp, fp, fn) == (1, 1, 1)
    assert A5.unmatched_gold(pred, gold) == {"amplitude modulation"}
    # perfect match -> no unmatched
    assert A5.unmatched_gold(gold, gold) == set()


def test_normalize_set_dedupes_and_drops_empty():
    out = A5.normalize_set(["AM (AM)", "am", "", "  "])
    # "AM (AM)" -> "am", "am" -> "am" : collapse to one; empties dropped
    assert out == {"am"}


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} A-series tests passed.")


if __name__ == "__main__":
    main()

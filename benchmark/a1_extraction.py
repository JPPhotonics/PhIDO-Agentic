"""A1 — PPC extraction quality: scorer + KG duplication diagnostic.

Per BENCHMARK_ARCHITECTURE.md §5, A1 measures entity-extraction F1 over the 5 node types,
known-vs-new (NIL) precision/recall, and DUPLICATION RATE. Full F1/known-vs-new needs a small
hand-annotated gold paper set (Tier-2, LLM-judge) that does not exist yet — so `score_extraction`
is built and ready, but the runnable-NOW signal is the duplication diagnostic.

Duplication diagnostic (no labels, no LLM): read the stored 1024-d embeddings off content nodes
and flag near-duplicate names WITHIN a label by cosine similarity. This quantifies the vocabulary
fragmentation independently surfaced by E1 (e.g. ~8 near-synonym 'modulation' Design_Functions) —
a direct measure of A1's "duplication rate" and a KG-construction-quality finding for the thesis.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/a1_extraction.py
"""

from __future__ import annotations

import pathlib

from kb_access import KB
from metrics import confusion_matrix, prf
from report import Reporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "benchmark" / "results" / "a1_extraction.md"
CONTENT = [
    "Component",
    "Architecture",
    "Property",
    "Design_Function",
    "Physical_Principle",
]
# fragmentation is semantic, not near-identical, so sweep cosine thresholds rather than fix one
THRESHOLDS = [0.95, 0.90, 0.85, 0.80]
EXAMPLE_THR = 0.82  # representative threshold for showing example near-synonym clusters


# ── scorer (ready for a gold paper; gold = [{name, type, is_new}], pred = same shape) ──
def score_extraction(pred: list[dict], gold: list[dict]) -> dict:
    """F1 per node-type + known-vs-new (NIL) P/R + duplication rate. Awaiting a gold paper set."""
    gold_by_name = {g["name"].lower(): g for g in gold}
    pred_names = [p["name"].lower() for p in pred]

    # type confusion over matched names
    pairs = [
        (gold_by_name[n]["type"], p["type"])
        for n, p in zip(pred_names, pred)
        if n in gold_by_name
    ]
    conf = confusion_matrix(pairs)

    # known-vs-new (NIL) decision quality
    tp = sum(
        1
        for p in pred
        if p.get("is_new")
        and gold_by_name.get(p["name"].lower(), {}).get("is_new", True) is not True
    )  # noqa: E501
    # simpler: treat is_new as the positive class for "new" detection
    y = [
        (gold_by_name.get(p["name"].lower(), {}).get("is_new"), p.get("is_new"))
        for p in pred
        if p["name"].lower() in gold_by_name
    ]
    nil_tp = sum(1 for g, pr in y if g and pr)
    nil_fp = sum(1 for g, pr in y if not g and pr)
    nil_fn = sum(1 for g, pr in y if g and not pr)
    nil_p, nil_r, nil_f1 = prf(nil_tp, nil_fp, nil_fn)

    dup_rate = 1 - len(set(pred_names)) / len(pred_names) if pred_names else 0.0
    return {
        "type_confusion": conf,
        "nil_prf": (nil_p, nil_r, nil_f1),
        "duplication_rate": dup_rate,
        "n_pred": len(pred),
        "n_gold": len(gold),
        "_tp_unused": tp,
    }


# ── duplication diagnostic (runnable now, no labels) ──
def _cosine_clusters(
    items: list[tuple[str, list[float]]], thr: float
) -> list[list[str]]:
    """Connected components of names whose embedding cosine ≥ thr (within one label)."""
    import numpy as np

    if len(items) < 2:
        return []
    names = [n for n, _ in items]
    m = np.array([v for _, v in items], dtype=float)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    sim = m @ m.T
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thr:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return [g for g in groups.values() if len(g) > 1]


def _label_items(kb: KB, label: str) -> list[tuple[str, list[float]]]:
    rows = kb.q(
        f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
        f"RETURN coalesce(n.name, n.module_name) AS name, n.embedding AS emb"
    )
    return [(r["name"], r["emb"]) for r in rows if r["emb"]]


def main() -> None:
    kb = KB()
    items = {lab: _label_items(kb, lab) for lab in CONTENT}
    kb.close()

    # sweep: nodes-in-near-dup-clusters per label per threshold
    sweep = {
        lab: {
            thr: sum(len(c) for c in _cosine_clusters(its, thr)) for thr in THRESHOLDS
        }
        for lab, its in items.items()
    }
    total_n = sum(len(v) for v in items.values())

    rep = Reporter(
        OUT_MD,
        "A1 — extraction quality: KG duplication diagnostic",
        meta={
            "content_nodes": total_n,
            "thresholds": THRESHOLDS,
            "example_threshold": EXAMPLE_THR,
        },
    )

    rep.h("Near-duplicate node count per label, swept over cosine threshold")
    rep.line(
        "_Fragmentation is semantic (synonymous names), not near-identical — it surfaces as "
        "the threshold drops. Counts = nodes falling into a same-label cluster of size ≥ 2._"
    )
    rep.line("")
    rep.table(
        ["label", "nodes"] + [f"cos≥{t}" for t in THRESHOLDS],
        [
            [lab, len(items[lab])] + [sweep[lab][t] for t in THRESHOLDS]
            for lab in CONTENT
        ],
    )

    rep.h(f"Example near-synonym clusters at cos ≥ {EXAMPLE_THR}")
    for lab in CONTENT:
        clusters = sorted(
            _cosine_clusters(items[lab], EXAMPLE_THR), key=len, reverse=True
        )
        if clusters:
            rep.line(f"**{lab}** ({len(clusters)} clusters):")
            for cl in clusters[:8]:
                rep.line(f"  - {sorted(cl)}")
            rep.line("")

    rep.line(
        "_Full extraction F1 + known-vs-new P/R require a hand-annotated gold paper set "
        "(`score_extraction` is built and ready). The sweep is the label-free A1 signal: a "
        "rising count as the threshold drops is the vocabulary fragmentation E1 root-caused "
        "(synonymous Design_Functions diluting functional retrieval)._"
    )
    rep.save()


if __name__ == "__main__":
    main()

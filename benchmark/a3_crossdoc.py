"""A3 — DIA cross-document integration: recall@K scorer + cross-doc consolidation diagnostic.

Per BENCHMARK_ARCHITECTURE.md §5, A3 measures whether the Database Integration Agent finds links
that span documents: candidate-retrieval recall@K (primary), a semantic-verification gate
(§6.6 gate template), and end-to-end edge precision. recall@K and the gate need gold cross-doc
links (Tier-2 LLM-judge on a fixed snapshot) that don't exist yet, so `recall_at_k` and
`score_dia_gate` are built and ready; the runnable-NOW signal is the cross-doc consolidation
diagnostic.

Cross-doc consolidation (no labels, no LLM): using `RelationshipObservation` provenance
(from_entity / to_entity / source_document), count entities that are referenced by observations
from ≥2 distinct documents — i.e. concepts the pipeline merged ACROSS papers rather than
duplicating per-paper. A healthy DIA consolidates; a broken one leaves per-document silos.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/a3_crossdoc.py
"""

from __future__ import annotations

import pathlib

from gate_eval import evaluate_gate
from kb_access import KB
from report import Reporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "benchmark" / "results" / "a3_crossdoc.md"


# ── scorers (ready for gold cross-doc links / labeled candidate verifications) ──
def recall_at_k(gold_links: list, retrieved_per_query: list[list], k: int) -> float:
    """Fraction of gold cross-doc links whose target is in the query's top-k candidates."""
    hits = sum(
        1 for gold, ret in zip(gold_links, retrieved_per_query) if gold in ret[:k]
    )
    return hits / len(gold_links) if gold_links else float("nan")


def score_dia_gate(records: list[dict]) -> dict:
    """DIA semantic-verification gate via the §6.6 template (catch/false-reject on labeled links)."""
    return evaluate_gate(records)


# ── cross-doc consolidation via EXTRACTED_FROM provenance (primary, runnable now) ──
def document_consolidation(kb: KB) -> dict:
    """Entities linked by EXTRACTED_FROM to ≥2 distinct :Document nodes.

    This is the authoritative cross-doc signal: it spans ALL provenance-linked entities (every
    content node materialized from a paper), not only the DIA global-inference proposals captured
    as RelationshipObservations. Requires the Document layer (created at ingestion / via
    benchmark/backfill_documents.py); returns n_documents=0 if it has not been materialized.
    """
    n_docs = kb.scalar("MATCH (d:Document) RETURN count(d)") or 0
    rows = kb.q(
        "MATCH (n)-[:EXTRACTED_FROM]->(d:Document) "
        "WITH labels(n)[0] AS typ, n.name AS name, count(DISTINCT d) AS docs "
        "RETURN typ, name, docs"
    )
    by_type: dict[str, dict] = {}
    multi = []
    for r in rows:
        t = by_type.setdefault(r["typ"] or "?", {"total": 0, "multi": 0})
        t["total"] += 1
        if r["docs"] >= 2:
            t["multi"] += 1
            multi.append({"entity": r["name"], "type": r["typ"] or "?", "n_docs": r["docs"]})
    multi.sort(key=lambda m: m["n_docs"], reverse=True)
    n_ent = sum(v["total"] for v in by_type.values())
    return {
        "n_documents": n_docs,
        "n_entities": n_ent,
        "n_multi_doc": len(multi),
        "by_type": by_type,
        "top": multi[:12],
    }


# ── cross-doc consolidation via RelationshipObservations (secondary / DIA-proposal view) ──
def consolidation_diagnostic(kb: KB) -> dict:
    """Entities referenced by RelationshipObservations spanning ≥2 distinct documents."""
    rows = kb.q(
        "MATCH (o:RelationshipObservation) "
        "RETURN o.from_entity AS a, o.from_entity_type AS at, "
        "o.to_entity AS b, o.to_entity_type AS bt, o.source_document AS d"
    )
    ent_docs: dict[tuple, set] = {}
    for r in rows:
        for name, typ in ((r["a"], r["at"]), (r["b"], r["bt"])):
            if name:
                ent_docs.setdefault((name, typ or "?"), set()).add(r["d"])
    multi = {e: ds for e, ds in ent_docs.items() if len(ds) >= 2}
    by_type: dict[str, dict] = {}
    for (_name, typ), ds in ent_docs.items():
        t = by_type.setdefault(typ, {"total": 0, "multi": 0})
        t["total"] += 1
        if len(ds) >= 2:
            t["multi"] += 1
    top = sorted(multi.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]
    return {
        "n_entities": len(ent_docs),
        "n_multi_doc": len(multi),
        "by_type": by_type,
        "top": [{"entity": e[0], "type": e[1], "n_docs": len(ds)} for e, ds in top],
    }


def _by_type_rows(by_type: dict) -> list:
    return [
        [t, v["total"], v["multi"], f"{v['multi'] / v['total']:.2f}" if v["total"] else "—"]
        for t, v in sorted(by_type.items())
    ]


def main() -> None:
    kb = KB()
    docdiag = document_consolidation(kb)
    diag = consolidation_diagnostic(kb)
    kb.close()

    rep = Reporter(
        OUT_MD,
        "A3 — DIA cross-document integration (consolidation diagnostic)",
        meta={
            "documents": docdiag["n_documents"],
            "entities_provenance_linked": docdiag["n_entities"],
            "multi_document_entities": docdiag["n_multi_doc"],
            "multi_doc_rate": round(docdiag["n_multi_doc"] / docdiag["n_entities"], 3)
            if docdiag["n_entities"]
            else 0.0,
            "entities_in_observations": diag["n_entities"],
        },
    )

    rep.h("Cross-document consolidation via EXTRACTED_FROM → Document (primary)")
    if docdiag["n_documents"] == 0:
        rep.line(
            "_No :Document nodes present — run the ingestion pipeline (now materializes Documents) "
            "or `benchmark/backfill_documents.py` to enable this measure._"
        )
    else:
        rep.line(
            f"_{docdiag['n_documents']} documents; entities linked by EXTRACTED_FROM to ≥2 of them "
            "are concepts merged across papers. This spans all provenance-linked entities, not only "
            "DIA global-inference proposals (the observation view below)._"
        )
        rep.line("")
        rep.table(
            ["type", "entities", "multi-doc", "multi-doc rate"],
            _by_type_rows(docdiag["by_type"]),
        )
        rep.h("Most cross-document-supported entities (EXTRACTED_FROM)")
        rep.table(
            ["entity", "type", "# documents"],
            [[r["entity"], r["type"], r["n_docs"]] for r in docdiag["top"]],
        )

    rep.h("Cross-document consolidation by entity type (RelationshipObservation view)")
    rep.line(
        "_Entities referenced by observations from ≥2 distinct documents = concepts merged "
        "across papers (the cross-doc integration DIA exists to perform)._"
    )
    rep.line("")
    rep.table(
        ["type", "entities", "multi-doc", "multi-doc rate"],
        [
            [
                t,
                v["total"],
                v["multi"],
                f"{v['multi'] / v['total']:.2f}" if v["total"] else "—",
            ]
            for t, v in sorted(diag["by_type"].items())
        ],
    )

    rep.h("Most cross-document-supported entities")
    rep.table(
        ["entity", "type", "# documents"],
        [[r["entity"], r["type"], r["n_docs"]] for r in diag["top"]],
    )

    rep.line("")
    rep.line(
        "_Candidate-retrieval recall@K and the DIA semantic-verification gate (§6.6) require "
        "gold cross-doc links on a fixed snapshot (`recall_at_k` / `score_dia_gate` are built "
        "and ready). Note: cross-doc EDGES are DIA-inferred (not backed by a single "
        "observation), so edge-level recall needs the gold-link set; the entity-consolidation "
        "rates above are the label-free A3 signal._"
    )
    rep.save()


if __name__ == "__main__":
    main()

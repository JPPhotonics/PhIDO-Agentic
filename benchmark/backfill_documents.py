"""Backfill :Document nodes + EXTRACTED_FROM provenance into an EXISTING KB.

Why: the paper-ingestion pipeline historically never created Document nodes (the `add_document`
helper existed but was never called), so provenance survived only as string properties
(`source` on entities, `source_document` on edges). That blocks A3 (cross-document integration)
and any document-level graph reasoning, and leaves the schema's `EXTRACTED_FROM -> Document`
relationship + DIA's `REVIEW_OF -> Document` link permanently unsatisfiable.

This migration reconstructs the Document layer from the provenance already in the graph WITHOUT a
multi-hour rebuild. It discovers every paper id from edge `source_document` values (the
authoritative, all-17-papers signal) plus any content-node `source` that looks like a paper id,
then calls `Neo4jClient.materialize_document` for each — the SAME method the ingestion pipeline now
runs per paper, so backfilled and freshly-ingested graphs are identical.

Idempotent (MERGE throughout) — safe to re-run. Run:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/backfill_documents.py
"""

from __future__ import annotations

import re

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

# Paper ids in this corpus are stem-like: "06_Reed_Silicon_optical_modulators_NatPhoton2010".
# Require a leading NN_ so "Chapter 7, Section 7.3.3." style section-citation sources are excluded.
PAPER_ID = re.compile(r"^\d{2}_\S")


def discover_paper_ids(client: Neo4jClient) -> list[str]:
    with client.driver.session() as s:
        from_edges = [
            r["d"]
            for r in s.run(
                "MATCH ()-[e]->() WHERE e.source_document IS NOT NULL "
                "RETURN DISTINCT e.source_document AS d"
            )
        ]
        from_nodes = [
            r["d"]
            for r in s.run(
                "MATCH (n) WHERE n.source IS NOT NULL RETURN DISTINCT n.source AS d"
            )
        ]
    ids = {d for d in (from_edges + from_nodes) if d and PAPER_ID.match(d)}
    return sorted(ids)


def main() -> None:
    client = Neo4jClient(config=Neo4jConfig())
    client.connect()

    before = 0
    with client.driver.session() as s:
        before = s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]

    paper_ids = discover_paper_ids(client)
    print(f"Existing Document nodes: {before}")
    print(f"Discovered {len(paper_ids)} paper ids to materialize:\n")

    grand_total = 0
    for doc_id in paper_ids:
        res = client.materialize_document(doc_id)
        grand_total += res["total_linked"]
        print(f"  {doc_id}: {res['total_linked']:4d} entities "
              f"(edges={res['linked_via_edges']}, source={res['linked_via_source']})")

    with client.driver.session() as s:
        after = s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]
        ef = s.run("MATCH ()-[e:EXTRACTED_FROM]->() RETURN count(e) AS c").single()["c"]

    print(f"\nDocument nodes: {before} -> {after}")
    print(f"EXTRACTED_FROM edges now: {ef}  ({grand_total} entity links across papers)")
    client.close()


if __name__ == "__main__":
    main()

"""Live-KB cleanup for the two KG-construction defects whose code fixes only affect FUTURE builds.

Brings the existing graph into line with the committed fixes WITHOUT a multi-hour rebuild:

  #1 Duplicate PDK_Cell names — the docstring `Name:` fields were de-duplicated in the
     DesignLibrary, but live nodes still carry the old colliding `name`. Re-derive name/display_name
     for the affected cells and re-embed (embedding text = display_name + description + aka + labels,
     matching pdk_ingestion_agent._update_pdk_cell_embedding) so retrieval matches a fresh build.

  #2 Schema-non-conformant typed edges — DIA now demotes typed edges whose endpoint labels violate
     the schema signature (COMMIT_NONCONFORMANT_TYPED=False). This applies the same demotion to the
     edges already committed: queue a ReviewItem (reason='schema_nonconformant_typed_demoted',
     mirroring dia_agent._queue_review_item exactly) then delete the edge from the main graph.

Idempotent (MERGE / converged target state) — safe to re-run. Run:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/cleanup_kg_defects.py
"""

from __future__ import annotations

import datetime
import json
from uuid import uuid4

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

# #1: module_name -> corrected display_name (must match the edited DesignLibrary docstring `Name:`).
PDK_NAME_FIXES = {
    "_directional_coupler_adiabatic": "adiabatic_directional_coupler",
    "mrr_1x1": "mrr_1x1",
    "mrr_2x2": "mrr_2x2",
}

# typed relations are schema-checked; these generic ones are the schema's "Any" escape hatches.
EXEMPT_RELS = {"RELATED_TO", "EXTRACTED_FROM"}


def fix_duplicate_names(client: Neo4jClient) -> None:
    print("=== #1 Duplicate PDK_Cell names ===")
    with client.driver.session() as s:
        for module_name, new_name in PDK_NAME_FIXES.items():
            row = s.run(
                "MATCH (n:PDK_Cell {module_name: $m}) "
                "RETURN n.name AS name, n.description AS desc, n.aka AS aka, n.labels_list AS labels",
                m=module_name,
            ).single()
            if not row:
                print(f"  [skip] {module_name}: not found")
                continue
            if row["name"] == new_name:
                print(f"  [ok]   {module_name}: already '{new_name}'")
                continue

            text = f"{new_name} {row['desc'] or ''}"
            if row["aka"]:
                text += f" {row['aka']}"
            if row["labels"]:
                text += f" {' '.join(row['labels'])}"
            embedding = client.importer.generate_embedding(text)

            s.run(
                "MATCH (n:PDK_Cell {module_name: $m}) "
                "SET n.name = $new, n.display_name = $new"
                + (", n.embedding = $emb" if embedding else ""),
                m=module_name, new=new_name,
                **({"emb": embedding} if embedding else {}),
            )
            print(f"  [fix]  {module_name}: '{row['name']}' -> '{new_name}'"
                  + (" (+re-embedded)" if embedding else " (embedding unchanged)"))

    # verify uniqueness restored
    with client.driver.session() as s:
        coll = s.run(
            "MATCH (p:PDK_Cell) WITH toLower(coalesce(p.name,p.module_name)) AS nm, count(*) AS c "
            "WHERE c>1 RETURN collect(nm) AS dups"
        ).single()["dups"]
    print(f"  remaining duplicate PDK_Cell names: {coll or 'none'}\n")


def demote_nonconformant_edges(client: Neo4jClient) -> None:
    print("=== #2 Schema-non-conformant typed edges ===")
    with client.driver.session() as s:
        rules = {
            r["name"]: (set(r["src"] or []), set(r["tgt"] or []))
            for r in s.run(
                "MATCH (st:SchemaRelationType) RETURN st.name AS name, "
                "st.allowed_source_types AS src, st.allowed_target_types AS tgt"
            )
        }

        edges = s.run(
            "MATCH (a)-[e]->(b) "
            "RETURN elementId(e) AS eid, type(e) AS rel, labels(a)[0] AS src, labels(b)[0] AS tgt, "
            "a.name AS an, b.name AS bn, properties(e) AS props"
        )
        bad = []
        for r in edges:
            rel = r["rel"]
            if rel in EXEMPT_RELS or rel not in rules:
                continue
            allowed_src, allowed_tgt = rules[rel]
            if r["src"] not in allowed_src or r["tgt"] not in allowed_tgt:
                bad.append(r)

    print(f"  found {len(bad)} non-conformant typed edges to demote")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    config = Neo4jConfig()
    for r in bad:
        props = r["props"] or {}
        payload = {
            "edge_collection": r["rel"],
            "from_node": r["an"],
            "from_collection": r["src"],
            "to_node": r["bn"],
            "to_collection": r["tgt"],
            "operation": "MERGE",
            "description": props.get("description", ""),
            "evidence_quotes": props.get("evidence_quotes", []),
            "weight": props.get("weight"),
            "confidence": props.get("confidence"),
            "source_document": props.get("source_document"),
            "provenance": props.get("provenance"),
            "extracted_at": props.get("extracted_at"),
        }
        src_doc = props.get("source_document") or ""
        item_props = {
            "id": str(uuid4()),
            "item_type": "edge",
            "reason": "schema_nonconformant_typed_demoted",
            "priority": "low",
            "status": "pending",
            "created_at": now,
            "source_document": src_doc,
            "payload_json": json.dumps(payload, default=str),
        }
        with client.driver.session() as s:
            s.run("MERGE (r:ReviewItem {id: $id}) SET r += $props",
                  id=item_props["id"], props=item_props)
            if src_doc:
                s.run(
                    "MATCH (r:ReviewItem {id: $id}) MATCH (d:Document {title: $title}) "
                    "MERGE (r)-[:REVIEW_OF]->(d)",
                    id=item_props["id"], title=src_doc,
                )
            s.run("MATCH ()-[e]->() WHERE elementId(e) = $eid DELETE e", eid=r["eid"])
        print(f"  [demote] {r['src']} {r['an']!r} -[{r['rel']}]-> {r['tgt']} {r['bn']!r}")

    with client.driver.session() as s:
        remaining = 0
        for r in s.run(
            "MATCH (a)-[e]->(b) RETURN type(e) AS rel, labels(a)[0] AS src, labels(b)[0] AS tgt"
        ):
            rel = r["rel"]
            if rel in EXEMPT_RELS or rel not in rules:
                continue
            allowed_src, allowed_tgt = rules[rel]
            if r["src"] not in allowed_src or r["tgt"] not in allowed_tgt:
                remaining += 1
        nq = s.run(
            "MATCH (r:ReviewItem {reason:'schema_nonconformant_typed_demoted'}) RETURN count(r) AS c"
        ).single()["c"]
    print(f"  remaining non-conformant typed edges: {remaining}")
    print(f"  ReviewItems(reason=schema_nonconformant_typed_demoted): {nq}\n")


def main() -> None:
    client = Neo4jClient(config=Neo4jConfig())
    client.connect()
    fix_duplicate_names(client)
    demote_nonconformant_edges(client)
    client.close()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()

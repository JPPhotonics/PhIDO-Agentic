#!/usr/bin/env python3
"""Rebuild the photonics knowledge graph from this export into a Neo4j 5.x instance.

Usage:
    pip install neo4j
    python import_kg.py --uri bolt://localhost:7687 --user neo4j --password <pw> \
        [--nodes nodes.jsonl] [--rels relationships.jsonl]

The target database should be empty. Node identity is preserved through a temporary
`_import_id` property (removed at the end); every label, relationship type, and property
in the export is restored verbatim. The source graph stores timestamps as ISO-8601 strings; pass --retype-datetimes to
convert the keys in DATETIME_KEYS to native Neo4j datetimes instead (a deliberate
deviation from the source).

Constraints and vector indexes are NOT created here: they are created idempotently by
the accompanying code release (PhotonicsAI/KnowledgeBase/Neo4j/client.py,
Neo4jClient.initialize(), https://github.com/JPPhotonics/PhIDO-Agentic), which should be
run once after this import. Alternatively pass --schema to create the uniqueness
constraints and cosine vector indexes directly.
"""
import argparse
import json

from neo4j import GraphDatabase

DATETIME_KEYS = {"created_at", "updated_at", "timestamp", "superseded_at"}
BATCH = 500
VECTOR_LABELS = ["Component", "Architecture", "Property", "Design_Function",
                 "Physical_Principle", "PDK_Cell"]
UNIQUE = [("Architecture", ["name"]), ("Component", ["name"]), ("Design_Function", ["name"]),
          ("Document", ["title"]), ("Physical_Principle", ["name"]), ("Property", ["name"]),
          ("PDK_Cell", ["pdk_name", "module_name"]),
          ("PDK_Cell_History", ["pdk_name", "module_name", "pdk_version"])]


def batches(path):
    buf = []
    with open(path) as f:
        for line in f:
            buf.append(json.loads(line))
            if len(buf) >= BATCH:
                yield buf
                buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--database", default="neo4j")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", required=True)
    ap.add_argument("--nodes", default="nodes.jsonl")
    ap.add_argument("--rels", default="relationships.jsonl")
    ap.add_argument("--retype-datetimes", action="store_true",
                    help="convert ISO timestamp strings to native datetimes")
    ap.add_argument("--schema", action="store_true",
                    help="also create uniqueness constraints and cosine vector indexes")
    args = ap.parse_args()

    drv = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with drv.session(database=args.database) as s:
        existing = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if existing:
            raise SystemExit(f"target database is not empty ({existing} nodes); aborting")

        nn = 0
        for batch in batches(args.nodes):
            rows = []
            for r in batch:
                props = dict(r["properties"])
                props["_import_id"] = r["id"]
                rows.append({"labels": r["labels"], "props": props})
            s.run("""
                UNWIND $rows AS row
                CALL apoc.create.node(row.labels, row.props) YIELD node
                RETURN count(node)
            """, rows=rows)
            nn += len(batch)
        print(f"nodes created: {nn}")

        s.run("CREATE INDEX _import_id_idx IF NOT EXISTS FOR (n:Component) ON (n._import_id)")
        # a global lookup across labels needs no index for a graph of this size; join in batches
        nr = 0
        for batch in batches(args.rels):
            rows = [{"s": r["start"], "t": r["end"], "ty": r["type"],
                     "props": r["properties"]} for r in batch]
            s.run("""
                UNWIND $rows AS row
                MATCH (a {_import_id: row.s}), (b {_import_id: row.t})
                CALL apoc.create.relationship(a, row.ty, row.props, b) YIELD rel
                RETURN count(rel)
            """, rows=rows)
            nr += len(batch)
        print(f"relationships created: {nr}")

        if args.retype_datetimes:
          for k in sorted(DATETIME_KEYS):
            s.run(f"""
                MATCH (n) WHERE n.{k} IS NOT NULL AND valueType(n.{k}) STARTS WITH 'STRING'
                SET n.{k} = datetime(n.{k})
            """)
        s.run("MATCH (n) REMOVE n._import_id")
        s.run("DROP INDEX _import_id_idx IF EXISTS")

        if args.schema:
            for label, keys in UNIQUE:
                keyexpr = ", ".join(f"n.{k}" for k in keys)
                s.run(f"CREATE CONSTRAINT {label.lower()}_import_unique IF NOT EXISTS "
                      f"FOR (n:{label}) REQUIRE ({keyexpr}) IS UNIQUE")
            dim = None
            for rec in s.run("MATCH (n:Component) WHERE n.embedding IS NOT NULL "
                             "RETURN size(n.embedding) AS d LIMIT 1"):
                dim = rec["d"]
            if dim:
                for label in VECTOR_LABELS:
                    s.run(f"""
                        CREATE VECTOR INDEX {label.lower()}_embedding_index IF NOT EXISTS
                        FOR (n:{label}) ON (n.embedding)
                        OPTIONS {{indexConfig: {{`vector.dimensions`: {dim},
                                                 `vector.similarity_function`: 'cosine'}}}}
                    """)
            print("schema created")

        n2 = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r2 = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"final: {n2} nodes, {r2} relationships (export: 2707 / 3333)")
    drv.close()


if __name__ == "__main__":
    main()

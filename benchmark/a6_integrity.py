"""A6 — Neo4j KG integrity (L1) + query-latency baseline (L0). READ-ONLY, no labels, no LLM.

Per BENCHMARK_ARCHITECTURE.md §5: A6 has no standalone quality metric — it checks structural
integrity (no orphans / duplicates / contradictions / schema violations) and operational cost
(query latency vs graph size). This harness runs the integrity constraints as Cypher and times a
set of representative read queries.

Integrity findings are reported as (count, verdict) where verdict is ok / CONCERN. "CONCERN" is
not necessarily a bug — e.g. RELATED_TO residual and the review backlog are expected interim
state from incremental SEA recategorization — but they're surfaced as KG-health signals.

Latency here is a SINGLE-SIZE baseline (one KB snapshot); a true scalability curve needs the KB
captured at several sizes (50/131/262-cell scaling, Phase −1) — flagged, not faked.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/a6_integrity.py
"""

from __future__ import annotations

import pathlib
import time

from kb_access import KB
from report import Reporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "benchmark" / "results" / "a6_integrity.md"

# content labels (exclude internal bookkeeping labels from coverage/orphan expectations)
CONTENT = [
    "Component",
    "Architecture",
    "Property",
    "Design_Function",
    "Physical_Principle",
    "PDK_Cell",
]
INTERNAL = [
    "ReviewItem",
    "RelationshipObservation",
    "SchemaRelationType",
    "PDK_Cell_History",
    "Document",
]


def integrity(kb: KB) -> list[dict]:
    """Each finding: {check, count, detail, verdict}."""
    out = []

    # 1. orphan content nodes (degree 0) — content nodes should be connected
    orphans = kb.q(
        "MATCH (n) WHERE labels(n)[0] IN $c AND NOT (n)--() "
        "RETURN labels(n)[0] AS l, count(*) AS n",
        c=CONTENT,
    )
    n_orphan = sum(r["n"] for r in orphans)
    out.append(
        {
            "check": "orphan content nodes (degree 0)",
            "count": n_orphan,
            "detail": {r["l"]: r["n"] for r in orphans if r["n"]} or "none",
            "verdict": "ok" if n_orphan == 0 else "CONCERN",
        }
    )

    # 2. duplicate names within a label (case/space-insensitive) — fragmentation/merge failure
    dups = kb.q(
        "MATCH (n) WHERE labels(n)[0] IN $c "
        "WITH labels(n)[0] AS l, toLower(trim(coalesce(n.name, n.module_name, n.title, ''))) AS nm, count(*) AS c "
        "WHERE nm <> '' AND c > 1 RETURN l, nm, c ORDER BY c DESC",
        c=CONTENT,
    )
    out.append(
        {
            "check": "duplicate names within label (case-insensitive)",
            "count": len(dups),
            "detail": [f"{d['l']}:{d['nm']}(x{d['c']})" for d in dups[:8]] or "none",
            "verdict": "ok" if not dups else "CONCERN",
        }
    )

    # 3. self-loops (a node related to itself)
    loops = kb.scalar("MATCH (n)-[e]->(n) RETURN count(e) AS c")
    out.append(
        {
            "check": "self-loop edges",
            "count": loops or 0,
            "detail": "—",
            "verdict": "ok" if not loops else "CONCERN",
        }
    )

    # 4. duplicate triples (same src,type,tgt more than once)
    dtrip = kb.scalar(
        "MATCH (a)-[e]->(b) WITH a, type(e) AS t, b, count(*) AS c WHERE c > 1 RETURN count(*) AS d"
    )
    out.append(
        {
            "check": "duplicate triples (src,type,tgt repeated)",
            "count": dtrip or 0,
            "detail": "—",
            "verdict": "ok" if not dtrip else "CONCERN",
        }
    )

    # 5. RELATED_TO residual (generic edges not yet recategorized by SEA)
    rel = kb.scalar("MATCH ()-[e:RELATED_TO]->() RETURN count(e) AS c") or 0
    total = kb.scalar("MATCH ()-[e]->() RETURN count(e) AS c") or 1
    out.append(
        {
            "check": "RELATED_TO residual (generic edges)",
            "count": rel,
            "detail": f"{100 * rel / total:.1f}% of all edges",
            "verdict": "ok"
            if rel == 0
            else "CONCERN (interim recategorization backlog)",
        }
    )

    # 6. review backlog (low-confidence items queued for human review)
    rv = kb.scalar("MATCH (r:ReviewItem) RETURN count(r) AS c") or 0
    out.append(
        {
            "check": "ReviewItem backlog",
            "count": rv,
            "detail": "queued, awaiting review",
            "verdict": "ok" if rv == 0 else "CONCERN (human-review backlog)",
        }
    )

    # 7. embedding coverage on content nodes (needed for semantic retrieval)
    cov = kb.q(
        "MATCH (n) WHERE labels(n)[0] IN $c "
        "RETURN labels(n)[0] AS l, count(*) AS tot, count(n.embedding) AS emb",
        c=CONTENT,
    )
    missing = sum(r["tot"] - r["emb"] for r in cov)
    out.append(
        {
            "check": "missing embeddings on content nodes",
            "count": missing,
            "detail": {r["l"]: f"{r['emb']}/{r['tot']}" for r in cov},
            "verdict": "ok" if missing == 0 else "CONCERN",
        }
    )

    return out


def schema_signatures(kb: KB) -> list[dict]:
    """Distinct (sourceLabel)-[TYPE]->(targetLabel) signatures actually present — schema conformance view."""
    return kb.q(
        "MATCH (a)-[e]->(b) "
        "RETURN labels(a)[0] AS src, type(e) AS rel, labels(b)[0] AS tgt, count(*) AS n "
        "ORDER BY n DESC"
    )


def schema_rules(kb: KB) -> dict[str, dict]:
    """Allowed source/target labels per relationship type, loaded from SchemaRelationType nodes."""
    return {
        r["name"]: {"src": set(r["src"] or []), "tgt": set(r["tgt"] or [])}
        for r in kb.q(
            "MATCH (s:SchemaRelationType) "
            "RETURN s.name AS name, s.allowed_source_types AS src, "
            "s.allowed_target_types AS tgt"
        )
    }


def conformance_violations(sigs: list[dict], rules: dict[str, dict]) -> list[dict]:
    """Present (src,rel,tgt) signatures that violate the schema's allowed types.

    RELATED_TO / EXTRACTED_FROM are the generic 'Any' escape hatches — exempt from typing.
    REVIEW_OF (ReviewItem→Document) and SUPERSEDES (PDK_Cell→PDK_Cell_History) are internal
    bookkeeping edges over non-content nodes, not part of the content schema — also exempt.
    A signature whose rel has no rule, or whose src/tgt is outside the allowed set, is a violation.
    """
    exempt = {"RELATED_TO", "EXTRACTED_FROM", "REVIEW_OF", "SUPERSEDES"}
    out = []
    for s in sigs:
        rel = s["rel"]
        if rel in exempt:
            continue
        rule = rules.get(rel)
        if rule is None:
            out.append({**s, "why": "unknown relationship type (not in schema)"})
        elif s["src"] not in rule["src"] or s["tgt"] not in rule["tgt"]:
            out.append(
                {
                    **s,
                    "why": f"expected {sorted(rule['src'])}-[{rel}]->{sorted(rule['tgt'])}",
                }
            )
    return out


def latency_baseline(kb: KB, reps: int = 5) -> list[dict]:
    """Time representative read queries (median of `reps`). Single-size baseline, not a curve."""
    probes = {
        "label scan (Component)": "MATCH (n:Component) RETURN count(n)",
        "2-hop traversal (PDK_Cell→fn→cell)": "MATCH (p:PDK_Cell)-[:PERFORMS_FUNCTION]->(:Design_Function)<-[:PERFORMS_FUNCTION]-(q:PDK_Cell) "
        "RETURN count(DISTINCT q)",
        "edge aggregation (all types)": "MATCH ()-[e]->() RETURN type(e), count(*)",
        "property filter (Property w/ units)": "MATCH (n:Property) WHERE n.units IS NOT NULL RETURN count(n)",
    }
    out = []
    for name, cy in probes.items():
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            kb.q(cy)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        out.append(
            {
                "query": name,
                "median_ms": round(ts[len(ts) // 2], 1),
                "min_ms": round(ts[0], 1),
                "max_ms": round(ts[-1], 1),
            }
        )
    return out


def main() -> None:
    kb = KB()
    labels, rels = kb.label_counts(), kb.rel_counts()
    n_nodes, n_edges = sum(labels.values()), sum(rels.values())

    findings = integrity(kb)
    sigs = schema_signatures(kb)
    rules = schema_rules(kb)
    violations = conformance_violations(sigs, rules)
    n_viol_edges = sum(v["n"] for v in violations)
    findings.append(
        {
            "check": "schema-conformance violations (typed edges off allowed src/tgt)",
            "count": n_viol_edges,
            "detail": [
                f"{v['src']}-[{v['rel']}]->{v['tgt']} (x{v['n']})"
                for v in violations[:8]
            ]
            or "none",
            "verdict": "ok" if not violations else "CONCERN",
        }
    )
    lat = latency_baseline(kb)
    kb.close()

    n_concern = sum(1 for f in findings if f["verdict"] != "ok")
    rep = Reporter(
        OUT_MD,
        "A6 — KG integrity (L1) + latency baseline (L0)",
        meta={
            "nodes": n_nodes,
            "edges": n_edges,
            "labels": len(labels),
            "rel_types": len(rels),
            "integrity_concerns": n_concern,
        },
    )

    rep.h("Integrity constraints")
    rep.table(
        ["check", "count", "verdict", "detail"],
        [[f["check"], f["count"], f["verdict"], f["detail"]] for f in findings],
    )

    if violations:
        rep.h(
            "Schema-conformance VIOLATIONS (typed edges off their allowed source/target)"
        )
        rep.table(
            ["source", "rel", "target", "count", "expected"],
            [[v["src"], v["rel"], v["tgt"], v["n"], v["why"]] for v in violations],
        )

    rep.h("Schema conformance — (source)-[REL]->(target) signatures present")
    rep.table(
        ["source", "rel", "target", "count"],
        [[s["src"], s["rel"], s["tgt"], s["n"]] for s in sigs],
    )

    rep.h("Query-latency baseline (single KB snapshot — not a scaling curve)")
    rep.line(
        f"_{n_nodes} nodes / {n_edges} edges. A true scalability curve needs the KB at "
        f"multiple sizes (50/131/262-cell scaling) — out of scope for one snapshot._"
    )
    rep.line("")
    rep.table(
        ["query", "median ms", "min ms", "max ms"],
        [[x["query"], x["median_ms"], x["min_ms"], x["max_ms"]] for x in lat],
    )
    rep.save()


if __name__ == "__main__":
    main()

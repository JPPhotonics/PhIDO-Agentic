"""E1 retrieval benchmark driver — lexical vs KG vs KG+paper-enrichment.

Builds the candidate pool and query cases from the live KB's ``PDK_Cell`` nodes, then
runs three arms on the same queries + same candidates (KG / enrichment is the only
treatment):
  * ``lexical``      — token-overlap over name+description+function (stand-in for search_components)
  * ``kg``           — KG retrieval, ontology+PDK-native edges only (enrichment OFF)
  * ``kg+enrich``    — KG retrieval including paper-propagated edges (enrichment ON)  [decision #4]

Gold is the SOURCE cell for each query (arm-independent), so the arm comparison is fair.
Queries: a name paraphrase + up to N functional queries per cell. Functional queries
exercise PERFORMS_FUNCTION, where the KG (and the paper enrichment) should help most.

Scope: 34-cell DemoPDK (the buildable universe). The 262-component generic library is a
retrieval-only future extension (distractors aren't end-to-end buildable; not wired here).

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))            # metrics.py / report.py / e1_retrieval.py / kg_retrieve.py
sys.path.append(str(HERE.parent))     # PhotonicsAI

from e1_retrieval import lexical_arm, QueryCase, run_e1     # noqa: E402
from kg_retrieve import make_kg_retriever                   # noqa: E402
from report import Reporter                                 # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient   # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig   # noqa: E402

# Optional CLI arg: path to a query-set JSON (defaults to the curated worked-example set).
QUERIES_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "e1_queries.json"
OUT = HERE / "results" / f"e1_retrieval_{QUERIES_FILE.stem}.json"


def build_candidates(client) -> list[dict]:
    """Candidates = all PDK_Cells, exposing only name+description (what a real catalog search
    sees). The PERFORMS_FUNCTION 'function' field is deliberately NOT exposed — it is KG-derived
    and would leak the answer to the lexical arm. The KG arms read structure from the KB directly."""
    cy = """
    MATCH (p:PDK_Cell)
    RETURN p.module_name AS id,
           coalesce(p.display_name, p.module_name) AS name,
           coalesce(p.description, '') AS description
    ORDER BY id
    """
    with client.driver.session() as s:
        return [{"id": r["id"], "name": r["name"], "description": r["description"]}
                for r in s.run(cy) if r["id"]]


def load_queries(candidate_ids: set[str]) -> list[QueryCase]:
    """Load curated design-intent queries; gold is a SET of acceptable modules (arm-independent)."""
    spec = json.loads(QUERIES_FILE.read_text())
    cases: list[QueryCase] = []
    for q in spec["queries"]:
        gold = {m for m in q["gold"] if m in candidate_ids}
        missing = [m for m in q["gold"] if m not in candidate_ids]
        if missing:
            print(f"  [warn] query {q['id']}: gold modules not in KB: {missing}")
        if not gold:
            print(f"  [skip] query {q['id']}: no gold module present in KB")
            continue
        cases.append(QueryCase(q["id"], q["query"], gold, q.get("kind", "intent")))
    return cases


def main() -> None:
    client = Neo4jClient(config=Neo4jConfig())
    client.connect()

    candidates = build_candidates(client)
    cases = load_queries({c["id"] for c in candidates})
    print(f"Candidates (PDK_Cells): {len(candidates)} | curated design-intent queries: {len(cases)}")

    # Share ONE client across KG arms (avoids reloading the CPU embedding model per arm).
    arms = {
        "lexical": lexical_arm,
        "kg": make_kg_retriever(include_enrichment=False, client=client),
        "kg+enrich": make_kg_retriever(include_enrichment=True, client=client),
        # 2026-06-25: embedding-only KG arm (functional gate OFF) — the empirically better retriever
        # (the PERFORMS_FUNCTION functional boost is net-harmful; see E1_RETRIEVAL_FINDINGS.md).
        "kg_embed": make_kg_retriever(include_enrichment=False, client=client, functional=False),
    }

    results = run_e1(arms, cases, candidates, ks=(1, 3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "34-cell DemoPDK; curated design-intent queries; gold = domain-correct module set (arm-independent); lexical sees name+description only",
        "n_candidates": len(candidates),
        "n_queries": len(cases),
        "arms": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[json] -> {OUT}")

    # ---- human-readable report ----
    rep = Reporter(OUT.with_suffix(".md"), "E1 retrieval: lexical vs KG vs KG+enrichment",
                   meta={"candidates": len(candidates), "queries": len(cases),
                         "scope": "34-cell DemoPDK", "ablation": "decision #4 (paper enrichment)"})
    rep.h("Overall (all queries)")
    rep.table(
        ["arm", "pass@1", "pass@3", "mrr", "coverage@3", "set_prec@1"],
        [[name, f"{r['pass@1']:.3f}", f"{r['pass@3']:.3f}", f"{r['mrr']:.3f}",
          f"{r['coverage@3']:.3f}", f"{r['set_precision@1']:.3f}"] for name, r in results.items()],
    )
    rep.h("By query kind (pass@1 / mrr) — KG should help most on 'functional'")
    kinds = sorted({k for r in results.values() for k in r["by_kind"]})
    rep.table(
        ["arm", *[f"{k}:pass@1" for k in kinds], *[f"{k}:mrr" for k in kinds]],
        [[name, *[f"{r['by_kind'].get(k,{}).get('pass@1', float('nan')):.3f}" for k in kinds],
          *[f"{r['by_kind'].get(k,{}).get('mrr', float('nan')):.3f}" for k in kinds]]
         for name, r in results.items()],
    )
    rep.h("Read")
    rep.line("- `kg+enrich` vs `kg` isolates the paper-enrichment contribution (decision #4).")
    rep.line("- `kg` vs `lexical` isolates KG structure vs token overlap (decision #1).")
    rep.line("- Queries are leak-resistant **design intents** (user language, not PDK docstrings); "
             "gold is the domain-correct module SET, arm-independent. Lexical sees name+description "
             "only (the KG-derived function field is withheld to avoid leakage).")
    rep.line(f"- DRAFT query set ({len(cases)} q over 34 cells) — gold needs Poon-group sign-off; "
             "small N, read per-query, not as a powered estimate. Source: GETTING_STARTED worked "
             "examples + unambiguous single-type intents.")
    rep.save()


if __name__ == "__main__":
    main()

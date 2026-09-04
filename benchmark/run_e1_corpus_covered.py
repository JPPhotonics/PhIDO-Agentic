"""Decision-#4 pre-check: does paper enrichment help retrieval specifically on the queries whose
gold cell the corpus actually enriched?

Splits the testbench queries into:
  * COVERED   — at least one gold cell has a literature PERFORMS_FUNCTION edge (provenance=
                inferred_from_literature) — the exact edges kg+enrich uses that kg does not, so the
                only queries where enrichment *could* help; vs
  * UNCOVERED — no gold cell has such an edge.
Compares kg(weighted) vs kg+enrich(weighted) on each subset (rankings computed once per query, then
sliced). If enrichment adds ~nothing even on COVERED -> redundant with native PDK structure;
if it helps on COVERED -> the overall "neutral" is dilution by the uncovered targets.

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_corpus_covered.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE)); sys.path.append(str(HERE.parent))
from kg_retrieve import make_kg_retriever            # noqa: E402
from metrics import mean_pass_at_k, mrr              # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient  # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig  # noqa: E402

client = Neo4jClient(config=Neo4jConfig()); client.connect()
with client.driver.session() as s:
    candidates = [{"id": r["id"], "name": r["name"], "description": r["description"]} for r in s.run(
        "MATCH (p:PDK_Cell) RETURN p.module_name AS id, coalesce(p.display_name,p.module_name) AS name, "
        "coalesce(p.description,'') AS description")]
    covered_cells = {r["m"] for r in s.run(
        "MATCH (p:PDK_Cell)-[r:PERFORMS_FUNCTION {provenance:'inferred_from_literature'}]->() "
        "RETURN DISTINCT p.module_name AS m")}
cand_ids = {c["id"] for c in candidates}

cases = []   # (query, gold, covered?)
for q in json.load(open(HERE / "e1_queries_testbench.json"))["queries"]:
    gold = {m for m in q["gold"] if m in cand_ids}
    if gold:
        cases.append((q["query"], gold, bool(gold & covered_cells)))

kg  = make_kg_retriever(include_enrichment=False, client=client, weighted=True)
kge = make_kg_retriever(include_enrichment=True,  client=client, weighted=True)

# rank once per query per arm
ranks = [(kg(q, candidates), kge(q, candidates), gold, cov) for q, gold, cov in cases]

def report(label, subset):
    if not subset:
        print(f"\n[{label}] n=0"); return
    pk = [(rk, g) for rk, _, g, _ in subset]
    pe = [(re_, g) for _, re_, g, _ in subset]
    print(f"\n[{label}] n={len(subset)}")
    print(f"  kg(weighted)        pass@1={mean_pass_at_k(pk,1):.3f}  pass@3={mean_pass_at_k(pk,3):.3f}  mrr={mrr(pk):.3f}")
    print(f"  kg+enrich(weighted) pass@1={mean_pass_at_k(pe,1):.3f}  pass@3={mean_pass_at_k(pe,3):.3f}  mrr={mrr(pe):.3f}")
    print(f"  enrichment delta    pass@1={mean_pass_at_k(pe,1)-mean_pass_at_k(pk,1):+.3f}  "
          f"pass@3={mean_pass_at_k(pe,3)-mean_pass_at_k(pk,3):+.3f}  mrr={mrr(pe)-mrr(pk):+.3f}")

print(f"candidates={len(candidates)}  covered_cells={len(covered_cells)}/34  queries={len(cases)}")
report("ALL", ranks)
report("corpus-COVERED gold", [r for r in ranks if r[3]])
report("UNCOVERED gold", [r for r in ranks if not r[3]])

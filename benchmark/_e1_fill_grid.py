"""E1 grid completion (2026-08-27): all SIX retrieval arms on BOTH query sets.

Why: Figure 6.4 showed different arm sets per panel because two harnesses were built for two
questions (run_e1_stratified.py: 5 arms on the 231 testbench queries; run_e1_retrieval.py: 4 arms
on the 18 curated intents). This script computes the full 6x2 grid with the SAME arm definitions,
candidates, gold handling, and metric code as those harnesses, so the pre-existing cells must
reproduce exactly (asserted below) and the new cells are commensurable.

Arms (thesis label -> harness definition):
  Lexical              lexical_arm
  KG (functional)      make_kg_retriever(include_enrichment=False, weighted=True)   [= kg(weighted) / kg]
  KG + enrichment      make_kg_retriever(include_enrichment=True,  weighted=True)   [= kg+enrich]
  KG (embedding)       make_kg_retriever(include_enrichment=False, functional=False) [= kg_embed]
  Hybrid (embedding)   RRF(lexical, KG embedding), k=60                              [= hybrid_embed]
  Hybrid (production)  mcp_servers.component_retrieval.retrieve_candidates           [= hybrid(prod)]

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/_e1_fill_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE)); sys.path.append(str(HERE.parent))

try:                                   # shared embedding daemon if alive (bitwise-identical vectors)
    import _embed_proxy
    print("embed proxy:", _embed_proxy.install())
except Exception as e:                 # noqa: BLE001
    print("embed proxy unavailable, loading model locally:", e)

from e1_retrieval import QueryCase, lexical_arm, score_arm            # noqa: E402
from kg_retrieve import make_kg_retriever                             # noqa: E402
from metrics import mean_pass_at_k, mrr                               # noqa: E402
from run_e1_stratified import build_candidates, is_named               # noqa: E402
from mcp_servers.component_retrieval import retrieve_candidates       # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient        # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig        # noqa: E402

OUT = HERE / "results" / "e1_full_grid.json"
TB_FILE = HERE / "e1_queries_testbench.json"
CUR_FILE = HERE / "e1_queries.json"
COMMITTED_TB = HERE / "results" / "e1_stratified.json"
COMMITTED_CUR = HERE / "results" / "e1_retrieval_e1_queries.json"

client = Neo4jClient(config=Neo4jConfig()); client.connect()
candidates = build_candidates(client)
cand_ids = {c["id"] for c in candidates}
print(f"candidates: {len(candidates)}")

# ---- query sets, exactly as the two harnesses load them ----
tb = []
for q in json.loads(TB_FILE.read_text())["queries"]:
    gold = {m for m in q["gold"] if m in cand_ids}
    if gold:
        tb.append({"query": q["query"], "gold": gold,
                   "stratum": "named" if is_named(q["query"]) else "functional"})
cur = []
for q in json.loads(CUR_FILE.read_text())["queries"]:
    gold = {m for m in q["gold"] if m in cand_ids}
    if gold:
        cur.append(QueryCase(q["id"], q["query"], gold, q.get("kind", "intent")))
print(f"testbench: {len(tb)} (named={sum(q['stratum']=='named' for q in tb)}) | curated: {len(cur)}")

# ---- arms ----
kg_func = make_kg_retriever(include_enrichment=False, client=client, weighted=True)
kg_func_enrich = make_kg_retriever(include_enrichment=True, client=client, weighted=True)
kg_embed = make_kg_retriever(include_enrichment=False, client=client, weighted=True, functional=False)

def _rrf(query, cands, arms_to_fuse, k=60):
    scores = {c["id"]: 0.0 for c in cands}
    for arm in arms_to_fuse:
        for rank_i, cid in enumerate(arm(query, cands)):
            if cid in scores:
                scores[cid] += 1.0 / (k + rank_i + 1)
    return sorted(scores, key=lambda cid: (-scores[cid], str(cid)))

def hybrid_embed(query, cands):
    return _rrf(query, cands, [lexical_arm, kg_embed])

def hybrid_prod(query, cands):
    r = json.loads(retrieve_candidates(query, top_k=len(cands), client=client))
    return [c["module_name"] for c in r.get("results", [])]

ARMS = {
    "Lexical": lexical_arm,
    "KG (functional)": kg_func,
    "KG + enrichment": kg_func_enrich,
    "KG (embedding)": kg_embed,
    "Hybrid (embedding)": hybrid_embed,
    "Hybrid (production)": hybrid_prod,
}

def tb_metrics(pairs):
    return {"n": len(pairs), "pass@1": mean_pass_at_k(pairs, 1),
            "pass@3": mean_pass_at_k(pairs, 3), "mrr": mrr(pairs)}

grid = {"testbench": {}, "curated": {}}
for name, arm in ARMS.items():
    ranked = [(arm(q["query"], candidates), q["gold"], q["stratum"]) for q in tb]
    grid["testbench"][name] = {
        "overall": tb_metrics([(r, g) for r, g, _ in ranked]),
        "named": tb_metrics([(r, g) for r, g, s in ranked if s == "named"]),
        "functional": tb_metrics([(r, g) for r, g, s in ranked if s == "functional"]),
    }
    grid["curated"][name] = score_arm(arm, cur, candidates, ks=(1, 3))
    print(f"done: {name}")

# ---- reproduction check against the committed record (arms that already existed) ----
ctb = json.load(open(COMMITTED_TB))["arms"]
ccur = json.load(open(COMMITTED_CUR))["arms"]
MAP_TB = {"Lexical": "lexical", "KG (functional)": "kg(weighted)", "KG (embedding)": "kg_embed",
          "Hybrid (embedding)": "hybrid_embed", "Hybrid (production)": "hybrid(prod)"}
MAP_CUR = {"Lexical": "lexical", "KG (functional)": "kg", "KG + enrichment": "kg+enrich",
           "KG (embedding)": "kg_embed"}
worst = 0.0
print("\n=== reproduction check (max |new - committed| per arm) ===")
for new, old in MAP_TB.items():
    d = max(abs(grid["testbench"][new][s][m] - ctb[old][s][m])
            for s in ("overall", "named", "functional") for m in ("pass@1", "pass@3", "mrr"))
    worst = max(worst, d); print(f"  testbench {new:20s} max diff {d:.4f}")
for new, old in MAP_CUR.items():
    d = max(abs(grid["curated"][new][m] - ccur[old][m])
            for m in ("pass@1", "pass@3", "mrr", "coverage@3", "set_precision@1"))
    worst = max(worst, d); print(f"  curated   {new:20s} max diff {d:.4f}")
print(f"  WORST: {worst:.4f}  ({'REPRODUCES' if worst < 1e-9 else 'DIFFERS — investigate before use'})")

OUT.write_text(json.dumps({"n_candidates": len(candidates), "n_testbench": len(tb),
                           "n_curated": len(cur), "grid": grid,
                           "reproduction_worst_abs_diff": worst}, indent=2, default=list))
print(f"\n[json] -> {OUT}")

print("\n=== TESTBENCH (231): pass@3 overall / named / functional ; MRR ===")
for name in ARMS:
    v = grid["testbench"][name]
    print(f"  {name:20s} {v['overall']['pass@3']:.2f}  {v['named']['pass@3']:.2f}  "
          f"{v['functional']['pass@3']:.2f}   p@1 {v['overall']['pass@1']:.2f}  mrr {v['overall']['mrr']:.2f}")
print("\n=== CURATED (18): p@1  p@3  coverage@3  set-P@1  MRR ===")
for name in ARMS:
    v = grid["curated"][name]
    print(f"  {name:20s} {v['pass@1']:.2f}  {v['pass@3']:.2f}  {v['coverage@3']:.2f}  "
          f"{v['set_precision@1']:.2f}  {v['mrr']:.2f}")

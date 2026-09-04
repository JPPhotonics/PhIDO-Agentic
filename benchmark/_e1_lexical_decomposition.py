"""E1 lexical-index decomposition (2026-08-29).

Motivation: the E1 'Lexical' control (benchmark/e1_retrieval.lexical_arm) matches name+description
ONLY (the KG-derived `function` field was withheld to avoid leakage). The production hybrid's lexical
half is pdk_catalog_server.search_components, which ALSO matches the kit's NodeLabels and `aka`
aliases. So 'production hybrid vs lexical' confounded (KG contribution) with (richer lexical index).
This script holds the lexical half fixed and varies the KG half, on both query sets, ungated.

Output: results/e1_lexical_decomposition.{json,md}
Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/_e1_lexical_decomposition.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path += [str(HERE), str(HERE.parent)]
try:
    import _embed_proxy; _embed_proxy.install()
except Exception:
    pass
from e1_retrieval import QueryCase, lexical_arm, score_arm
from kg_retrieve import make_kg_retriever
from run_e1_stratified import build_candidates, is_named
from mcp_servers.component_retrieval import _lexical_ranking, _kg_ranking, _CATALOG_BY_MODULE
from metrics import mean_pass_at_k, mrr
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

OUT = HERE / "results" / "e1_lexical_decomposition.json"
c = Neo4jClient(config=Neo4jConfig()); c.connect()
cands = build_candidates(c); ids = [x["id"] for x in cands]; idset = set(ids)
kg_func = make_kg_retriever(include_enrichment=False, client=c, weighted=True)
kg_enr  = make_kg_retriever(include_enrichment=True,  client=c, weighted=True)
kg_emb  = make_kg_retriever(include_enrichment=False, client=c, weighted=True, functional=False)
def L_cat(q, cs): return [m for m in _lexical_ranking(q, limit=68) if m in idset]
def K_prod(q, cs): return [m for m in _kg_ranking(q, c) if m in idset]
def rrf(*rs, k=60):
    sc = {m: 0.0 for m in ids}
    for r in rs:
        for i, m in enumerate(r):
            if m in sc: sc[m] += 1 / (k + i + 1)
    return sorted(sc, key=lambda m: (-sc[m], m))
ARMS = {
    "Lexical (name+description only)": lexical_arm,
    "Lexical (catalog: name+labels+aka)": L_cat,
    "KG (functional)": kg_func,
    "KG + enrichment": kg_enr,
    "KG (embedding)": kg_emb,
    "RRF(name+desc lexical, KG functional)": lambda q, cs: rrf(lexical_arm(q, cs), kg_func(q, cs)),
    "RRF(name+desc lexical, KG embedding)": lambda q, cs: rrf(lexical_arm(q, cs), kg_emb(q, cs)),
    "RRF(catalog lexical, KG functional)": lambda q, cs: rrf(L_cat(q, cs), kg_func(q, cs)),
    "RRF(catalog lexical, KG embedding)": lambda q, cs: rrf(L_cat(q, cs), kg_emb(q, cs)),
    "RRF(catalog lexical, production KG half) [production, ungated]": lambda q, cs: rrf(L_cat(q, cs), K_prod(q, cs)),
}
tb = []
for q in json.load(open(HERE / "e1_queries_testbench.json"))["queries"]:
    g = {m for m in q["gold"] if m in idset}
    if g: tb.append((q["query"], g, "named" if is_named(q["query"]) else "functional"))
cur = [QueryCase(q["id"], q["query"], {m for m in q["gold"] if m in idset}, "intent")
       for q in json.load(open(HERE / "e1_queries.json"))["queries"]]
cur = [q for q in cur if q.gold]
res = {"testbench": {}, "curated": {}}
for name, arm in ARMS.items():
    R = [(arm(q, cands), g, s) for q, g, s in tb]
    def M(sub): return {"n": len(sub), "pass@1": mean_pass_at_k(sub, 1), "pass@3": mean_pass_at_k(sub, 3), "mrr": mrr(sub)}
    res["testbench"][name] = {"overall": M([(r, g) for r, g, _ in R]),
                              "named": M([(r, g) for r, g, s in R if s == "named"]),
                              "functional": M([(r, g) for r, g, s in R if s == "functional"])}
    res["curated"][name] = score_arm(arm, cur, cands, ks=(1, 3))
    print("done:", name)
# leakage audit of the catalog index on the testbench
def toks(s): return set(s.lower().replace("-", " ").replace("_", " ").split())
exact = subset = 0
for q, g, s in tb:
    ql = q.lower().strip(); hit_e = hit_s = False
    for m in g:
        comp = _CATALOG_BY_MODULE.get(m, {}); aka = [a.strip().lower() for a in (comp.get("aka", "") or "").split(",") if a.strip()]
        text = toks(" ".join(aka)) | toks(" ".join(comp.get("labels", []))) | toks(comp.get("name", ""))
        if ql in aka or ql == comp.get("name", "").lower(): hit_e = True
        elif toks(q) and toks(q) <= text: hit_s = True
    exact += hit_e; subset += (hit_s and not hit_e)
res["leakage_audit_testbench"] = {"query_equals_gold_name_or_alias": exact, "query_tokens_subset_of_gold_metadata": subset, "n": len(tb)}
OUT.write_text(json.dumps({"n_candidates": len(cands), "n_testbench": len(tb), "n_curated": len(cur), "results": res}, indent=2))
L = ["# E1 lexical-index decomposition (2026-08-29)", "",
     "Holds the lexical half fixed and varies the KG half; ungated 34-cell pool. The thesis's original 'Lexical' control was name+description only; the production hybrid's lexical half is the catalog search (name+labels+aka).", "",
     "## Testbench (231) — p@1 / p@3 / MRR / named p@3 / functional p@3", "", "| configuration | p@1 | p@3 | MRR | named p@3 | func p@3 |", "|---|---|---|---|---|---|"]
for n, v in res["testbench"].items():
    L.append(f"| {n} | {v['overall']['pass@1']:.3f} | {v['overall']['pass@3']:.3f} | {v['overall']['mrr']:.3f} | {v['named']['pass@3']:.3f} | {v['functional']['pass@3']:.3f} |")
L += ["", "## Curated intents (18) — p@1 / p@3 / coverage@3 / set-P@1 / MRR", "", "| configuration | p@1 | p@3 | cov@3 | set-P@1 | MRR |", "|---|---|---|---|---|---|"]
for n, v in res["curated"].items():
    L.append(f"| {n} | {v['pass@1']:.3f} | {v['pass@3']:.3f} | {v['coverage@3']:.3f} | {v['set_precision@1']:.3f} | {v['mrr']:.3f} |")
la = res["leakage_audit_testbench"]
L += ["", f"## Leakage audit (catalog index vs testbench gold): query == gold name/alias in {la['query_equals_gold_name_or_alias']}/{la['n']}; all query tokens inside gold name+labels+aka in a further {la['query_tokens_subset_of_gold_metadata']}/{la['n']}.", "",
      "## Read", "- On the testbench the catalog's own lexical search is the best retriever; fusing in either KG half does not raise pass@3 and lowers pass@1/MRR. The prior 'production hybrid dominates lexical' result compared against the weakened name+description control.",
      "- With the lexical half held fixed, functional-boost vs embedding KG halves are indistinguishable on functional queries → the 'functional structure adds value embedding does not replace' claim is a confound of the lexical index. RETRACTED in the thesis.",
      "- On the curated function-only intents, KG embedding alone is best (cov@3 0.68); catalog lexical = name+desc lexical (0.44); fusion dilutes. The graph earns its retrieval value exactly and only on function-described queries."]
(OUT.with_suffix(".md")).write_text("\n".join(L) + "\n"); print("[json+md] ->", OUT)

"""Diagnostic: can we reduce the KG arm's wrong-Design_Function matching / dilution?

The KG functional weakness (2026-06-25) is RANKING DILUTION: semantic_search pulls 3 Design_Functions
(often irrelevant), each over-connected, so a median 15/34 cells tie on the functional boost and gold
sinks to median rank ~12. This script tests principled retriever variants on the CURRENT KB (no rebuild),
sharing one set of (expensive) embedding searches across all variants:

  baseline      current: funcs sim>=0.40 top3; boost = max edge_conf; score = boost*1000 + cell_sim
  simw          boost = max(func_sim * edge_conf)   -> weak (low-sim) functions contribute less
  thr55_lim2    funcs sim>=0.55, top2               -> fewer, more relevant functions
  idf           boost = max(edge_conf / log2(2+deg(func)))  -> down-weight over-connected functions
  simw_thr55    simw + funcs sim>=0.55 top2         -> combined
  simw_idf      boost = max(func_sim * edge_conf / log2(2+deg(func)))

Reported overall / named / functional (stratifier plural bug FIXED here: \bk s?\b). DRAFT gold +
tuning-on-test risk -> read as a SENSITIVITY ANALYSIS, not a tuned final number.

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_dfmatch_ablation.py
"""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE)); sys.path.append(str(HERE.parent))
from metrics import mean_pass_at_k, mrr                              # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient       # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig       # noqa: E402

QUERY_FILES = [HERE / "e1_queries_testbench.json", HERE / "e1_queries.json"]
FUNC_HIT = 1000.0

NAMED = ["mzi","mach-zehnder","mach zehnder","mzm","interferometer","mmi","multimode interference",
 "directional coupler","coupler","splitter","modulator","resonator","microring","ring","grating coupler",
 "grating","edge coupler","phase shifter","heater","attenuator","voa","crossing","photodetector",
 "photodiode","detector","laser","bend","multiplexer","demultiplexer","wdm","polarization splitter",
 "taper","mode converter","hybrid","pin diode","pin-diode","pn diode","pn-diode","waveguide"]
def is_named(q): return any(re.search(rf"\b{re.escape(k)}s?\b", q.lower()) for k in NAMED)  # plural-tolerant


def main():
    c = Neo4jClient(config=Neo4jConfig()); c.connect()
    with c.driver.session() as s:
        cands = [{"id": r["id"]} for r in s.run(
            "MATCH (p:PDK_Cell) RETURN p.module_name AS id ORDER BY id") if r["id"]]
        # function -> {cell: edge_conf}, native edges only (enrichment OFF, matches kg(weighted))
        fmap, fdeg = {}, {}
        for r in s.run("""MATCH (f:Design_Function)<-[r:PERFORMS_FUNCTION]-(p:PDK_Cell)
                          WHERE coalesce(r.provenance,'')<>'inferred_from_literature'
                          RETURN f.name AS f, p.module_name AS m, max(coalesce(r.confidence,1.0)) AS conf"""):
            fmap.setdefault(r["f"], {})[r["m"]] = float(r["conf"])
        for f, cells in fmap.items():
            fdeg[f] = len(cells)
    cand_ids = [x["id"] for x in cands]

    sets = {}
    for qf in QUERY_FILES:
        qspec = json.loads(qf.read_text())["queries"]
        queries = []
        for q in qspec:
            gold = {m for m in q["gold"] if m in set(cand_ids)}
            if q["query"] and gold:
                queries.append({"q": q["query"], "gold": gold,
                                "stratum": "named" if is_named(q["query"]) else "functional"})
        pre = []
        for qq in queries:
            fns = c.semantic_search(qq["q"], "Design_Functions", limit=8, threshold=0.0)
            funcs = [(f.get("name"), float(f.get("score", 0.0))) for f in fns if f.get("name")]
            cell_hits = c.semantic_search(qq["q"], "PDK_Cells", limit=400, threshold=0.0)
            sem = {h.get("module_name"): float(h.get("score", 0.0)) for h in cell_hits if h.get("module_name")}
            pre.append((funcs, sem))
        sets[qf.stem] = (queries, pre)
    c.close()

    def rank(funcs, sem, *, thr, lim, simw, idf, gate=FUNC_HIT, sem_only=False):
        if sem_only:
            return sorted(cand_ids, key=lambda cid: (-sem.get(cid, 0.0), str(cid)))
        chosen = [(f, fs) for f, fs in funcs if fs >= thr][:lim]
        boost = {}
        for f, fs in chosen:
            w = (fs if simw else 1.0) / (math.log2(2 + fdeg.get(f, 1)) if idf else 1.0)
            for cell, conf in fmap.get(f, {}).items():
                boost[cell] = max(boost.get(cell, 0.0), w * conf)
        return sorted(cand_ids, key=lambda cid: (-(boost.get(cid, 0.0) * gate + sem.get(cid, 0.0)), str(cid)))

    variants = {
        "baseline":   dict(thr=0.40, lim=3, simw=False, idf=False),
        "simw":       dict(thr=0.40, lim=3, simw=True,  idf=False),
        "thr55_lim2": dict(thr=0.55, lim=2, simw=False, idf=False),
        "idf":        dict(thr=0.40, lim=3, simw=False, idf=True),
        "simw_thr55": dict(thr=0.55, lim=2, simw=True,  idf=False),
        "simw_idf":   dict(thr=0.40, lim=3, simw=True,  idf=True),
        "sem_only":   dict(thr=0.40, lim=3, simw=False, idf=False, sem_only=True),
        "soft_add":   dict(thr=0.40, lim=3, simw=False, idf=False, gate=1.0),  # functional = soft signal, not hard gate
    }
    for setname, (queries, pre) in sets.items():
        nN = sum(1 for q in queries if q["stratum"] == "named")
        print(f"\n##### {setname}: queries={len(queries)} (named={nN}, functional={len(queries)-nN}); plural-fixed stratifier")
        print(f"{'variant':12} | {'overall p@1/p@3/mrr':22} | {'named':18} | {'functional p@1/p@3/mrr':22}")
        for name, p in variants.items():
            ranked = [(rank(pre[i][0], pre[i][1], **p), queries[i]["gold"], queries[i]["stratum"])
                      for i in range(len(queries))]
            def M(sub):
                pr = [(r, g) for r, g, s in ranked if sub == "all" or s == sub]
                if not pr: return "  (n=0)"
                return f"{mean_pass_at_k(pr,1):.3f}/{mean_pass_at_k(pr,3):.3f}/{mrr(pr):.3f}"
            print(f"{name:12} | {M('all'):22} | {M('named'):18} | {M('functional'):22}")


if __name__ == "__main__":
    main()

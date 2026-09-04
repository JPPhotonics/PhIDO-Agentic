"""E1 stratified analysis: does the KG advantage concentrate on FUNCTIONAL (un-named) queries?

Re-scores the expanded testbench query set, splitting each query into:
  * "named"      — the query contains an explicit component-type noun (MZI, MMI, coupler,
                   splitter, modulator, ring resonator, grating coupler, ...); lexical token
                   overlap can solve these directly.
  * "functional" — no component-type noun; the need is described by behaviour only. This is
                   where the KG's PERFORMS_FUNCTION structure (and the paper enrichment) should help.

One retrieval pass per arm; metrics computed overall and per stratum. The keyword set is listed
in the report for auditability (the named/functional split is a transparent, deterministic rule).

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_stratified.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))

from e1_retrieval import lexical_arm                       # noqa: E402
from metrics import mean_pass_at_k, mrr                    # noqa: E402
from report import Reporter                                # noqa: E402
# KG / Neo4j imports are deferred into main() (2026-08-28) so that `is_named` and
# NAMED_KEYWORDS can be imported by other drivers (run_e1_llm_json.py) without loading
# the embedding model. The stratification rule stays defined in exactly one place.

QUERIES_FILE = HERE / "e1_queries_testbench.json"
OUT = HERE / "results" / "e1_stratified.json"

# Component-type nouns: presence => the query NAMES a component (lexical-solvable).
NAMED_KEYWORDS = [
    "mzi", "mach-zehnder", "mach zehnder", "mzm", "interferometer",
    "mmi", "multimode interference", "directional coupler", "coupler",
    "splitter", "modulator", "resonator", "microring", "ring",
    "grating coupler", "grating", "edge coupler", "phase shifter", "heater",
    "attenuator", "voa", "crossing", "photodetector", "photodiode", "detector",
    "laser", "bend", "multiplexer", "demultiplexer", "wdm", "polarization splitter",
    "taper", "mode converter", "hybrid", "pin diode", "pin-diode", "pn diode", "pn-diode",
    "waveguide",
]


def is_named(query: str) -> bool:
    q = query.lower()
    # plural-tolerant (`s?`): the old `\bk\b` missed "MMIs"/"MZIs"/"bends" → named queries
    # leaked into the functional stratum. Fixed 2026-06-25.
    return any(re.search(rf"\b{re.escape(k)}s?\b", q) for k in NAMED_KEYWORDS)


def build_candidates(client) -> list[dict]:
    cy = ("MATCH (p:PDK_Cell) RETURN p.module_name AS id, "
          "coalesce(p.display_name,p.module_name) AS name, coalesce(p.description,'') AS description "
          "ORDER BY id")
    with client.driver.session() as s:
        return [{"id": r["id"], "name": r["name"], "description": r["description"]}
                for r in s.run(cy) if r["id"]]


def metrics_for(pairs: list) -> dict:
    return {"n": len(pairs), "pass@1": mean_pass_at_k(pairs, 1),
            "pass@3": mean_pass_at_k(pairs, 3), "mrr": mrr(pairs)}


def main() -> None:
    from kg_retrieve import make_kg_retriever                  # noqa: E402
    from mcp_servers.component_retrieval import retrieve_candidates  # noqa: E402  (production path)
    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient   # noqa: E402
    from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig   # noqa: E402
    client = Neo4jClient(config=Neo4jConfig())
    client.connect()
    candidates = build_candidates(client)
    qspec = json.loads(QUERIES_FILE.read_text())["queries"]
    cand_ids = {c["id"] for c in candidates}

    queries = []
    for q in qspec:
        gold = {m for m in q["gold"] if m in cand_ids}
        if gold:
            queries.append({"query": q["query"], "gold": gold, "stratum": "named" if is_named(q["query"]) else "functional"})

    import json as _json
    def _prod_hybrid(query, candidates):
        """The SHIPPED Phase-4 retriever (lexical + confidence-weighted KG, RRF-fused)."""
        r = _json.loads(retrieve_candidates(query, top_k=len(candidates), client=client))
        return [c["module_name"] for c in r.get("results", [])]

    # 2026-06-25: embedding-only KG arm (functional gate OFF) + a hybrid built from it.
    kg_embed = make_kg_retriever(include_enrichment=False, client=client, weighted=True, functional=False)

    def _rrf(query, candidates, arms_to_fuse, k=60):
        """Reciprocal-Rank-Fusion of several arms' rankings (same fusion the prod hybrid uses)."""
        scores = {c["id"]: 0.0 for c in candidates}
        for arm in arms_to_fuse:
            for rank_i, cid in enumerate(arm(query, candidates)):
                if cid in scores:
                    scores[cid] += 1.0 / (k + rank_i + 1)
        return sorted(scores, key=lambda cid: (-scores[cid], str(cid)))

    def _hybrid_embed(query, candidates):
        """Hybrid with the FIX: lexical + embedding-only KG, RRF-fused (KG half = kg_embed, no func gate)."""
        return _rrf(query, candidates, [lexical_arm, kg_embed])

    arms = {
        "lexical": lexical_arm,
        "kg(weighted)": make_kg_retriever(include_enrichment=False, client=client, weighted=True),
        "kg_embed": kg_embed,
        "hybrid(prod)": _prod_hybrid,
        "hybrid_embed": _hybrid_embed,
    }

    n_named = sum(1 for q in queries if q["stratum"] == "named")
    n_func = len(queries) - n_named
    print(f"queries: {len(queries)} (named={n_named}, functional={n_func})")

    out = {}
    for name, arm in arms.items():
        ranked = [(arm(q["query"], candidates), q["gold"], q["stratum"]) for q in queries]
        out[name] = {
            "overall": metrics_for([(r, g) for r, g, _ in ranked]),
            "named": metrics_for([(r, g) for r, g, s in ranked if s == "named"]),
            "functional": metrics_for([(r, g) for r, g, s in ranked if s == "functional"]),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_candidates": len(candidates), "arms": out}, indent=2))
    print(f"[json] -> {OUT}")

    rep = Reporter(OUT.with_suffix(".md"), "E1 stratified: named vs functional queries",
                   meta={"queries": len(queries), f"named": n_named, "functional": n_func,
                         "candidates": len(candidates), "scope": "34-cell DemoPDK; DRAFT gold"})
    for stratum in ("overall", "named", "functional"):
        rep.h(f"{stratum}  (pass@1 / pass@3 / mrr)")
        rep.table(["arm", "n", "pass@1", "pass@3", "mrr"],
                  [[name, out[name][stratum]["n"], f"{out[name][stratum]['pass@1']:.3f}",
                    f"{out[name][stratum]['pass@3']:.3f}", f"{out[name][stratum]['mrr']:.3f}"]
                   for name in arms])
    rep.h("Read")
    rep.line("- Hypothesis: KG ~= lexical on **named** queries; KG > lexical on **functional** queries, "
             "with enrichment's lift concentrated in functional.")
    rep.line("- 'named' = query contains a component-type noun (keyword list in run_e1_stratified.py). "
             "Transparent deterministic split.")
    rep.line("- DRAFT gold (TYPE_TO_MODULES + LLM phrase->type typing) — audit before final use.")
    rep.save()


if __name__ == "__main__":
    main()

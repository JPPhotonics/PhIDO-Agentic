"""Hybrid PDK component retrieval for Phase-4 selection.

Fuses two rankings over the PDK catalog and returns the top candidates as a JSON string
(drop-in for ``pdk_catalog_server.search_components``):

* **lexical** — token-overlap (``search_components``); strong when the query NAMES a component.
* **confidence-weighted KG** — PDK_Cell embedding similarity + PERFORMS_FUNCTION reachability,
  where each functional hit is scaled by the MAX edge confidence (native 0.8-1.0 >> literature-
  inferred 0.6), so low-confidence enrichment edges can't outrank native structure. Strong when
  the query describes a FUNCTION without naming the part. (Validated in the E1 benchmark; paper
  enrichment is retrieval-neutral, and this weighting keeps it from harming.)

The two are combined with **Reciprocal-Rank Fusion** (parameter-free). If Neo4j / the KG is
unavailable, the retriever transparently **falls back to lexical-only** so the online pipeline
never breaks. Pure graph+embedding (no LLM), so candidate generation stays deterministic; the
orchestrator's downstream LLM still makes the final selection.

Backend via ``RETRIEVAL_BACKEND`` env: ``hybrid`` (default) | ``lexical`` | ``kg``.
"""
from __future__ import annotations

import json
import os

from mcp_servers.pdk_catalog_server import CATALOG, search_components
from mcp_servers.pdk_whitelist import selectable_modules, simulatable_modules

_FUNC_WEIGHT = 1000.0          # a functional hit dominates; embedding similarity (0-1) breaks ties
_RRF_K = 60                    # standard RRF damping constant
_CATALOG_BY_MODULE = {c["module_name"]: c for c in CATALOG}


def _lexical_ranking(query: str, limit: int) -> list[str]:
    """Ranked module_names from the lexical catalog search (best-first)."""
    res = json.loads(search_components(query, limit=limit))
    if isinstance(res, dict):          # {"message": "No matching components found."}
        return []
    return [r["module_name"] for r in res]


def _kg_ranking(query: str, client) -> list[str]:
    """Ranked module_names from confidence-weighted KG retrieval (raises if KG is unreachable)."""
    # functional reachability, weighted by max PERFORMS_FUNCTION edge confidence
    func: dict[str, float] = {}
    fns = client.semantic_search(query, "Design_Functions", limit=3, threshold=0.4)
    names = [f.get("name") for f in fns if f.get("name")]
    if names:
        cy = ("MATCH (f:Design_Function)<-[r:PERFORMS_FUNCTION]-(p:PDK_Cell) "
              "WHERE f.name IN $names "
              "RETURN p.module_name AS m, max(coalesce(r.confidence, 1.0)) AS conf")
        with client.driver.session() as s:
            func = {rec["m"]: float(rec["conf"]) for rec in s.run(cy, names=names) if rec["m"]}
    # embedding similarity over PDK cells
    hits = client.semantic_search(query, "PDK_Cells", limit=400, threshold=0.0)
    sem = {h.get("module_name"): float(h.get("score", 0.0)) for h in hits if h.get("module_name")}

    mods = set(func) | set(sem)
    def score(m: str) -> float:
        return _FUNC_WEIGHT * func.get(m, 0.0) + sem.get(m, 0.0)
    return sorted((m for m in mods if score(m) > 0), key=lambda m: (-score(m), m))


def _rrf(rankings: list[list[str]]) -> list[str]:
    """Reciprocal-Rank Fusion: score(m) = sum_r 1/(K + rank_r(m))."""
    agg: dict[str, float] = {}
    for ranking in rankings:
        for i, m in enumerate(ranking):
            agg[m] = agg.get(m, 0.0) + 1.0 / (_RRF_K + i + 1)
    return sorted(agg, key=lambda m: (-agg[m], m))


def _get_client():
    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
    from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
    c = Neo4jClient(config=Neo4jConfig())
    c.connect()
    return c


def retrieve_candidates(query: str, top_k: int = 8, backend: str | None = None, client=None) -> str:
    """Drop-in for ``search_components(query)``: returns a JSON string of the top fused candidates
    (each with full metadata + a ``via`` provenance field). Never raises — degrades to lexical."""
    backend = (backend or os.getenv("RETRIEVAL_BACKEND", "hybrid")).lower()

    lex = _lexical_ranking(query, limit=max(top_k * 2, 20))
    kg: list[str] = []
    used = backend
    if backend in ("hybrid", "kg"):
        try:
            kg = _kg_ranking(query, client or _get_client())
        except Exception as e:  # Neo4j down / index missing / etc. -> degrade to lexical
            print(f"  [component_retrieval] KG unavailable, lexical fallback: {e}")
            used = "lexical(fallback)"

    if backend == "lexical" or not kg:
        fused = lex
    elif backend == "kg":
        fused = kg
    else:                                   # hybrid
        fused = _rrf([lex, kg])
        used = "hybrid"

    # Gate the candidate pool to what the pipeline can actually build & simulate. This is the
    # enforcement the rigid baseline gets for free (it indexes a curated list): drop anything
    # not in the catalog and the `_`-prefixed internal building blocks (never top-level
    # selectable), then — unless RETRIEVAL_ENFORCE_SIMULATABLE=0 — keep only simulatable
    # modules. Fallback: if simulatable-filtering empties the list, fall back to the selectable
    # set so a query never returns nothing (the selection gate still flags the non-simulatable
    # pick downstream).
    selectable, simset = selectable_modules(), simulatable_modules()
    fused = [m for m in fused if m in selectable]
    enforce = os.getenv("RETRIEVAL_ENFORCE_SIMULATABLE", "1").lower() not in ("0", "false", "no")
    if enforce:
        fused = [m for m in fused if m in simset] or fused

    lexset, kgset = set(lex), set(kg)
    results = []
    for m in fused[:top_k]:
        meta = dict(_CATALOG_BY_MODULE.get(m, {"module_name": m}))
        meta["via"] = ("both" if m in lexset and m in kgset
                       else "lexical" if m in lexset else "kg")
        meta["simulatable"] = m in simset
        results.append(meta)

    if not results:
        return json.dumps({"message": "No matching components found.", "query": query})
    return json.dumps({"backend": used, "results": results}, indent=2)

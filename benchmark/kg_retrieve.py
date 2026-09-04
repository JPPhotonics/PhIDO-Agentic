"""KG-grounded retriever for the E1 retrieval benchmark (plan decisions #1 and #4).

Ranks PDK candidate ``module_name``s for a query using the knowledge graph:
  * functional reachability via ``PDK_Cell-[:PERFORMS_FUNCTION]->Design_Function``
    (the edge type the paper enrichment populates), and
  * ``PDK_Cell`` embedding similarity as the tie-breaker / fallback.

``include_enrichment`` is the decision-#4 ablation knob: when False, edges stamped
with ``provenance='inferred_from_literature'`` (the paper-propagated enrichment) are
excluded from the functional traversal, so the arm sees only ontology+PDK-native
structure. When True, the paper-derived edges participate.

Pure graph + embedding (no LLM call), so the arm is deterministic relative to the
deterministic embedding model — keeps the arm comparison stable (no seed needed).

Arm signature (matches benchmark/e1_retrieval.py): ``(query, candidates) -> [module_name, ...]``.
"""
from __future__ import annotations

from typing import Callable

# A functional hit dominates the score; embedding similarity (in [0,1]) breaks ties.
_FUNC_HIT = 1000.0


def make_kg_retriever(include_enrichment: bool, client=None, weighted: bool = True,
                      functional: bool = True
                      ) -> Callable[[str, list[dict]], list[str]]:
    """Build a KG retrieval arm.

    Args:
        include_enrichment: include paper-propagated (``inferred_from_literature``) edges.
        client: a connected ``Neo4jClient``; created+connected if None. Pass a shared
            client across arms to avoid reloading the embedding model per arm.
        weighted: if True, a cell's functional boost scales with the MAX confidence of its
            matching PERFORMS_FUNCTION edges (native 0.8-1.0 >> literature-inferred 0.6), so
            low-confidence enrichment edges can't outrank native structure. If False, the
            functional hit is binary (original behaviour).
        functional: if True (default), rank by ``PERFORMS_FUNCTION`` functional reachability
            (boost) with PDK_Cell embedding as tie-break. If False, IGNORE the functional graph
            entirely and rank by PDK_Cell embedding similarity ONLY. The 2026-06-25 ablation
            (benchmark/run_e1_dfmatch_ablation.py + E1_RETRIEVAL_FINDINGS.md) found the functional
            gate is net-HARMFUL — over-broad Design_Function matching dilutes gold into a large
            boost-tie — so ``functional=False`` is the empirically better retriever on this KB.
            (With functional=False the ``include_enrichment`` knob is moot: enrichment only adds
            PERFORMS_FUNCTION edges, which are unused.)
    """
    if client is None:
        from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
        from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
        client = Neo4jClient(config=Neo4jConfig())
        client.connect()

    # Enrichment toggle: drop paper-propagated edges for the no-enrichment arm.
    prov_clause = "" if include_enrichment else \
        "AND coalesce(r.provenance, '') <> 'inferred_from_literature'"
    func_cypher = f"""
        MATCH (f:Design_Function)<-[r:PERFORMS_FUNCTION]-(p:PDK_Cell)
        WHERE f.name IN $names {prov_clause}
        RETURN p.module_name AS m, max(coalesce(r.confidence, 1.0)) AS conf
    """

    def _functional_scores(query: str) -> dict:
        """{module_name: max edge confidence} for cells reachable from query-matched functions."""
        fns = client.semantic_search(query, "Design_Functions", limit=3, threshold=0.4)
        names = [f.get("name") for f in fns if f.get("name")]
        if not names:
            return {}
        with client.driver.session() as s:
            return {rec["m"]: float(rec["conf"]) for rec in s.run(func_cypher, names=names) if rec["m"]}

    def _semantic_scores(query: str) -> dict:
        """module_name -> PDK_Cell embedding similarity for the query."""
        hits = client.semantic_search(query, "PDK_Cells", limit=400, threshold=0.0)
        return {h.get("module_name"): float(h.get("score", 0.0))
                for h in hits if h.get("module_name")}

    def _arm(query: str, candidates: list[dict]) -> list[str]:
        ids = [c["id"] for c in candidates]
        func = _functional_scores(query) if functional else {}
        sem = _semantic_scores(query)
        def score(cid: str) -> float:
            conf = func.get(cid, 0.0)
            boost = (conf if weighted else (1.0 if conf > 0 else 0.0)) * _FUNC_HIT
            return boost + sem.get(cid, 0.0)
        # Descending score; deterministic id tie-break (mirrors lexical_arm).
        return sorted(ids, key=lambda cid: (-score(cid), str(cid)))

    return _arm

"""Dump structural KB stats to JSON — used by the variance campaign to check whether the
KB *structure* (not just the E1 metric) is reproducible across rebuilds.

Usage:  CUDA_VISIBLE_DEVICES="" python benchmark/kb_stats.py <out.json>
"""
import json, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

ONE = lambda s, q: s.run(q).single()[0]

def main(out):
    c = Neo4jClient(config=Neo4jConfig()); c.connect()
    st = {}
    with c.driver.session() as s:
        st["nodes"] = ONE(s, "MATCH (n) RETURN count(n)")
        st["edges"] = ONE(s, "MATCH ()-[r]->() RETURN count(r)")
        st["design_function_nodes"] = ONE(s, "MATCH (f:Design_Function) RETURN count(f)")
        st["pdk_cells"] = ONE(s, "MATCH (p:PDK_Cell) RETURN count(p)")
        st["edge_types"] = {r["t"]: r["c"] for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")}
        st["pdk_performs_function"] = {
            "native": ONE(s, "MATCH (:PDK_Cell)-[r:PERFORMS_FUNCTION]->(:Design_Function) "
                             "WHERE coalesce(r.provenance,'')<>'inferred_from_literature' RETURN count(r)"),
            "inferred": ONE(s, "MATCH (:PDK_Cell)-[r:PERFORMS_FUNCTION]->(:Design_Function) "
                               "WHERE r.provenance='inferred_from_literature' RETURN count(r)"),
        }
        st["related_to_total"] = ONE(s, "MATCH ()-[r:RELATED_TO]->() RETURN count(r)")
        st["review_items"] = {r["reason"]: r["c"] for r in s.run(
            "MATCH (r:ReviewItem) RETURN coalesce(r.reason,'?') AS reason, count(*) AS c")}
        st["cells_with_pf"] = ONE(s, "MATCH (p:PDK_Cell) WHERE (p)-[:PERFORMS_FUNCTION]->(:Design_Function) "
                                     "RETURN count(DISTINCT p)")
    c.close()
    Path(out).write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kb_stats.json")

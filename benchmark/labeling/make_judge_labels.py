"""Build judge_labels.json from DIA's semantic-verifier verdicts for the bundle triples.

DIA's SemanticVerificationResult (is_related + reasoning + confidence) IS the LLM-judge the
benchmark validates. For each bundle edge we recover the verdict (approved vs "Rejected by
semantic verifier", from the edge description) and its confidence. Outputs:
  - judge_labels.json     {triple_id: "supported"|"not_supported"}  -> ingest_labels.py --judge
  - judge_confidence.json {triple_id: confidence}                   -> calibration analysis

NB: bundle edges are DIA-APPROVED (they exist in the KG), so expert<->judge agreement here measures
DIA PRECISION (do experts confirm what DIA accepted). Measuring recall would require also annotating
DIA-REJECTED relationships (available in output/full_llm_trace.json, 514 DIA records) — a future arm.

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/labeling/make_judge_labels.py
"""
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent.parent))
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient  # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig  # noqa: E402

ids = []
for f in sorted(glob.glob(str(HERE / "bundles" / "bundle_meta_*.json"))):
    ids += list(json.load(open(f))["items"].keys())
ids = list(dict.fromkeys(ids))

client = Neo4jClient(config=Neo4jConfig()); client.connect()
with client.driver.session() as s:
    rows = list(s.run(
        "MATCH (h)-[r]->(t) WHERE elementId(r) IN $ids "
        "RETURN elementId(r) AS id, type(r) AS rel, coalesce(h.name,h.title) AS head, "
        "coalesce(t.name,t.title) AS tail, r.description AS d, r.confidence AS conf, r.provenance AS prov",
        ids=ids))

verdicts, conf = {}, {}
for r in rows:
    d = r["d"] or ""
    verdicts[r["id"]] = "not_supported" if d.startswith("Rejected by semantic verifier") else "supported"
    if r["conf"] is not None:
        conf[r["id"]] = r["conf"]

(HERE / "judge_labels.json").write_text(json.dumps(verdicts, indent=2))
(HERE / "judge_confidence.json").write_text(json.dumps(conf, indent=2))

print(f"bundle edges: {len(rows)}")
print("verdict distribution (DIA is_related):", dict(Counter(verdicts.values())))
cs = list(conf.values())
if cs:
    print(f"DIA confidence: n={len(cs)} min={min(cs):.2f} median={statistics.median(cs):.2f} max={max(cs):.2f}")
print(f"[ok] -> judge_labels.json, judge_confidence.json")

print("\n--- DIA reasoning examples (one per relation type) for the Poon review ---")
seen = set()
for r in sorted(rows, key=lambda x: -(x["conf"] or 0)):
    d = r["d"] or ""
    if r["rel"] not in seen and d.startswith("Global Inference"):
        seen.add(r["rel"])
        print(f"\n[{r['rel']}]  ({r['head']}) -> ({r['tail']})   DIA conf={r['conf']:.2f}")
        print(f"   {d[:300]}")

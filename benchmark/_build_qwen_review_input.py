"""Adapt the Qwen n=24 results (agentic shards w0-w3 + rigid) into the arm->pid->[rec] shape that
build_review_html.py consumes, so Qwen gets the SAME blind 3-axis human review as gpt-5.4.

Qwen recs store {nodes: {id:class}, edges: [frozenset-repr strings], dot_string, status, ...}.
The review tool wants {pred_node_map, pred_edges: [[a,b]], dot_string, status, level_paper, _edgeF1}.
Failures are kept (empty node_map -> the UI shows "No circuit produced"), status 'ok'->'scored'.
"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "qwen/qwen3.6-27b"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")
import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

GOLD = {g["id"]: g for g in json.load(open(ROOT / "benchmark/b3_gold_v2.json"))["gold"]}


def parse_edges(edge_list):
    """frozenset-repr strings (or lists) -> [[a, b], ...] endpoint-pair lists."""
    out = []
    for e in edge_list:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2:
            out.append(list(eps))
    return out


def edgef1(node_map, edges, pid):
    if not node_map:
        return None
    try:
        s = SC._score_topo(Topology(nodes=dict(node_map),
                                    edges=[frozenset(e) for e in edges], external={}), pid)
        return {"edgeF1": s["edgeF1"], "compF1": s["compF1"], "ged": s.get("ged")}
    except Exception:  # noqa: BLE001
        return None


def to_rec(rec, pid):
    node_map = dict(rec.get("nodes") or {})
    edges = parse_edges(rec.get("edges") or [])
    status = rec.get("status", "")
    m = edgef1(node_map, edges, pid) if status == "ok" else None
    return {
        "pred_node_map": node_map,
        "pred_edges": edges,
        "dot_string": rec.get("dot_string", ""),
        # 'ok' -> 'scored' so the UI doesn't flag every Qwen run as bad; failures keep their status
        "status": "scored" if status == "ok" else (status or "unknown"),
        "error": rec.get("error", ""),
        "level": GOLD.get(pid, {}).get("level"),
        "level_paper": GOLD.get(pid, {}).get("level"),
        "compF1": (m or {}).get("compF1"),
        "edgeF1": (m or {}).get("edgeF1"),
        "ged": (m or {}).get("ged"),
    }


out = defaultdict(lambda: defaultdict(list))

# agentic arms (merge shards)
for f in glob.glob(str(ROOT / "benchmark/results/qwen_trace_ablation_w*.json")):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items():
            for rec in reps:
                out[arm][pid].append(to_rec(rec, pid))

# rigid
rd = json.load(open(ROOT / "benchmark/results/qwen_rigid_baseline.json"))["rigid_baseline"]
for pid, reps in rd.items():
    for rec in reps:
        out["rigid_baseline"][pid].append(to_rec(rec, pid))

dst = ROOT / "benchmark/results/_qwen_review_input.json"
dst.write_text(json.dumps({a: dict(b) for a, b in out.items()}, indent=1))
n = sum(len(r) for b in out.values() for r in b.values())
print(f"wrote {dst.name}: {len(out)} arms, {n} runs")
for a in out:
    print(f"  {a:16} {sum(len(r) for r in out[a].values())} runs")

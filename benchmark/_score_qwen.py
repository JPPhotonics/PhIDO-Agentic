"""Score the Qwen3.6-27B ablation (agentic shards + rigid) vs the REVISED b3_gold_v2, and compare
per-arm edge-F1 to the gpt-5.4 run. Pure scoring, no API. Reconstructs Topology from stored
nodes + edge-strings (driver serialized frozensets via default=str)."""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "qwen/qwen3.6-27b"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")     # the REVISED gold
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")

import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

PROMPTS = ["L3_01", "L3_02", "L3_04", "L3_07", "L3_09", "L3_10", "L3_12",
           "L4_08", "L4_09", "L4_10", "L4_11", "L4_12"]
ARMS = ["base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]


def parse_edges(edge_list):
    out = []
    for e in edge_list:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2:
            out.append(frozenset(eps))
    return out


def score(rec, pid):
    """-> (edgeF1, compF1) or None if not scoreable."""
    if rec.get("status") != "ok" or not rec.get("nodes"):
        return None
    pred = Topology(nodes=dict(rec["nodes"]), edges=parse_edges(rec.get("edges", [])), external={})
    try:
        s = SC._score_topo(pred, pid)
        return s["edgeF1"], s["compF1"]
    except Exception:  # noqa: BLE001
        return None


def arm_mean(pp):
    """per-prompt mean then over prompts (edgeF1); also n prompts scored."""
    pm = [st.mean(v) for p, v in pp.items() if v]
    return (st.mean(pm) if pm else float("nan")), len(pm)


# ---- Qwen agentic (merge shards) ----
qa_edge = {a: defaultdict(list) for a in ARMS}
qa_comp = {a: defaultdict(list) for a in ARMS}
for f in glob.glob(str(ROOT / "benchmark" / "results" / "qwen_trace_ablation_w*.json")):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items():
            for rec in reps:
                s = score(rec, pid)
                if s and s[0] is not None:
                    qa_edge[arm][pid].append(s[0]); qa_comp[arm][pid].append(s[1])

# ---- Qwen rigid ----
qr_edge = defaultdict(list); qr_comp = defaultdict(list)
rd = json.load(open(ROOT / "benchmark" / "results" / "qwen_rigid_baseline.json"))["rigid_baseline"]
for pid, reps in rd.items():
    for rec in reps:
        s = score(rec, pid)
        if s and s[0] is not None:
            qr_edge[pid].append(s[0]); qr_comp[pid].append(s[1])

# ---- gpt-5.4 (already scored in place vs the revised gold), same 12 prompts ----
g54 = json.load(open(ROOT / "benchmark" / "results" / "_ablation_correctness_gpt54_v2.json"))


def g54_arm(arm):
    pp = defaultdict(list)
    for pid in PROMPTS:
        for r in g54.get(arm, {}).get(pid, []):
            if isinstance(r.get("edgeF1"), (int, float)):
                pp[pid].append(r["edgeF1"])
    return arm_mean(pp)


print("\n================  QWEN3.6-27B ablation vs revised b3_gold_v2  ================")
print(f"{'arm':16} {'edgeF1':>8} {'compF1':>8} {'#prompts':>9}   | gpt-5.4 edgeF1")
re_, rn = arm_mean(qr_edge); rc, _ = arm_mean(qr_comp)
g54_rigid, _ = g54_arm("rigid_baseline")
print(f"{'rigid_baseline':16} {re_:8.2f} {rc:8.2f} {rn:9}   | {g54_rigid:.2f}")
print("-" * 70)
base_e = None
for arm in ARMS:
    e, n = arm_mean(qa_edge[arm]); c, _ = arm_mean(qa_comp[arm])
    g, _ = g54_arm(arm)
    if arm == "base_agentic":
        base_e = e
    print(f"{arm:16} {e:8.2f} {c:8.2f} {n:9}   | {g:.2f}")

print("\n---- UPLIFT (Qwen edgeF1) ----")
print(f"  rigid -> base_agentic : {re_:.2f} -> {base_e:.2f}  (Δ {base_e - re_:+.2f})")
for arm in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    e, _ = arm_mean(qa_edge[arm])
    print(f"  base_agentic -> {arm:16}: {base_e:.2f} -> {e:.2f}  (Δ {e - base_e:+.2f})")

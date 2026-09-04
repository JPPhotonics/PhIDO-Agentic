"""Score the Nemotron-3-Ultra ablation (agentic shards + 3-way-sharded rigid) vs the REVISED
b3_gold_v2, over all 24 prompts, with gpt-5.4 and Qwen3.6-27B edge-F1 as comparison columns.
Pure scoring, no API. Reconstructs Topology from stored nodes + edge-strings (driver
serialized frozensets via default=str)."""
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")     # the REVISED gold
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")

import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

PROMPTS = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
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


def load_arm_files(pattern, key=None):
    """-> {arm: {pid: [rec]}} merged across shard files (skips .bak / .pre_shard)."""
    merged = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(ROOT / "benchmark" / "results" / pattern))):
        if f.endswith(".bak") or ".pre_shard" in f:
            continue
        d = json.load(open(f))
        for arm, byp in d.items():
            for pid, reps in byp.items():
                merged[arm][pid].extend(reps)
    return merged


# ---- Nemotron: agentic (2 shards) + rigid (3 shards) merged into one arm dict ----
nem_raw = load_arm_files("nemotron_trace_ablation_w*.json")
for arm, byp in load_arm_files("nemotron_rigid_baseline*.json").items():
    for pid, reps in byp.items():
        nem_raw[arm][pid].extend(reps)

nem_edge = {a: defaultdict(list) for a in ["rigid_baseline", *ARMS]}
nem_comp = {a: defaultdict(list) for a in ["rigid_baseline", *ARMS]}
nem_status = {a: Counter() for a in ["rigid_baseline", *ARMS]}
for arm in ["rigid_baseline", *ARMS]:
    for pid, reps in nem_raw.get(arm, {}).items():
        for rec in reps:
            nem_status[arm][rec.get("status")] += 1
            s = score(rec, pid)
            if s and s[0] is not None:
                nem_edge[arm][pid].append(s[0]); nem_comp[arm][pid].append(s[1])

# ---- Qwen comparison (agentic shards + single rigid store), same scorer ----
qwen_edge = defaultdict(lambda: defaultdict(list))
qw_raw = load_arm_files("qwen_trace_ablation_w*.json")
for arm, byp in load_arm_files("qwen_rigid_baseline*.json").items():
    for pid, reps in byp.items():
        qw_raw[arm][pid].extend(reps)
for arm, byp in qw_raw.items():
    for pid, reps in byp.items():
        for rec in reps:
            s = score(rec, pid)
            if s and s[0] is not None:
                qwen_edge[arm][pid].append(s[0])

# ---- gpt-5.4 (already scored in place vs the revised gold), all 24 prompts ----
g54 = json.load(open(ROOT / "benchmark" / "results" / "_ablation_correctness_gpt54_v2.json"))


def g54_arm(arm):
    pp = defaultdict(list)
    for pid in PROMPTS:
        for r in g54.get(arm, {}).get(pid, []):
            if isinstance(r.get("edgeF1"), (int, float)):
                pp[pid].append(r["edgeF1"])
    return arm_mean(pp)


G54_KEY = {"rigid_baseline": "baseline"}

print("\n==========  NEMOTRON-3-ULTRA-550B ablation vs revised b3_gold_v2 (n=24 prompts)  ==========")
print(f"{'arm':16} {'edgeF1':>8} {'compF1':>8} {'#prompts':>9} {'ok/total reps':>14}   | {'qwen':>5} {'gpt5.4':>6}")
base_e = None
for arm in ["rigid_baseline", *ARMS]:
    e, n = arm_mean(nem_edge[arm]); c, _ = arm_mean(nem_comp[arm])
    ok = nem_status[arm].get("ok", 0); tot = sum(nem_status[arm].values())
    q, _ = arm_mean(qwen_edge[arm])
    g, _ = g54_arm(G54_KEY.get(arm, arm))
    if arm == "base_agentic":
        base_e = e
    print(f"{arm:16} {e:8.2f} {c:8.2f} {n:9} {ok:>6}/{tot:<7}   | {q:5.2f} {g:6.2f}")
    if arm == "rigid_baseline":
        print("-" * 78)

print("\n---- per-arm status breakdown (non-ok) ----")
for arm in ["rigid_baseline", *ARMS]:
    bad = {k: v for k, v in nem_status[arm].items() if k != "ok"}
    if bad:
        print(f"  {arm:16} {bad}")

re_, _ = arm_mean(nem_edge["rigid_baseline"])
print("\n---- UPLIFT (Nemotron edgeF1) ----")
print(f"  rigid -> base_agentic : {re_:.2f} -> {base_e:.2f}  (Δ {base_e - re_:+.2f})")
for arm in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    e, n = arm_mean(nem_edge[arm])
    print(f"  base_agentic -> {arm:16}: {base_e:.2f} -> {e:.2f}  (Δ {e - base_e:+.2f})  [n={n} prompts]")

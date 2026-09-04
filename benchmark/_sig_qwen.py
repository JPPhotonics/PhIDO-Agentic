"""Paired significance (Wilcoxon, n=12 prompts) on Qwen edge-F1, + Qwen-vs-gpt5.4 uplift contrast."""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "qwen/qwen3.6-27b"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")
import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

PROMPTS = ["L3_01", "L3_02", "L3_04", "L3_07", "L3_09", "L3_10", "L3_12",
           "L4_08", "L4_09", "L4_10", "L4_11", "L4_12"]


def _edges(el):
    out = []
    for e in el:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2:
            out.append(frozenset(eps))
    return out


def _score(rec, pid):
    if rec.get("status") != "ok" or not rec.get("nodes"):
        return None
    try:
        return SC._score_topo(Topology(nodes=dict(rec["nodes"]), edges=_edges(rec.get("edges", [])),
                                       external={}), pid)["edgeF1"]
    except Exception:  # noqa: BLE001
        return None


# Qwen per-prompt mean edgeF1 vectors
qwen = defaultdict(lambda: defaultdict(list))
for f in glob.glob(str(ROOT / "benchmark/results/qwen_trace_ablation_w*.json")):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items():
            for r in reps:
                v = _score(r, pid)
                if v is not None:
                    qwen[arm][pid].append(v)
rd = json.load(open(ROOT / "benchmark/results/qwen_rigid_baseline.json"))["rigid_baseline"]
for pid, reps in rd.items():
    for r in reps:
        v = _score(r, pid)
        if v is not None:
            qwen["rigid_baseline"][pid].append(v)

# gpt-5.4 per-prompt mean edgeF1 (already scored vs revised gold; rigid arm = "baseline")
g54raw = json.load(open(ROOT / "benchmark/results/_ablation_correctness_gpt54_v2.json"))
g54 = defaultdict(lambda: defaultdict(list))
for arm, byp in g54raw.items():
    for pid in PROMPTS:
        for r in byp.get(pid, []):
            if isinstance(r.get("edgeF1"), (int, float)):
                g54[arm][pid].append(r["edgeF1"])


def pm(model, arm):
    return {p: st.mean(v) for p, v in model[arm].items() if v}


def paired(model, a, b):
    A, B = pm(model, a), pm(model, b)
    ps = sorted(set(A) & set(B))
    da, db = [A[p] for p in ps], [B[p] for p in ps]
    diff = [x - y for x, y in zip(da, db)]
    md = st.mean(diff)
    try:
        _, p = wilcoxon(da, db) if any(diff) else (0, 1.0)
    except ValueError:
        p = 1.0
    return md, p, len(ps), sum(1 for d in diff if d > 0), sum(1 for d in diff if d < 0)


def line(model, a, b):
    md, p, n, w, l = paired(model, a, b)
    star = "*" if p < 0.05 else ""
    return f"    {a} -> {b:18} Δ={md:+.3f}  w/l={w}/{l}  p={p:.3f}{star}"


print("\n===== QWEN edge-F1 paired Wilcoxon (n=12 prompts) =====")
print("  scaffold uplift:")
print(line(qwen, "rigid_baseline", "base_agentic"))
print("  per-feature (vs base_agentic):")
for a in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    print(line(qwen, "base_agentic", a))

print("\n===== gpt-5.4 edge-F1 paired Wilcoxon (same 12 prompts) =====")
print(line(g54, "baseline", "base_agentic"))
for a in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    print(line(g54, "base_agentic", a))

print("\n===== per-feature uplift: QWEN vs gpt-5.4 (mean Δ edge-F1) =====")
for a in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    qd = paired(qwen, "base_agentic", a)[0]
    gd = paired(g54, "base_agentic", a)[0]
    print(f"    {a:18}  Qwen Δ={qd:+.3f}   gpt-5.4 Δ={gd:+.3f}")

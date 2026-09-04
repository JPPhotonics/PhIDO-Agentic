"""Paired significance (Wilcoxon, n=24 prompts) on Nemotron edge-F1, + 3-model uplift contrast
(Nemotron vs Qwen3.6-27B vs gpt-5.4). Arms with no usable reps (outage-poisoned) are skipped."""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")
import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

PROMPTS = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
FEATURES = ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]


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


def load(tag):
    """-> {arm: {pid: [edgeF1 reps]}} for one RUN_TAG, merging all shard files."""
    m = defaultdict(lambda: defaultdict(list))
    pats = [f"benchmark/results/{tag}_trace_ablation_w*.json",
            f"benchmark/results/{tag}_rigid_baseline*.json"]
    for pat in pats:
        for f in sorted(glob.glob(str(ROOT / pat))):
            if f.endswith(".bak") or ".pre_shard" in f:
                continue
            for arm, byp in json.load(open(f)).items():
                for pid, reps in byp.items():
                    for r in reps:
                        v = _score(r, pid)
                        if v is not None:
                            m[arm][pid].append(v)
    return m


nem = load("nemotron")
qwen = load("qwen")

# gpt-5.4 per-prompt edgeF1 (already scored vs revised gold; rigid arm = "baseline")
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
    if not ps:
        return None
    da, db = [A[p] for p in ps], [B[p] for p in ps]
    diff = [x - y for x, y in zip(da, db)]
    md = st.mean(diff)
    try:
        _, p = wilcoxon(da, db) if any(diff) else (0, 1.0)
    except ValueError:
        p = 1.0
    return md, p, len(ps), sum(1 for d in diff if d > 0), sum(1 for d in diff if d < 0)


def line(model, a, b):
    r = paired(model, a, b)
    if r is None:
        return f"    {a} -> {b:18} (no usable reps — skipped)"
    md, p, n, w, l = r
    star = "*" if p < 0.05 else ""
    return f"    {a} -> {b:18} Δ={md:+.3f}  n={n}  w/l={w}/{l}  p={p:.3f}{star}"


print("\n===== NEMOTRON edge-F1 paired Wilcoxon (b <- a is Δ = b - a ... reported as b-a per pair order) =====")
print("  scaffold uplift:")
print(line(nem, "base_agentic", "rigid_baseline").replace("base_agentic -> rigid_baseline", "base_agentic vs rigid    "))
print("  per-feature (arm vs base_agentic; Δ>0 = feature helps):")
for a in FEATURES:
    print(line(nem, a, "base_agentic").replace(f"{a} -> base_agentic", f"{a:18} vs base"))

print("\n===== 3-model contrast: mean paired Δ edge-F1 (scaffold = agentic - rigid; features = arm - base) =====")
print(f"    {'contrast':22} {'nemotron':>9} {'qwen':>7} {'gpt5.4':>7}")


def d(model, a, b):
    r = paired(model, a, b)
    return f"{r[0]:+.3f}" if r else "   —  "


rigid_key = {"gpt": "baseline"}
print(f"    {'rigid -> base_agentic':22} {d(nem,'base_agentic','rigid_baseline'):>9} "
      f"{d(qwen,'base_agentic','rigid_baseline'):>7} {d(g54,'base_agentic','baseline'):>7}")
for a in FEATURES:
    print(f"    {'base -> ' + a:22} {d(nem,a,'base_agentic'):>9} "
          f"{d(qwen,a,'base_agentic'):>7} {d(g54,a,'base_agentic'):>7}")

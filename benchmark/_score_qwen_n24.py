"""Score the COMPLETE Qwen n=24 suite (agentic 5 arms merged across shards w0-w3 + rigid)
vs revised b3_gold_v2, with paired Wilcoxon significance and the gpt-5.4 contrast.

Merges all qwen_trace_ablation_w*.json shards (per (arm,pid) reps concatenate — disjoint rep
objects, so no double count), audits per-cell rep counts, and reports:
  - per-arm edgeF1 (+ coverage, + L3/L4 split)
  - rigid -> base_agentic scaffold jump (paired, matched scoreable prompts)
  - per-feature uplift vs base_agentic (paired)
  - Qwen-vs-gpt5.4 per-feature contrast
"""
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

ALL24 = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
ARMS = ["base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]


def _edges(el):
    out = []
    for e in el:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2:
            out.append(frozenset(eps))
    return out


# Infra failures are NOT the scaffold's fault -> exclude (return None). Everything else
# non-ok is a legitimate model/pipeline failure -> it produced no valid circuit, so it
# scores 0 (unconditional/production scoring; failures can't hide by being dropped).
INFRA = ("timeout", "EmptyResponse", "ENOSPC", "ratelimit", "RateLimit")


def score(rec, pid):
    status = rec.get("status", "")
    if status != "ok":
        if any(k in status for k in INFRA):
            return None          # infra: exclude
        return 0.0               # legitimate failure: score 0
    if not rec.get("nodes"):
        return 0.0               # ok status but empty netlist = failure
    try:
        return SC._score_topo(Topology(nodes=dict(rec["nodes"]), edges=_edges(rec.get("edges", [])),
                                       external={}), pid)["edgeF1"]
    except Exception:  # noqa: BLE001
        return 0.0               # produced an unscoreable topology = failure


# ---- load + merge agentic shards ----
qa = {a: defaultdict(list) for a in ARMS}       # arm -> pid -> [edgeF1]
qraw = {a: defaultdict(int) for a in ARMS}       # arm -> pid -> raw rep count (audit)
for f in glob.glob(str(ROOT / "benchmark/results/qwen_trace_ablation_w*.json")):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items():
            for r in reps:
                qraw[arm][pid] += 1
                v = score(r, pid)
                if v is not None:
                    qa[arm][pid].append(v)

# ---- rigid ----
qr = defaultdict(list)
rd = json.load(open(ROOT / "benchmark/results/qwen_rigid_baseline.json"))["rigid_baseline"]
for pid, reps in rd.items():
    for r in reps:
        v = score(r, pid)
        if v is not None:
            qr[pid].append(v)

# ---- gpt-5.4 (already scored vs revised gold; rigid arm keyed 'baseline') ----
g54raw = json.load(open(ROOT / "benchmark/results/_ablation_correctness_gpt54_v2.json"))
g54 = defaultdict(lambda: defaultdict(list))
for arm, byp in g54raw.items():
    for pid, reps in byp.items():
        for r in reps:
            if isinstance(r.get("edgeF1"), (int, float)):
                g54[arm][pid].append(r["edgeF1"])
            elif not any(k in r.get("status", "") for k in INFRA):
                g54[arm][pid].append(0.0)  # legitimate failure -> 0 (consistent with Qwen)


def pm(d):  # per-prompt mean -> {pid: mean}
    return {p: st.mean(v) for p, v in d.items() if v}


def arm_summary(d):
    p = pm(d)
    l3 = [v for k, v in p.items() if k.startswith("L3")]
    l4 = [v for k, v in p.items() if k.startswith("L4")]
    return (st.mean(p.values()) if p else float("nan"), len(p),
            st.mean(l3) if l3 else float("nan"), st.mean(l4) if l4 else float("nan"))


def paired(A, B):
    a, b = pm(A), pm(B)
    ps = sorted(set(a) & set(b))
    da, db = [a[p] for p in ps], [b[p] for p in ps]
    diff = [x - y for x, y in zip(da, db)]
    md = st.mean(diff) if diff else float("nan")
    try:
        _, pv = wilcoxon(da, db) if any(diff) else (0, 1.0)
    except ValueError:
        pv = 1.0
    return md, pv, len(ps), sum(1 for x in diff if x > 0), sum(1 for x in diff if x < 0)


# ===== audit =====
print("===== SHARD AUDIT (reps per (arm,pid); expect 3) =====")
anomalies = []
for a in ARMS:
    for pid in ALL24:
        n = qraw[a][pid]
        if n != 3:
            anomalies.append(f"{a}/{pid}={n}")
print("  anomalies (!=3 reps):", anomalies or "none")
# failures now score 0 (not dropped), so every arm covers all 24 prompts. Report the
# failure load being folded in as zeros, per arm.
print("  failures folded in as edgeF1=0 (per arm, incl rigid):")
for a in ARMS:
    nfail = sum(1 for pid in ALL24 for v in qa[a][pid] if v == 0.0)
    print(f"    {a:16} {nfail} zero-scored reps")
print(f"    {'rigid':16} {sum(1 for pid in ALL24 for v in qr[pid] if v == 0.0)} zero-scored reps")

# ===== per-arm table =====
print("\n===== QWEN per-arm edgeF1 (per-prompt mean; L3/L4 split) =====")
print(f"  {'arm':16} {'edgeF1':>7} {'#prompts':>9} {'L3':>6} {'L4':>6}")
rm, rn, rl3, rl4 = arm_summary(qr)
print(f"  {'rigid_baseline':16} {rm:7.3f} {rn:9} {rl3:6.3f} {rl4:6.3f}")
print("  " + "-" * 46)
for a in ARMS:
    m, n, l3, l4 = arm_summary(qa[a])
    print(f"  {a:16} {m:7.3f} {n:9} {l3:6.3f} {l4:6.3f}")

# ===== scaffold jump + per-feature (paired) =====
print("\n===== SCAFFOLD JUMP + PER-FEATURE (paired Wilcoxon) =====")
md, pv, n, w, l = paired(qa["base_agentic"], qr)
print(f"  rigid -> base_agentic   Δ={md:+.3f}  n={n}  agentic>rigid {w}/{l}  p={pv:.4f}{'*' if pv<0.05 else ''}")
for a in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    md, pv, n, w, l = paired(qa[a], qa["base_agentic"])
    print(f"  base_agentic -> {a:16} Δ={md:+.3f}  n={n}  feat>base {w}/{l}  p={pv:.4f}{'*' if pv<0.05 else ''}")

# ===== gpt-5.4 same comparisons + contrast =====
print("\n===== gpt-5.4 (same prompts) =====")
md, pv, n, w, l = paired(g54["base_agentic"], g54["baseline"])
print(f"  rigid -> base_agentic   Δ={md:+.3f}  n={n}  p={pv:.4f}{'*' if pv<0.05 else ''}")
for a in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    md, pv, n, w, l = paired(g54[a], g54["base_agentic"])
    print(f"  base_agentic -> {a:16} Δ={md:+.3f}  n={n}  p={pv:.4f}{'*' if pv<0.05 else ''}")

print("\n===== per-feature uplift CONTRAST: Qwen vs gpt-5.4 (mean Δ) =====")
print(f"  {'feature':16} {'Qwen':>8} {'gpt-5.4':>9}")
print(f"  {'scaffold(rigid→ag)':16} {paired(qa['base_agentic'], qr)[0]:+8.3f} {paired(g54['base_agentic'], g54['baseline'])[0]:+9.3f}")
for a in ["base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]:
    print(f"  {a:16} {paired(qa[a], qa['base_agentic'])[0]:+8.3f} {paired(g54[a], g54['base_agentic'])[0]:+9.3f}")

"""Nemotron-3-Ultra ablation scored under the THESIS convention (failure-as-zero), mirroring
_score_qwen_n24.py exactly: infra statuses excluded, every other non-ok rep scores 0, empty or
unscoreable netlists score 0; per-prompt mean over reps, then mean over the 24 prompts.
Paired Wilcoxon over prompts for each transition. Pure scoring, no API."""
import glob, json, os, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")
import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

PROMPTS = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
ARMS = ["rigid_baseline", "base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]
INFRA = ("timeout", "EmptyResponse", "ENOSPC", "ratelimit", "RateLimit", "NotFoundError")

def _edges(el):
    out = []
    for e in el:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2: out.append(frozenset(eps))
    return out

def score(rec, pid):
    """-> (edgeF1, compF1) ; None = infra (excluded)."""
    status = rec.get("status", "")
    if status != "ok":
        return None if any(k in status for k in INFRA) else (0.0, 0.0)
    if not rec.get("nodes"): return (0.0, 0.0)
    try:
        s = SC._score_topo(Topology(nodes=dict(rec["nodes"]), edges=_edges(rec.get("edges", [])), external={}), pid)
        return (s["edgeF1"], s["compF1"])
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)

merged = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(str(ROOT/"benchmark/results/nemotron_trace_ablation_w*.json"))) + \
         sorted(glob.glob(str(ROOT/"benchmark/results/nemotron_rigid_baseline_w*.json"))):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items(): merged[arm][pid].extend(reps)

E = {a: {} for a in ARMS}; C = {a: {} for a in ARMS}; zeros = Counter(); infra = Counter(); nrep = Counter()
for a in ARMS:
    for pid in PROMPTS:
        ev, cv = [], []
        for rec in merged[a][pid]:
            s = score(rec, pid)
            if s is None: infra[a] += 1; continue
            nrep[a] += 1
            if s[0] == 0.0: zeros[a] += 1
            ev.append(s[0]); cv.append(s[1])
        E[a][pid] = st.mean(ev) if ev else float("nan"); C[a][pid] = st.mean(cv) if cv else float("nan")

def mean_over(d, pids): return st.mean(d[p] for p in pids)
L3 = PROMPTS[:12]; L4 = PROMPTS[12:]
print("Nemotron-3-Ultra-550B (OpenRouter free) — failure-as-zero, b3_gold_v2, n=24 prompts, K=3\n")
print(f"{'arm':18} {'edgeF1':>7} {'L3':>6} {'L4':>6} {'compF1':>7} {'zero-reps':>10} {'infra-excl':>10} {'reps':>5}")
for a in ARMS:
    print(f"{a:18} {mean_over(E[a],PROMPTS):7.3f} {mean_over(E[a],L3):6.3f} {mean_over(E[a],L4):6.3f} {mean_over(C[a],PROMPTS):7.3f} {zeros[a]:>10} {infra[a]:>10} {nrep[a]:>5}")

def contrast(a, b):
    da = [E[a][p] for p in PROMPTS]; db = [E[b][p] for p in PROMPTS]
    diff = [y - x for x, y in zip(da, db)]
    w = sum(d > 0 for d in diff); l = sum(d < 0 for d in diff)
    p = wilcoxon(da, db).pvalue if any(diff) else 1.0
    return st.mean(diff), w, l, p

print("\nTransitions (paired Wilcoxon over 24 prompts, edge-F1):")
for a, b in [("rigid_baseline","base_agentic"),("base_agentic","base_plus_kg"),("base_agentic","base_plus_gate"),
             ("base_agentic","base_plus_critic"),("base_agentic","full")]:
    d, w, l, p = contrast(a, b)
    print(f"  {a:16} -> {b:17} Δ={d:+.3f}  win/loss={w}/{l}  p={p:.3f}")

print("\nNon-ok status breakdown per arm:")
for a in ARMS:
    stc = Counter(r.get("status") for pid in PROMPTS for r in merged[a][pid] if r.get("status") != "ok")
    print(f"  {a:16} {dict(stc)}")
print("\nPer-prompt edge-F1 (failure-as-zero):")
print("pid      " + " ".join(f"{a[:9]:>9}" for a in ARMS))
for pid in PROMPTS:
    print(f"{pid:8} " + " ".join(f"{E[a][pid]:9.2f}" for a in ARMS))

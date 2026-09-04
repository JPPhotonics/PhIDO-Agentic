"""Nemotron final scoring v2: classify every non-ok rep from the worker logs as
provider load-shed (infra, EXCLUDED per thesis convention) or genuine failure (scored 0),
by joining log result lines to banked records on (arm, pid, status, n_llm_calls, latency)."""
import glob, json, os, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
os.environ["E2_GOLD"] = str(ROOT/"benchmark/b3_gold_v2.json"); os.environ["E2_PROMPTS"] = str(ROOT/"benchmark/e2_prompts_v2.json")
import _score_correctness as SC  # noqa
from topology_eval import Topology  # noqa
PROMPTS = [f"L3_{i:02d}" for i in range(1,13)] + [f"L4_{i:02d}" for i in range(1,13)]
ARMS = ["rigid_baseline","base_agentic","base_plus_kg","base_plus_gate","base_plus_critic","full"]
INFRA = ("timeout","EmptyResponse","ENOSPC","ratelimit","RateLimit","NotFoundError")

# ---- 1. log parse: result lines + shed lookback ----
RES = re.compile(r"\[w\d+ (\S+)\s+(L[34]_\d\d) rep\d/3\] (.+?) nodes=\d+ edges=\d+ calls=(\d+) ([\d.]+)s")
shed_keys = set(); genuine_ctx = {}; shed_by_day = Counter()
for lf in glob.glob(str(ROOT/"benchmark/logs/nemo_*.log")):
    lines = open(lf, errors="ignore").read().splitlines()
    for i, ln in enumerate(lines):
        if "load-shed" in ln:
            m = re.search(r"\[(\d\d)/(\d\d)/\d\d", " ".join(lines[max(0,i-3):i+1]))
            if m: shed_by_day[f"{m.group(1)}-{m.group(2)}"] += 1
        m = RES.search(ln)
        if not m or m.group(3) == "ok": continue
        arm, pid, status, calls, lat = m.group(1), m.group(2), m.group(3), int(m.group(4)), float(m.group(5))
        back = "\n".join(lines[max(0, i-400):i])
        key = (arm, pid, status, calls, round(lat))
        if "shed retry 10/10" in back or "load-shed" in back or re.search(r"attempt 4/4|RateLimit", back):
            shed_keys.add(key)
        else:
            genuine_ctx[key] = [l.strip()[:90] for l in lines[max(0,i-3):i] if l.strip()]

# ---- 2. banked records ----
merged = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(str(ROOT/"benchmark/results/nemotron_trace_ablation_w*.json"))) + sorted(glob.glob(str(ROOT/"benchmark/results/nemotron_rigid_baseline_w*.json"))):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items(): merged[arm][pid].extend(reps)

def _edges(el):
    out=[]
    for e in el:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e,str) else list(e)
        if len(eps)==2: out.append(frozenset(eps))
    return out
def topo_score(rec, pid):
    try: s = SC._score_topo(Topology(nodes=dict(rec["nodes"]), edges=_edges(rec.get("edges",[])), external={}), pid); return s["edgeF1"], s["compF1"]
    except Exception: return 0.0, 0.0

cls = {a: Counter() for a in ARMS}
E = {a: {} for a in ARMS}; C = {a: {} for a in ARMS}
E_worst = {a: {} for a in ARMS}
for a in ARMS:
    for pid in PROMPTS:
        ev, cv, evw = [], [], []
        for rec in merged[a][pid]:
            status = rec.get("status","")
            if status == "ok" and rec.get("nodes"):
                e, c = topo_score(rec, pid); ev.append(e); cv.append(c); evw.append(e); cls[a]["ok"] += 1; continue
            if status == "ok": ev.append(0.0); cv.append(0.0); evw.append(0.0); cls[a]["empty-netlist"] += 1; continue
            if any(k in status for k in INFRA): cls[a]["infra(status)"] += 1; continue
            key = (a, pid, status, rec.get("n_llm_calls"), round(rec.get("latency_s") or 0))
            if key in shed_keys:
                cls[a]["infra(load-shed)"] += 1; evw.append(0.0); continue   # excluded in main; 0 in worst-case
            cls[a]["genuine-fail"] += 1; ev.append(0.0); cv.append(0.0); evw.append(0.0)
        E[a][pid] = st.mean(ev) if ev else float("nan"); C[a][pid] = st.mean(cv) if cv else float("nan")
        E_worst[a][pid] = st.mean(evw) if evw else float("nan")

def m(d, pids):
    vals = [d[p] for p in pids if d[p] == d[p]]
    return st.mean(vals) if vals else float("nan")
L3, L4 = PROMPTS[:12], PROMPTS[12:]
print("Provider load-shed warnings per day (all Nemotron logs):", dict(sorted(shed_by_day.items())))
print("\nRep classification per arm:")
for a in ARMS: print(f"  {a:18} {dict(cls[a])}")
print("\nMAIN (thesis convention: load-shed = infra, excluded; genuine failures = 0):")
print(f"{'arm':18} {'edgeF1':>7} {'L3':>6} {'L4':>6} {'compF1':>7} {'prompts':>7}")
for a in ARMS:
    n_p = sum(1 for p in PROMPTS if E[a][p] == E[a][p])
    print(f"{a:18} {m(E[a],PROMPTS):7.3f} {m(E[a],L3):6.3f} {m(E[a],L4):6.3f} {m(C[a],PROMPTS):7.3f} {n_p:>7}")
print("\nWORST CASE (load-shed scored 0, as in the first pass):")
for a in ARMS: print(f"  {a:18} edgeF1={m(E_worst[a],PROMPTS):.3f}")
def contrast(D, a, b):
    pairs = [(D[a][p], D[b][p]) for p in PROMPTS if D[a][p]==D[a][p] and D[b][p]==D[b][p]]
    diff = [y-x for x,y in pairs]
    p = wilcoxon([x for x,_ in pairs],[y for _,y in pairs]).pvalue if any(diff) else 1.0
    return st.mean(diff), sum(d>0 for d in diff), sum(d<0 for d in diff), p, len(pairs)
print("\nTransitions, MAIN scoring (paired Wilcoxon over prompts with data in both arms):")
for a,b in [("rigid_baseline","base_agentic"),("base_agentic","base_plus_kg"),("base_agentic","base_plus_gate"),("base_agentic","base_plus_critic"),("base_agentic","full")]:
    d,w,l,p,n = contrast(E,a,b); print(f"  {a:16} -> {b:17} Δ={d:+.3f} win/loss={w}/{l} p={p:.3f} n={n}")
print("\nGenuine (non-shed) failures and their log context:")
for k, ctx in list(genuine_ctx.items())[:12]: print(f"  {k[0]:16} {k[1]} {k[2][:45]:45} | {ctx[-1][:80] if ctx else ''}")

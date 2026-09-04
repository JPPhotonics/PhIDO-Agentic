"""Nemotron final scoring v3 (2026-08-30): classification by RECORDED STATUS only.
ok+nodes -> topology score vs b3_gold_v2; ok without nodes -> empty netlist (0);
retryable/infra statuses -> excluded (must be zero after the paid-route refill);
every other status -> genuine failure (0), per the thesis failure-as-zero rule.
Writes benchmark/results/_nemotron_scoring_final.md."""
import glob, json, os, re, statistics as st, datetime
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b"
os.environ["E2_GOLD"] = str(ROOT/"benchmark/b3_gold_v2.json"); os.environ["E2_PROMPTS"] = str(ROOT/"benchmark/e2_prompts_v2.json")
import _score_correctness as SC  # noqa
from topology_eval import Topology  # noqa
PROMPTS = [f"L3_{i:02d}" for i in range(1,13)] + [f"L4_{i:02d}" for i in range(1,13)]
L3, L4 = PROMPTS[:12], PROMPTS[12:]
ARMS = ["rigid_baseline","base_agentic","base_plus_kg","base_plus_gate","base_plus_critic","full"]
RETRY = ("error:EmptyResponseError","error:APIConnectionError","error:InternalServerError","error:APITimeoutError",
         "error:RateLimitError","error:NotFoundError","error:APIStatusError","error:PermissionDeniedError",
         "failed:Interpreter failed to produce DesignIntent",
         "failed:LLM refused to produce DesignIntent: OpenRouter structured output failed after 3 attempts (last: finish_reason=error")
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
def topo(rec, pid):
    try: s = SC._score_topo(Topology(nodes=dict(rec["nodes"]), edges=_edges(rec.get("edges",[])), external={}), pid); return s["edgeF1"], s["compF1"]
    except Exception: return 0.0, 0.0
cls = {a: Counter() for a in ARMS}; census = {a: Counter() for a in ARMS}
E = {a:{} for a in ARMS}; C = {a:{} for a in ARMS}; nreps = {a:0 for a in ARMS}
for a in ARMS:
    for pid in PROMPTS:
        ev, cv = [], []
        for rec in merged[a][pid][:]:
            status = str(rec.get("status",""))
            if status.startswith(RETRY): cls[a]["infra(excluded)"] += 1; continue
            nreps[a] += 1
            if status != "ok": cls[a]["genuine-fail"] += 1; census[a]["no netlist"] += 1; ev.append(0.0); cv.append(0.0); continue
            if not rec.get("nodes"): cls[a]["empty-netlist"] += 1; census[a]["no netlist"] += 1; ev.append(0.0); cv.append(0.0); continue
            e, c = topo(rec, pid); ev.append(e); cv.append(c); cls[a]["ok"] += 1
            if not rec.get("edges"): census[a]["edgeless"] += 1
            elif e >= 1.0-1e-9: census[a]["exact"] += 1
            elif c >= 1.0-1e-9: census[a]["miswired"] += 1
            else: census[a]["wrong parts"] += 1
        E[a][pid] = st.mean(ev) if ev else float("nan"); C[a][pid] = st.mean(cv) if cv else float("nan")
def m(d, pids):
    v=[d[p] for p in pids if d[p]==d[p]]; return st.mean(v) if v else float("nan")
def contrast(D, a, b):
    pairs=[(D[a][p],D[b][p]) for p in PROMPTS if D[a][p]==D[a][p] and D[b][p]==D[b][p]]
    diff=[y-x for x,y in pairs]; p = wilcoxon([x for x,_ in pairs],[y for _,y in pairs]).pvalue if any(diff) else 1.0
    return st.mean(diff), sum(d>0 for d in diff), sum(d<0 for d in diff), p, len(pairs)
def holm(ps):
    order=sorted(range(len(ps)), key=lambda i: ps[i]); adj=[0]*len(ps); run=0
    for rank,i in enumerate(order): run=max(run, ps[i]*(len(ps)-rank)); adj[i]=min(1.0,run)
    return adj
out=[]; P=out.append
P(f"# Nemotron-3-Ultra-550B ablation, final scoring v3 (status-based)  \n_generated {datetime.datetime.now():%Y-%m-%d %H:%M}_\n")
P("Model `nvidia/nemotron-3-ultra-550b-a55b` via OpenRouter; 386 reps on the `:free` route (NVIDIA), 46 reps refilled on the paid route (BaseTen/Venice, `PHIDO_JSON_MODE=0`, provider order BaseTen,Venice) after the free tier stalled. Context window 128k (harness default) throughout. Golden reference b3_gold_v2; 24 prompts (12 L3, 12 L4) x 3 repeats; failure-as-zero.\n")
P("## Rep classification per arm\n\n| arm | reps | ok | genuine fail | empty netlist | infra (excluded) |\n|---|---|---|---|---|---|")
for a in ARMS: P(f"| {a} | {nreps[a]} | {cls[a]['ok']} | {cls[a]['genuine-fail']} | {cls[a]['empty-netlist']} | {cls[a]['infra(excluded)']} |")
P("\n## Mean scores (failure-as-zero; per-prompt means over 3 repeats, then over prompts)\n\n| arm | edge-F1 | edge-F1 L3 | edge-F1 L4 | comp-F1 | comp-F1 L4 |\n|---|---|---|---|---|---|")
for a in ARMS: P(f"| {a} | {m(E[a],PROMPTS):.3f} | {m(E[a],L3):.3f} | {m(E[a],L4):.3f} | {m(C[a],PROMPTS):.3f} | {m(C[a],L4):.3f} |")
T=[("rigid_baseline","base_agentic"),("base_agentic","base_plus_kg"),("base_agentic","base_plus_gate"),("base_agentic","base_plus_critic"),("base_agentic","full")]
rows=[contrast(E,a,b) for a,b in T]; adj=holm([r[3] for r in rows])
P("\n## Transitions on edge-F1 (paired Wilcoxon over 24 prompts; Holm over the five)\n\n| transition | Δ | win/loss | p | p_Holm | n |\n|---|---|---|---|---|---|")
for (a,b),(d,w,l,p,n),pa in zip(T,rows,adj): P(f"| {a} → {b} | {d:+.3f} | {w}/{l} | {p:.3f} | {pa:.3f} | {n} |")
rows_c=[contrast(C,a,b) for a,b in T]; adj_c=holm([r[3] for r in rows_c])
P("\n## Transitions on component-F1\n\n| transition | Δ | win/loss | p | p_Holm |\n|---|---|---|---|---|")
for (a,b),(d,w,l,p,n),pa in zip(T,rows_c,adj_c): P(f"| {a} → {b} | {d:+.3f} | {w}/{l} | {p:.3f} | {pa:.3f} |")
base=m(E["base_agentic"],PROMPTS); pred=base+sum(m(E[x],PROMPTS)-base for x in ["base_plus_kg","base_plus_gate","base_plus_critic"])
best=max(m(E[x],PROMPTS) for x in ["base_plus_kg","base_plus_gate","base_plus_critic"])
P(f"\n## Composition\n\nadditive prediction for full = {pred:.3f}; observed full = {m(E['full'],PROMPTS):.3f} (residual {m(E['full'],PROMPTS)-pred:+.3f}); best single feature = {best:.3f} (full − best = {m(E['full'],PROMPTS)-best:+.3f})\n")
P("## Failure census (runs per mode, all scored reps)\n\n| arm | exact | miswired | wrong parts | edgeless | no netlist |\n|---|---|---|---|---|---|")
for a in ARMS: P(f"| {a} | {census[a]['exact']} | {census[a]['miswired']} | {census[a]['wrong parts']} | {census[a]['edgeless']} | {census[a]['no netlist']} |")
P("\n## Per-prompt edge-F1\n\n| prompt | " + " | ".join(ARMS) + " |\n|---|" + "---|"*len(ARMS))
for p in PROMPTS: P(f"| {p} | " + " | ".join(f"{E[a][p]:.2f}" for a in ARMS) + " |")
txt="\n".join(out); open(ROOT/"benchmark/results/_nemotron_scoring_final.md","w").write(txt); print(txt[:6000])

"""How much of the edge-F1 signal is PORT-NAME DRIFT rather than wrong topology?

Typed edges are keyed `CLASS.port -- CLASS.port`, so a design that wires the right components in
the right pattern but invents port names (e.g. `i1`/`lower_in` where the kit exposes `o1..o4`)
scores edge-F1 = 0 despite being topologically defensible. Tracing L4_06 found four such reps.

This audit finds every rep in the corpus with that signature and reports what re-keying would do,
so the thesis can separate "the pipeline wired it wrong" from "the pipeline named ports wrong".

Signature (deterministic, conservative):
  status ok, edges present, component multiset == gold, edge-F1 == 0, and the predicted edges'
  port tokens are DISJOINT from the gold port vocabulary of the same component classes.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results"
DB = json.load(open(RES / "_transition_cases.json"))
ALL, GOLD = DB["all_reps"], DB["gold"]
ARMS = ["rigid_baseline", "baseline", "base_agentic", "base_plus_kg", "base_plus_gate",
        "base_plus_critic", "full_minus_kg", "full_minus_gate", "full_minus_critic", "full"]


def endpoints(edge_str):
    """'CLASS.port -- CLASS.port' -> [(class, port), ...]"""
    out = []
    for side in str(edge_str).split(" -- "):
        side = side.strip()
        if "." in side:
            cls, _, port = side.rpartition(".")
            out.append((cls, port))
    return out


def port_vocab(edge_strs):
    v = defaultdict(set)
    for e in edge_strs:
        for cls, port in endpoints(e):
            v[cls].add(port)
    return v


rows = []
per_arm = defaultdict(lambda: Counter())
for model, byarm in ALL.items():
    for arm, byp in byarm.items():
        for pid, reps in byp.items():
            gv = port_vocab(GOLD[pid]["gold_edges"])
            gcls = Counter({k: v for k, v in GOLD[pid]["gold_classes"]})
            for i, r in enumerate(reps, 1):
                per_arm[(model, arm)]["reps"] += 1
                if str(r.get("status")) not in ("ok", "scored"):
                    continue
                if not r.get("n_edges") or not isinstance(r.get("edgeF1"), (int, float)):
                    continue
                if r["edgeF1"] > 0:
                    continue
                pcls = Counter({k: v for k, v in (r.get("pred_classes") or [])})
                if pcls != gcls:
                    continue                       # wrong parts: not a pure naming issue
                pv = port_vocab(r.get("extra") or [])
                shared = [c for c in pv if c in gv and (pv[c] & gv[c])]
                if shared:
                    continue                       # some ports do match: real wiring error
                if not pv:
                    continue
                per_arm[(model, arm)]["port_drift"] += 1
                rows.append({"model": model, "arm": arm, "prompt": pid, "rep": i,
                             "n_edges": r["n_edges"],
                             "gold_ports": {c: sorted(gv[c]) for c in sorted(gv)},
                             "pred_ports": {c: sorted(pv[c]) for c in sorted(pv)}})

print("=" * 78)
print("PORT-NAME-DRIFT AUDIT — reps scoring edge-F1 = 0 with the CORRECT component multiset")
print("whose predicted port names share nothing with the gold port vocabulary")
print("=" * 78)
print(f"{'model':6} {'arm':18} {'reps':>5} {'port-drift':>10} {'%':>5}")
for model in ("qwen", "gpt54"):
    for arm in ARMS:
        c = per_arm.get((model, arm))
        if not c:
            continue
        d = c["port_drift"]
        print(f"{model:6} {arm:18} {c['reps']:5} {d:10} {100*d/max(c['reps'],1):5.1f}")

print(f"\nTOTAL port-drift reps: {len(rows)}")
byp = Counter(f"{r['model']}|{r['prompt']}" for r in rows)
print("by prompt:", dict(byp.most_common()))

if rows:
    print("\nexamples (gold port vocabulary vs invented one):")
    for r in rows[:6]:
        print(f"  {r['model']} {r['arm']} {r['prompt']} rep{r['rep']} ({r['n_edges']} edges)")
        print(f"     gold: {r['gold_ports']}")
        print(f"     pred: {r['pred_ports']}")

json.dump(rows, open(RES / "_port_drift_audit.json", "w"), indent=1)
print(f"\nwrote {len(rows)} records -> results/_port_drift_audit.json")

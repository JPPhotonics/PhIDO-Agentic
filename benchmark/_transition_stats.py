"""Quantitative backbone for the transition forensics: (a) a per-transition case census,
(b) a rep-level NOISE FLOOR so per-prompt deltas can be judged against within-arm variance,
(c) a structural census (edgeless netlists, empty netlists, status mix) across every arm.

Reads the case DB written by _transition_cases.py plus the raw stores. No API, no scoring.
"""
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results"
DB = json.load(open(RES / "_transition_cases.json"))
CASES = DB["cases"]

TRS = ["rigid_baseline->base_agentic", "base_agentic->base_plus_kg",
       "base_agentic->base_plus_gate", "base_agentic->base_plus_critic",
       "base_agentic->full",
       "full_minus_kg->full", "full_minus_gate->full", "full_minus_critic->full"]
SHORT = {t: (t.split("->")[1].replace("base_plus_", "+").replace("base_agentic", "base")
             if not t.startswith("full_minus")
             else "LOO:" + t.split("->")[0].replace("full_minus_", "+"))
         for t in TRS}
EPS = 1e-9


def reps_f1(reps):
    return [r["edgeF1"] for r in reps if isinstance(r.get("edgeF1"), (int, float))]


# ---------------------------------------------------------------- (a) case census
print("=" * 78)
print("(a) CASE CENSUS — differing cells per transition (of 24 prompts each)")
print("=" * 78)
print(f"{'model':6} {'transition':16} {'cases':>5} {'improv':>6} {'regr':>5} {'stat-only':>9} "
      f"{'mean|Δ|':>8} {'max|Δ|':>7}")
for model in ("qwen", "gpt54"):
    for t in TRS:
        cs = [c for c in CASES if c["model"] == model and c["transition"] == t]
        if not cs:
            continue
        imp = [c for c in cs if c["delta"] > EPS]
        reg = [c for c in cs if c["delta"] < -EPS]
        so = [c for c in cs if abs(c["delta"]) <= EPS]
        ad = [abs(c["delta"]) for c in cs if abs(c["delta"]) > EPS]
        print(f"{model:6} {SHORT[t]:16} {len(cs):5} {len(imp):6} {len(reg):5} {len(so):9} "
              f"{(st.mean(ad) if ad else 0):8.3f} {(max(ad) if ad else 0):7.3f}")

# ------------------------------------------------------------- (b) noise floor
# Within-arm rep spread: for every (model, arm, prompt) with >=2 scored reps, the rep SD.
# A per-prompt delta is "inside the noise band" if |Δ| <= combined SE of the two arm means.
print()
print("=" * 78)
print("(b) NOISE FLOOR — within-arm rep variability vs between-arm deltas")
print("=" * 78)
arm_reps = defaultdict(list)          # (model, arm, pid) -> [f1]
for c in CASES:
    a, b = c["transition"].split("->")
    arm_reps[(c["model"], a, c["prompt"])] = reps_f1(c["reps_a"])
    arm_reps[(c["model"], b, c["prompt"])] = reps_f1(c["reps_b"])

for model in ("qwen", "gpt54"):
    sds = [st.pstdev(v) for (m, _, _), v in arm_reps.items() if m == model and len(v) >= 2]
    nonzero = [s for s in sds if s > EPS]
    print(f"\n{model}: {len(sds)} (arm,prompt) cells with >=2 scored reps; "
          f"{len(nonzero)} ({100*len(nonzero)/max(len(sds),1):.0f}%) have non-zero rep spread")
    if sds:
        print(f"  rep SD: mean={st.mean(sds):.3f}  median={st.median(sds):.3f}  "
              f"p90={sorted(sds)[int(0.9*(len(sds)-1))]:.3f}  max={max(sds):.3f}")

print(f"\n{'model':6} {'transition':16} {'Δ≠0':>4} {'inside noise':>12} {'outside':>8}  "
      f"(inside = |Δ| <= sqrt(SE_a^2+SE_b^2))")
noise_flag = {}
for model in ("qwen", "gpt54"):
    for t in TRS:
        cs = [c for c in CASES if c["model"] == model and c["transition"] == t
              and abs(c["delta"]) > EPS]
        inside = 0
        for c in cs:
            va, vb = reps_f1(c["reps_a"]), reps_f1(c["reps_b"])
            sea = st.pstdev(va) / (len(va) ** 0.5) if len(va) >= 2 else 0.0
            seb = st.pstdev(vb) / (len(vb) ** 0.5) if len(vb) >= 2 else 0.0
            band = (sea ** 2 + seb ** 2) ** 0.5
            hit = abs(c["delta"]) <= band + EPS
            noise_flag[(c["model"], c["transition"], c["prompt"])] = hit
            inside += hit
        if cs:
            print(f"{model:6} {SHORT[t]:16} {len(cs):4} {inside:12} {len(cs)-inside:8}")

# ------------------------------------------------- (c) structural census per arm
print()
print("=" * 78)
print("(c) STRUCTURAL CENSUS — per arm, over ALL reps in the raw stores")
print("=" * 78)


def census_qwen():
    out = defaultdict(lambda: Counter())
    files = sorted(glob.glob(str(RES / "qwen_trace_ablation_w*.json")))
    files += [f for f in glob.glob(str(RES / "qwen_rigid_baseline*.json"))
              if not f.endswith(".bak") and ".pre_shard" not in f]
    for f in files:
        for arm, byp in json.load(open(f)).items():
            for _pid, reps in byp.items():
                for r in reps:
                    c = out[arm]
                    c["reps"] += 1
                    sstat = str(r.get("status"))
                    c["ok" if sstat == "ok" else ("failed" if sstat.startswith("failed")
                                                  else "error")] += 1
                    if sstat == "ok":
                        n, e = r.get("nodes") or {}, r.get("edges") or []
                        if not n:
                            c["empty_netlist(no nodes)"] += 1
                        elif not e:
                            c["edgeless(nodes,no edges)"] += 1
    return out


def census_g54():
    out = defaultdict(lambda: Counter())
    d = json.load(open(RES / "_ablation_correctness_gpt54_v2.json"))
    for arm, byp in d.items():
        for _pid, reps in byp.items():
            for r in reps:
                c = out[arm]
                c["reps"] += 1
                sstat = str(r.get("status"))
                c["ok" if sstat == "scored" else ("failed" if sstat.startswith("failed")
                                                  else "error")] += 1
                if sstat == "scored":
                    n, e = r.get("pred_node_map") or {}, r.get("pred_edges") or []
                    if not n:
                        c["empty_netlist(no nodes)"] += 1
                    elif not e:
                        c["edgeless(nodes,no edges)"] += 1
    return out


ORDER = ["rigid_baseline", "baseline", "base_agentic", "base_plus_kg", "base_plus_gate",
         "base_plus_critic", "full"]
for model, cen in (("qwen", census_qwen()), ("gpt54", census_g54())):
    print(f"\n{model}:")
    print(f"  {'arm':16} {'reps':>5} {'ok':>4} {'err':>4} {'fail':>5} "
          f"{'edgeless':>9} {'empty':>6}")
    for arm in ORDER:
        if arm not in cen:
            continue
        c = cen[arm]
        print(f"  {arm:16} {c['reps']:5} {c['ok']:4} {c['error']:4} {c['failed']:5} "
              f"{c['edgeless(nodes,no edges)']:9} {c['empty_netlist(no nodes)']:6}")

json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in noise_flag.items()},
          open(RES / "_transition_noise_flags.json", "w"), indent=1)
print(f"\nwrote noise flags for {len(noise_flag)} non-zero-delta cases -> "
      f"results/_transition_noise_flags.json")

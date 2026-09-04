"""L4 failure-mode analysis for the E2 9-arm correctness ablation.

Pure post-hoc: reads gold v2 + the ablation JSON. For each L4 prompt, compares
predicted component multiset / edge count against gold, per arm, and classifies
the dominant failure signature.
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
GOLD = json.loads((HERE / "b3_gold_v2.json").read_text())["gold"]
RES = json.loads((HERE / "results" / "_ablation_correctness_gpt54_v2.json").read_text())

# gold indexed by id, L4 only
gold = {e["id"]: e for e in GOLD if e["level"] == 4}
L4_IDS = sorted(gold)


def gold_counts(e):
    return Counter(e["nodes"].values()), len(e["edges"])


def rep_counts(rep):
    return Counter(rep.get("pred_nodes", [])), len(rep.get("pred_edges", []))


def multiset_diff(gc, pc):
    """Return (missing, extra) component-type Counters."""
    missing = Counter()
    extra = Counter()
    for t in set(gc) | set(pc):
        d = pc[t] - gc[t]
        if d < 0:
            missing[t] = -d
        elif d > 0:
            extra[t] = d
    return missing, extra


print("=" * 90)
print("L4 gold shape (component multiset + edge count)")
print("=" * 90)
for pid in L4_IDS:
    gc, ge = gold_counts(gold[pid])
    top = ", ".join(f"{t}×{n}" for t, n in gc.most_common())
    print(f"{pid}: {sum(gc.values())} nodes / {ge} edges | {top}")

arms = list(RES.keys())

print("\n" + "=" * 90)
print("Per-arm L4 summary: mean compF1 / edgeF1 / ged, rep status, mean node/edge counts")
print("=" * 90)
for arm in arms:
    print(f"\n### {arm}")
    print(f"{'prompt':<8}{'cF1':>6}{'eF1':>6}{'ged':>7}{'scored':>7}"
          f"{'bad':>5}{'gNodes':>7}{'pNodes':>7}{'gEdge':>6}{'pEdge':>6}  top missing / extra")
    arm_data = RES[arm]
    for pid in L4_IDS:
        reps = arm_data.get(pid, [])
        scored = [r for r in reps if r.get("status") == "scored"]
        bad = [r for r in reps if r.get("status") != "scored"]
        gc, ge = gold_counts(gold[pid])
        if not scored:
            statuses = Counter(r.get("status") for r in reps)
            print(f"{pid:<8}{'--':>6}{'--':>6}{'--':>7}{0:>7}{len(bad):>5}"
                  f"{sum(gc.values()):>7}{'--':>7}{ge:>6}{'--':>6}  "
                  f"NO SCORED REPS {dict(statuses)}")
            continue
        cf1 = sum(r["compF1"] for r in scored) / len(scored)
        ef1 = sum(r["edgeF1"] for r in scored) / len(scored)
        ged = sum(r["ged"] for r in scored) / len(scored)
        pn = sum(sum(rep_counts(r)[0].values()) for r in scored) / len(scored)
        pe = sum(rep_counts(r)[1] for r in scored) / len(scored)
        # aggregate missing/extra across scored reps (mean per rep)
        miss_tot, extra_tot = Counter(), Counter()
        for r in scored:
            pc, _ = rep_counts(r)
            m, x = multiset_diff(gc, pc)
            miss_tot += m
            extra_tot += x
        n = len(scored)
        miss_s = ", ".join(f"-{t}×{v/n:.1f}" for t, v in miss_tot.most_common(3))
        extra_s = ", ".join(f"+{t}×{v/n:.1f}" for t, v in extra_tot.most_common(3))
        sig = " | ".join(s for s in (miss_s, extra_s) if s)
        print(f"{pid:<8}{cf1:>6.2f}{ef1:>6.2f}{ged:>7.1f}{n:>7}{len(bad):>5}"
              f"{sum(gc.values()):>7}{pn:>7.0f}{ge:>6}{pe:>6.0f}  {sig}")

# focused contrast: baseline vs base_agentic vs full on L4 compF1 per prompt
print("\n" + "=" * 90)
print("L4 per-prompt compF1: baseline vs agentic arms (mean over scored reps)")
print("=" * 90)
show = ["baseline", "base_agentic", "full", "full_minus_gate", "full_minus_critic"]
print(f"{'prompt':<8}" + "".join(f"{a[:14]:>16}" for a in show))
for pid in L4_IDS:
    row = f"{pid:<8}"
    for a in show:
        reps = [r for r in RES[a].get(pid, []) if r.get("status") == "scored"]
        if reps:
            v = sum(r["compF1"] for r in reps) / len(reps)
            row += f"{v:>16.2f}"
        else:
            row += f"{'--':>16}"
    print(row)

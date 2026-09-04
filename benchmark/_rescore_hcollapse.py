"""SENSITIVITY re-score: apply a hierarchical-collapse accept rule to the stored
correctness results and recompute COMPONENT F1 only.

Motivation: the L3 compF1 collapse ([[e2-o1-correctness-3cond-run-2026-07-09]]) is driven by an
abstraction-granularity mismatch — the b3_gold encodes an MZI as a single atomic node
(MZI_2x2 / MZI_1x1), and the rigid baseline picks that same atomic template, but the agentic
builder emits the physically-correct PRIMITIVE decomposition (two 2x2 couplers + the phase
shifters/heaters the prompt named). The scorer's 1-to-1 accept map (b3_eval._apply_accept)
cannot recognise a {2 couplers + phase element} subgraph as an MZI, so a correct decomposition
scores compF1=0. This script quantifies how much of the deficit that artifact accounts for.

STRONG CAVEATS (this is a sensitivity estimate, NOT a final metric):
  1. NODE-MULTISET ONLY. We only persisted pred node classes, not edges, so the collapse cannot
     VERIFY that the couplers+phase actually wire into an MZI (they might not). edgeF1/GED are
     NOT recomputed here — only compF1.
  2. NON-COMPREHENSIVE. It handles the two MZI decompositions observed in this run
     (MMI_2x2 x2 + phase -> MZI_2x2; DC_2x2 x2 + straight arms -> MZI_1x1). It does NOT cover
     other hierarchical equivalences (ring-based filters, lattice/CROW, cascaded MZIs, splitter
     trees, ...) and can OVER-collapse (e.g. two genuinely independent DCs).
  3. Therefore the final correctness numbers still REQUIRE a human evaluator to adjudicate — this
     only bounds the artifact's contribution.
"""
import json
import os
from collections import Counter

from b3_eval import load_gold
from topology_eval import prf

MODEL = os.getenv("E2_MODEL", "o1")
IN = os.getenv("E2_CORRECTNESS_OUT", "results/_correctness_o1_3cond.json")

meta, GOLD = load_gold()
CM = meta["class_map"]

COUPLER_2x2 = ("MMI_2x2", "DC_2x2")
PHASE = ("HEATER", "_ELECTRICAL", "PIN", "PHASE_SHIFTER")
ARM = ("STRAIGHT",)


def collapse(node_classes):
    """Collapse observed primitive-MZI subgraphs into atomic MZI nodes (multiset heuristic)."""
    c = Counter(node_classes)
    # (a) active 2x2 MZI: two MMI_2x2 couplers + >=1 phase element (one/arm, consume up to 2)
    while c["MMI_2x2"] >= 2 and sum(c[p] for p in PHASE) >= 1:
        c["MMI_2x2"] -= 2
        rem = 2
        for p in PHASE:
            take = min(c[p], rem)
            c[p] -= take
            rem -= take
            if rem == 0:
                break
        c["MZI_2x2"] += 1
    # (b) passive/asym 1x1 MZI: two DC_2x2 couplers + straight arm(s)
    while c["DC_2x2"] >= 2 and sum(c[a] for a in ARM) >= 1:
        c["DC_2x2"] -= 2
        rem = 2
        for a in ARM:
            take = min(c[a], rem)
            c[a] -= take
            rem -= take
            if rem == 0:
                break
        c["MZI_1x1"] += 1
    return +c  # drop zero counts


def comp_f1(pred_classes, gid):
    """Component multiset F1 (matches topology_eval._multiset_prf + score_topology)."""
    g = GOLD[gid]
    gold = Counter(g["nodes"].values())
    pred = Counter(pred_classes)
    tp = sum((pred & gold).values())
    return round(prf(tp, sum((pred - gold).values()), sum((gold - pred).values()))[2], 3)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


out = json.load(open(IN))
prompts = {p["id"]: p for p in json.load(open("e2_prompts.json"))["prompts"]}
ARMS = ["baseline", "agentic", "agentic_nogate"]


def prompt_pair(arm, pid):
    """(orig compF1 mean, collapsed compF1 mean) over scored reps for one prompt."""
    reps = [r for r in out[arm].get(pid, []) if r["status"] == "scored"]
    if not reps:
        return None, None
    orig = mean([r.get("compF1") for r in reps])
    coll = mean([comp_f1(collapse(r.get("pred_nodes") or []), pid) for r in reps])
    return orig, coll


lines = [f"# E2 correctness — hierarchical-collapse SENSITIVITY re-score (compF1 only, model={MODEL})", ""]
lines.append("Component F1 recomputed after collapsing observed primitive-MZI subgraphs "
             "(`MMI_2x2×2 + phase → MZI_2x2`; `DC_2x2×2 + straight → MZI_1x1`). "
             "**Node-multiset heuristic, edges NOT verified; non-comprehensive; a human evaluator "
             "must adjudicate the final correctness numbers.** edgeF1/GED not recomputed (edges "
             "were not persisted).")
lines.append("")

# Overall paired headline (all levels), agentic arms vs baseline, orig vs collapsed
for arm in ("agentic", "agentic_nogate"):
    paired = [pid for pid in prompts
              if prompt_pair("baseline", pid)[0] is not None
              and prompt_pair(arm, pid)[0] is not None]
    b_o = mean([prompt_pair("baseline", pid)[0] for pid in paired])
    b_c = mean([prompt_pair("baseline", pid)[1] for pid in paired])
    a_o = mean([prompt_pair(arm, pid)[0] for pid in paired])
    a_c = mean([prompt_pair(arm, pid)[1] for pid in paired])
    lines.append(f"## {arm} vs baseline — compF1 (paired n={len(paired)})")
    lines.append("| | baseline | " + arm + " | Δ(arm-base) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| original | {b_o} | {a_o} | {round(a_o - b_o, 3)} |")
    lines.append(f"| collapsed | {b_c} | {a_c} | {round(a_c - b_c, 3)} |")
    lines.append("")
    lines.append(f"### {arm}: compF1 by level (paper Table 1 / component-count; original → collapsed)")
    lines.append("| level | baseline | " + arm + " |")
    lines.append("|---|---|---|")
    for L in (1, 2, 3, 4):
        ks = [pid for pid in paired if GOLD[pid]["level_paper"] == L]
        bo = mean([prompt_pair("baseline", pid)[0] for pid in ks])
        bc = mean([prompt_pair("baseline", pid)[1] for pid in ks])
        ao = mean([prompt_pair(arm, pid)[0] for pid in ks])
        ac = mean([prompt_pair(arm, pid)[1] for pid in ks])
        lines.append(f"| L{L} (n={len(ks)}) | {bo}→{bc} | {ao}→{ac} |")
    lines.append("")

# Per-prompt L3 detail
lines.append("## L3 per-prompt compF1 (original → collapsed)")
lines.append("| prompt | gold | baseline | agentic | agentic_nogate |")
lines.append("|---|---|---|---|---|")
for pid in [f"L3_{i}" for i in range(1, 7)]:
    gold = sorted(GOLD[pid]["nodes"].values())
    cells = [f"`{'+'.join(gold)}`"]
    for arm in ARMS:
        o, c = prompt_pair(arm, pid)
        cells.append(f"{o}→{c}" if o is not None else "—")
    lines.append("| " + pid + " | " + " | ".join(cells) + " |")
lines.append("")
lines.append("_Caveat: node-multiset heuristic, edges unverified, non-comprehensive — final "
             "correctness requires human oversight._")

path = IN.rsplit(".", 1)[0] + "_hcollapse.md"
open(path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nreport -> {path}")

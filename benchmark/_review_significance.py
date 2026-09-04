"""Paired significance tests on the BLIND human 3-axis review (A1/A2/A3).

Design: same 24 prompts x 3 reps per arm; arms are paired on prompt id (n=24). For each
(arm, pid) we take the mean axis score over that prompt's reps, giving paired length-24
vectors. Two-sided paired Wilcoxon signed-rank per axis; Holm-Bonferroni within each
comparison family. Axis scalars: A1/A2 {correct:1, minor/partial:.5, wrong:0};
A3 {yes:1, partial:.5, no:0}.

Families:
  baseline-vs-arm : each agentic arm vs the rigid baseline (does agentic help?)
  additive        : base_plus_{kg,gate,critic} vs base_agentic  (effect of ADDING a feature)
  LOO             : full_minus_{kg,gate,critic} vs full          (effect of REMOVING a feature)

mean_d = mean(arm - ref) over prompts; w/l = #prompts arm>ref / arm<ref.

Run:  .venv/bin/python benchmark/_review_significance.py [LABELS.json]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
LABELS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE / "results" / "_ablation_correctness_gpt54_v2_review3_labels.json"
OUT = HERE / "results" / (LABELS.stem.replace("_review3_labels", "") + "_significance.md")
SCALAR = {"correct": 1.0, "yes": 1.0, "minor": 0.5, "partial": 0.5, "wrong": 0.0, "no": 0.0}
AXES = ["A1", "A2", "A3"]
ALPHA = 0.05

ann = (lambda j: j.get("annotations", j))(json.loads(LABELS.read_text()))
# axis -> arm -> pid -> [per-rep scalars]
data = {ax: defaultdict(lambda: defaultdict(list)) for ax in AXES}
for rid, rec in ann.items():
    arm, pid, _ = rid.split("::")
    axs = rec.get("axes") or {}
    for ax in AXES:
        if axs.get(ax) in SCALAR:
            data[ax][arm][pid].append(SCALAR[axs[ax]])


def pmeans(ax, arm):
    return {pid: st.mean(v) for pid, v in data[ax][arm].items() if v}


def compare(ax, arm, ref):
    A, B = pmeans(ax, arm), pmeans(ax, ref)
    pids = sorted(set(A) & set(B))
    a = [A[p] for p in pids]
    b = [B[p] for p in pids]
    diffs = [x - y for x, y in zip(a, b)]
    wins = sum(1 for x in diffs if x > 0)
    losses = sum(1 for x in diffs if x < 0)
    nz = wins + losses
    mean_d = st.mean(diffs) if diffs else float("nan")
    if nz < 1:
        p = 1.0
    else:
        try:
            _, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p = 1.0
    return {"n": len(pids), "mean_d": mean_d, "w": wins, "l": losses, "p": float(p)}


def holm(pairs):
    """pairs: list of (key, p). Returns {key: (p, p_holm, reject)}."""
    m = len(pairs)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(sorted(pairs, key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = (p, running, running < ALPHA)
    return out


arms = sorted({rid.split("::")[0] for rid in ann})
# The rigid arm is "baseline" (gpt-5.4 run) or "rigid_baseline" (Qwen/Nemotron runs).
RIGID = "rigid_baseline" if "rigid_baseline" in arms else "baseline"
# Only include a family if its arms are actually present (Qwen has no LOO full_minus_* arms).
FAMILIES = {f"baseline-vs-arm (arm - {RIGID})": [(a, RIGID) for a in arms if a != RIGID]}
add = [(f"base_plus_{f}", "base_agentic") for f in ("kg", "gate", "critic")
       if f"base_plus_{f}" in arms]
if add and "base_agentic" in arms:
    FAMILIES["additive (base_plus_X - base_agentic)"] = add
loo = [(f"full_minus_{f}", "full") for f in ("kg", "gate", "critic") if f"full_minus_{f}" in arms]
if loo and "full" in arms:
    FAMILIES["LOO (full_minus_X - full)"] = loo

L = ["# E2 ablation — human review significance (paired Wilcoxon, n=24 prompts)",
     "", f"Labels: `{LABELS.name}` · axis scalars A1/A2 correct=1/partial=.5/wrong=0, "
     "A3 yes=1/partial=.5/no=0.", "Holm-Bonferroni within each family x axis. mean_d = mean(arm - ref).", ""]
for fam, comps in FAMILIES.items():
    L += [f"## {fam}", ""]
    for ax in AXES:
        res = {f"{arm}": compare(ax, arm, ref) for arm, ref in comps}
        hl = holm([(k, v["p"]) for k, v in res.items()])
        L += [f"### {ax}", "", "| comparison | n | mean_d | w/l | p_raw | p_holm | sig |",
              "|---|---|---|---|---|---|---|"]
        for arm, ref in comps:
            r = res[arm]
            p, ph, rej = hl[arm]
            L.append(f"| {arm} | {r['n']} | {r['mean_d']:+.3f} | {r['w']}/{r['l']} | "
                     f"{p:.4f} | {ph:.4f} | {'*' if rej else ''} |")
        L.append("")
OUT.write_text("\n".join(L))
print("\n".join(L))
print(f"\nwrote {OUT}")

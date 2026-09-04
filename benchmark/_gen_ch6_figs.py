"""Generate the Tier-1 Ch6 result figures (vector PDF + PNG preview) into thesis/figures/.

F1 ch6_human_gpt54  — 3-panel dot plot: E2 blind human scores per arm (9 arms), gpt-5.4.
F2 ch6_human_qwen   — same form/scale, 6 arms, Qwen3.6-27B (cross-model comparability).
F3 ch6_fault_tags   — paired horizontal bars: fault-tag rates per 72 runs, baseline vs agentic.
F4 ch6_stage_reach  — line chart: % of runs reaching each pipeline stage, by graded outcome.

All values computed from the committed labels/trace files (no hand-typed numbers); the script
prints each figure's data so it can be checked against the appendix tables. Visual style:
_thesis_fig_style (the Ch4 MATLAB idiom — validated palette, box axes, black ink).
"""
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import _thesis_fig_style as S
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/home/tony/PhIDOv1/PhIDO-Release/thesis/figures")
S.use_style()

SCALAR = {"correct": 1.0, "yes": 1.0, "minor": 0.5, "partial": 0.5, "wrong": 0.0, "no": 0.0}
AXES = ["A1", "A2", "A3"]
PANEL = {"A1": "Component (A1)", "A2": "Connectivity (A2)", "A3": "Intent (A3)"}


def arm_means(labels_path):
    ann = json.load(open(labels_path))["annotations"]
    per = defaultdict(lambda: defaultdict(list))
    for rid, a in ann.items():
        arm = rid.split("::")[0]
        for ax in AXES:
            v = a.get("axes", {}).get(ax)
            if v in SCALAR:
                per[arm][ax].append(SCALAR[v])
    return {arm: {ax: st.mean(v) for ax, v in d.items()} for arm, d in per.items()}, ann


def human_dotplot(means, arms, rigid_key, fname, height):
    """arms: list of (key, display) top-to-bottom; rigid drawn as orange square + ref line."""
    fig, axs = plt.subplots(1, 3, figsize=(5.9, height), sharey=True)
    ys = list(range(len(arms) - 1, -1, -1))
    for k, ax_id in enumerate(AXES):
        ax = axs[k]
        rigid_val = means[rigid_key][ax_id]
        ax.axvline(rigid_val, color=S.REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for y, (key, _) in zip(ys, arms):
            v = means[key][ax_id]
            if key == rigid_key:
                ax.scatter([v], [y], s=46, color=S.ORANGE, marker="s",
                           edgecolor=S.INK, lw=0.5, zorder=3)
            else:
                ax.scatter([v], [y], s=42, color=S.BLUE,
                           edgecolor=S.INK, lw=0.5, zorder=3)
        ax.set_title(PANEL[ax_id], fontsize=9.5)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "", "0.5", "", "1"])
        ax.set_ylim(-0.7, len(arms) - 0.3)
        S.box(ax)
    axs[0].set_yticks(ys, [d for _, d in arms])
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=S.ORANGE, marker="s", ls="", ms=7,
               markeredgecolor=S.INK, markeredgewidth=0.5, label="rigid pipeline"),
        Line2D([], [], color=S.BLUE, marker="o", ls="", ms=7,
               markeredgecolor=S.INK, markeredgewidth=0.5, label="agentic configurations"),
        Line2D([], [], color=S.REF, ls=(0, (4, 3)), label="baseline reference"),
    ], loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


# ================= F1: gpt-5.4 human =================
G_ARMS = [("baseline", "Rigid baseline"), ("base_agentic", "Minimal agentic"),
          ("base_plus_kg", "+ grounding"), ("base_plus_gate", "+ gate"),
          ("base_plus_critic", "+ critic"), ("full", "Full"),
          ("full_minus_kg", "− grounding"), ("full_minus_gate", "− gate"),
          ("full_minus_critic", "− critic")]
gm, gann = arm_means(ROOT / "benchmark/results/_ablation_correctness_gpt54_v2_review3_labels_ADJ.json")
print("F1 gpt-5.4 means (check vs Table e2-human):")
for k, d in G_ARMS:
    print(f"  {d:16}", {ax: round(gm[k][ax], 2) for ax in AXES})
human_dotplot(gm, G_ARMS, "baseline", "ch6_human_gpt54", 2.9)

# ================= F2: Qwen human =================
Q_ARMS = [("rigid_baseline", "Rigid baseline"), ("base_agentic", "Minimal agentic"),
          ("base_plus_kg", "+ grounding"), ("base_plus_gate", "+ gate"),
          ("base_plus_critic", "+ critic"), ("full", "Full")]
qm, qann = arm_means(ROOT / "benchmark/results/_qwen_review_input_review3_labels.json")
print("F2 Qwen means (check vs Table qwen-human):")
for k, d in Q_ARMS:
    print(f"  {d:16}", {ax: round(qm[k][ax], 2) for ax in AXES})
human_dotplot(qm, Q_ARMS, "rigid_baseline", "ch6_human_qwen", 2.25)

# ================= F3: fault tags, baseline vs agentic per-72 =================
TAGS = [("Connectivity", ["missing-edge", "disconnected", "port-mismatch", "wrong-edge", "wrong-class"]),
        ("Component", ["wrong-count", "missing", "wrong-type", "spurious"]),
        ("Intent", ["param-specified-unverifiable"])]
base_ct, ag_ct, ag_runs = Counter(), Counter(), set()
for rid, a in gann.items():
    arm = rid.split("::")[0]
    tgt = base_ct if arm == "baseline" else ag_ct
    if arm != "baseline":
        ag_runs.add(rid)
    for ax, tags in (a.get("tags") or {}).items():
        for t in tags or []:
            tgt[t] += 1
n_ag72 = len(ag_runs) / 72.0
rows = [(t, base_ct[t], ag_ct[t] / n_ag72) for _, ts in TAGS for t in ts]
print("F3 tag rates per 72 (check vs Table e2-tags): agentic runs =", len(ag_runs))
for t, b, a72 in rows:
    print(f"  {t:28} baseline {b:>3}  agentic/72 {a72:.1f}")

fig, ax = plt.subplots(figsize=(5.7, 3.4))
labels = [r[0] for r in rows]
ys, gap, y = [], 0.9, 0.0
for gi, (_, ts) in enumerate(TAGS):
    for _t in ts:
        ys.append(y); y -= 1.0
    y -= gap
H = 0.36
ax.barh([yy + 0.20 for yy in ys], [r[2] for r in rows], height=H, color=S.BLUE,
        edgecolor=S.INK, lw=0.4, label="agentic family (rate per 72 runs)")
ax.barh([yy - 0.20 for yy in ys], [r[1] for r in rows], height=H, color=S.ORANGE,
        hatch="////", edgecolor=S.ORANGE_DK, lw=0.4, label="rigid pipeline (72 runs)")
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontfamily="monospace", fontsize=8)
# group headers: fully inside the axes (clear of the left spine and the box top)
for (gname, ts), y0 in zip(TAGS, [ys[0], ys[5], ys[9]]):
    ax.text(0.35, y0 + 0.55, gname, fontsize=8.5, color=S.INK, fontweight="bold",
            ha="left", va="bottom")
ax.set_xlabel("fault tags per 72 runs")
ax.set_ylim(min(ys) - 0.75, max(ys) + 1.4)
S.box(ax, ticks_all_sides=False)
ax.tick_params(left=False)
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_fault_tags.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_fault_tags")

# F4 below is SUPERSEDED by the two-panel (Qwen + Nemotron) version in
# _gen_nemotron_trace_figs.py, which writes the same filename. Exit here so
# re-running this script cannot clobber it.
import sys
sys.exit(0)

# ================= F4: stage reach by outcome (Qwen traces) =================
STAGES = [("ExtractedConcepts", "Concepts"), ("RequirementManifest", "Requirements"),
          ("DesignIntent", "Design intent"), ("ComponentSelectionLLM", "Selection"),
          ("ComplianceVerdict", "Compliance"), ("CriticVerdict", "Critic")]
qlab = json.load(open(ROOT / "benchmark/results/_qwen_review_input_review3_labels.json"))["annotations"]
buckets = {"SUCCESS": [], "PARTIAL": [], "FAIL": []}
for tf in glob.glob(str(ROOT / "benchmark/results/qwen_traces/*.json")):
    d = json.load(open(tf))
    a2 = qlab.get(f"{d['arm']}::{d['prompt_id']}::rep{d['rep']}", {}).get("axes", {}).get("A2")
    b = {"correct": "SUCCESS", "partial": "PARTIAL", "wrong": "FAIL"}.get(a2)
    if b:
        buckets[b].append({s.get("schema") for s in d["trace"] if s.get("kind") == "structured"})
reach = {b: [100 * st.mean(sch in run for run in runs) for sch, _ in STAGES]
         for b, runs in buckets.items()}
print("F4 stage reach % (check vs Table trace-reach):")
for b in ("SUCCESS", "PARTIAL", "FAIL"):
    print(f"  {b:8} n={len(buckets[b]):3}", [round(v) for v in reach[b]])

fig, ax = plt.subplots(figsize=(5.6, 2.7))
x = range(len(STAGES))
for b, col, mk, lw in [("SUCCESS", S.BLUE, "o", 1.6), ("PARTIAL", S.PURPLE, "^", 1.1),
                       ("FAIL", S.ORANGE, "s", 1.6)]:
    ax.plot(list(x), reach[b], color=col, marker=mk, ms=5.5, lw=lw,
            markeredgecolor=S.INK, markeredgewidth=0.4,
            label=f"{b.lower()} (n={len(buckets[b])})")
ax.set_xticks(list(x), [lbl for _, lbl in STAGES], fontsize=8.5)
ax.set_ylabel("% of runs reaching stage")
ax.set_ylim(60, 104)
S.box(ax)
ax.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_stage_reach.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_stage_reach")

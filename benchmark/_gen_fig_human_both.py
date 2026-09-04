"""ch6_human_both — the two E2 blind-human-score dot plots (gpt-5.4, Qwen3.6-27B) stacked
into one figure with a shared scale and a single legend, for the merged §6.5 results.
Replaces ch6_human_gpt54 + ch6_human_qwen; also renames the legend entry to
"agentic configurations" (thesis-wide terminology change 2026-08-17).
Values computed from the committed label files; printed for checking against the tables.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import _thesis_fig_style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    return {arm: {ax: st.mean(v) for ax, v in d.items()} for arm, d in per.items()}


G_ARMS = [("baseline", "Rigid pipeline"), ("base_agentic", "Minimal agentic"),
          ("base_plus_kg", "+ grounding"), ("base_plus_gate", "+ gate"),
          ("base_plus_critic", "+ critic"), ("full", "Full"),
          ("full_minus_kg", "− grounding"), ("full_minus_gate", "− gate"),
          ("full_minus_critic", "− critic")]
Q_ARMS = [("rigid_baseline", "Rigid pipeline"), ("base_agentic", "Minimal agentic"),
          ("base_plus_kg", "+ grounding"), ("base_plus_gate", "+ gate"),
          ("base_plus_critic", "+ critic"), ("full", "Full")]

gm = arm_means(ROOT / "benchmark/results/_ablation_correctness_gpt54_v2_review3_labels_ADJ.json")
qm = arm_means(ROOT / "benchmark/results/_qwen_review_input_review3_labels.json")
nm = arm_means(ROOT / "benchmark/results/_nemotron_review_input_review3_labels.json")
for name, means, arms in (("gpt-5.4", gm, G_ARMS), ("Qwen3.6-27B", qm, Q_ARMS), ("Nemotron-3-Ultra-550B", nm, Q_ARMS)):
    print(f"{name} means:")
    for k, d in arms:
        print(f"  {d:16}", {ax: round(means[k][ax], 2) for ax in AXES})

fig = plt.figure(figsize=(5.9, 7.4))
gs = fig.add_gridspec(3, 3, height_ratios=[9, 6, 6], hspace=0.55, wspace=0.08)

for row, (means, arms, rigid_key, rowtitle) in enumerate([
        (gm, G_ARMS, "baseline", "(a) gpt-5.4"),
        (qm, Q_ARMS, "rigid_baseline", "(b) Qwen3.6-27B"),
        (nm, Q_ARMS, "rigid_baseline", "(c) Nemotron-3-Ultra-550B")]):
    ys = list(range(len(arms) - 1, -1, -1))
    axs = [fig.add_subplot(gs[row, k]) for k in range(3)]
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
        if row == 0:
            ax.set_title(PANEL[ax_id], fontsize=9.5)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "", "0.5", "", "1"])
        ax.set_ylim(-0.7, len(arms) - 0.3)
        if k > 0:
            ax.set_yticks([])
        S.box(ax)
    axs[0].set_yticks(ys, [d for _, d in arms])
    axs[1].text(0.5, 1.02 if row else 1.14, rowtitle, transform=axs[1].transAxes,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.legend(handles=[
    Line2D([], [], color=S.ORANGE, marker="s", ls="", ms=7,
           markeredgecolor=S.INK, markeredgewidth=0.5, label="rigid pipeline"),
    Line2D([], [], color=S.BLUE, marker="o", ls="", ms=7,
           markeredgecolor=S.INK, markeredgewidth=0.5, label="agentic configurations"),
    Line2D([], [], color=S.REF, ls=(0, (4, 3)), label="baseline reference"),
], loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_human_both.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_human_both")

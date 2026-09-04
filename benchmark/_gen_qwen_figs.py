"""Generate the two Ch6 Qwen figures (vector PDF + PNG preview) into thesis/figures/.

Fig 1  qwen_inversion      — dumbbell of per-transition mean edge-F1 deltas, Qwen vs gpt-5.4
                             (values = benchmark/_score_qwen_n24.py output, failures scored 0).
Fig 2  qwen_tool_mix       — paired horizontal bars of tool-call shares, SUCCESS vs FAIL runs
                             (recomputed from qwen_traces/ joined to the human A2 labels).

Palette: validated categorical slots 1-2 (blue #2a78d6 / orange #eb6834), shape + hatch as
secondary encoding so both figures survive grayscale print.
"""
import glob
import json
from collections import Counter
from pathlib import Path

import _thesis_fig_style as S
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/home/tony/PhIDOv1/PhIDO-Release/thesis/figures")
S.use_style()

# Ch4 MATLAB palette via the shared style module (validated: see _thesis_fig_style)
BLUE, ORANGE, ORANGE_DK, INK = S.BLUE, S.ORANGE, S.ORANGE_DK, S.INK
INK2, MUTED, BASE = S.INK, S.REF, "#bbbbbb"


def style(ax):
    S.box(ax, ticks_all_sides=False)


# ---------------- Fig 1: inversion dumbbell ----------------
# mean edge-F1 delta per transition (failures scored 0), from _score_qwen_n24.py
ROWS = [  # (label, qwen, gpt54) — listed top-to-bottom
    ("rigid $\\rightarrow$ minimal agentic", 0.243, -0.000),
    ("+ knowledge graph",                 0.087,  0.035),
    ("+ gate",                            0.012, -0.023),
    ("+ critic",                          0.051, -0.048),
    ("full pipeline",                    -0.002, -0.040),
]
fig, ax = plt.subplots(figsize=(5.6, 2.55))
ys = range(len(ROWS) - 1, -1, -1)  # first row on top
ax.scatter([r[1] for r in ROWS], list(ys), s=52, color=BLUE, zorder=3,
           edgecolor=INK, lw=0.5, label="Qwen3.6-27B")
ax.scatter([r[2] for r in ROWS], list(ys), s=46, color=ORANGE, marker="s", zorder=3,
           edgecolor=INK, lw=0.5, label="gpt-5.4")
ax.axvline(0, color=INK, lw=0.8, zorder=2)
ax.set_yticks(list(ys), [r[0] for r in ROWS])
ax.set_xlabel("mean edge-$F_1$ effect of the transition")
ax.set_xlim(-0.09, 0.27)
style(ax)
ax.legend(loc="lower right", fontsize=8.5, handletextpad=0.4)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"qwen_inversion.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote qwen_inversion")

# The single-panel tool-mix figure below is SUPERSEDED by the two-panel
# (Qwen + Nemotron) version in _gen_nemotron_trace_figs.py, which writes the
# same filename. Exit here so re-running this script cannot clobber it.
import sys
sys.exit(0)

# ---------------- Fig 2: tool mix, SUCCESS vs FAIL ----------------
LAB = json.load(open(ROOT / "benchmark/results/_qwen_review_input_review3_labels.json"))["annotations"]
succ, fail = Counter(), Counter()
for tf in glob.glob(str(ROOT / "benchmark/results/qwen_traces/*.json")):
    d = json.load(open(tf))
    a2 = LAB.get(f"{d['arm']}::{d['prompt_id']}::rep{d['rep']}", {}).get("axes", {}).get("A2")
    tgt = succ if a2 == "correct" else (fail if a2 == "wrong" else None)
    if tgt is None:
        continue
    for s in d["trace"]:
        for tc in s.get("tool_calls", []):
            tgt[tc["name"]] += 1
s_tot, f_tot = sum(succ.values()), sum(fail.values())
# fold the long tail (each < 1.5% in both buckets, incl. stray hallucinated tool names)
# into a single "other" row rather than 7 near-zero bars
main = [t for t in sorted(set(succ) | set(fail), key=lambda t: -succ[t])
        if 100 * succ[t] / s_tot >= 1.5 or 100 * fail[t] / f_tot >= 1.5]
tools = main + ["other"]
s_pct = [100 * succ[t] / s_tot for t in main] + [100 * sum(v for k, v in succ.items() if k not in main) / s_tot]
f_pct = [100 * fail[t] / f_tot for t in main] + [100 * sum(v for k, v in fail.items() if k not in main) / f_tot]
print("success shares:", {t: round(p, 1) for t, p in zip(tools, s_pct)})
print("fail shares:   ", {t: round(p, 1) for t, p in zip(tools, f_pct)})

fig, ax = plt.subplots(figsize=(5.6, 3.5))
ys = range(len(tools) - 1, -1, -1)
H = 0.36
ax.barh([y + 0.20 for y in ys], s_pct, height=H, color=BLUE, edgecolor=INK, lw=0.4,
        label="graded correct (success)")
ax.barh([y - 0.20 for y in ys], f_pct, height=H, color=ORANGE, hatch="////",
        edgecolor=ORANGE_DK, lw=0.4, label="graded wrong (failure)")
# selective direct labels on the three tools the text discusses
KEY = {"search_knowledge_graph", "validate_ports", "get_component_properties"}
for y, t, sp, fp in zip(ys, tools, s_pct, f_pct):
    if t in KEY:
        ax.text(sp + 0.4, y + 0.20, f"{sp:.0f}%", va="center", fontsize=7.5, color=INK2)
        ax.text(fp + 0.4, y - 0.20, f"{fp:.0f}%", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(list(ys), [t.replace("_", "\\_") for t in tools], fontfamily="monospace", fontsize=8)
ax.set_yticks(list(ys))
ax.set_yticklabels(tools, fontfamily="monospace", fontsize=8)
ax.set_xlabel("share of the run's tool calls (%)")
style(ax)
ax.tick_params(left=False)
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"qwen_tool_mix.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote qwen_tool_mix")

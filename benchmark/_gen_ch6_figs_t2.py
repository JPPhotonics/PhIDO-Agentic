"""Generate the Tier-2 Ch6 result figures (vector PDF + PNG preview) into thesis/figures/.

F5 ch6_e1_retrieval — two panels, one metric each: testbench pass@3 by query type
                      (6 configurations, dot plot) and curated-intent coverage@3 (same 6; the
                      enrichment-null visible). coverage@3 = set-valued pass@3.
F6 ch6_a2_faithfulness — A2 faithfulness by relationship type with Wilson 95% intervals.
F7 ch6_a1_sweep     — A1 near-duplicate counts vs cosine threshold (Component, Property,
                      other labels, total).
F8 ch6_qwen_levels  — Qwen edge-F1 per arm, L3 vs L4 dumbbells (failures scored 0).
F9 ch6_validation_scatter — validation calls vs success rate per agentic arm (recomputed
                      from the 360 traces joined to the human A2 labels).

Style: _thesis_fig_style (Ch4 MATLAB idiom). F9 is recomputed from committed data; F5-F8
carry the values of the appendix data tables (the committed record), stated in comments.
"""
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import _thesis_fig_style as S
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/home/tony/PhIDOv1/PhIDO-Release/thesis/figures")
S.use_style()
CONN = "#bbbbbb"  # dumbbell connectors

# ================= F5: E1 retrieval, two panels =================
# values = results/e1_full_grid.json (testbench: 6 configurations incl. enrichment; curated:
# 4 single-signal configurations — hybrids deliberately NOT shown there, see §6.4).
TB = [  # (configuration, named p@3, functional p@3) — 231 testbench queries
    ("Lexical",              0.56, 0.40),
    ("KG (functional)",      0.40, 0.20),
    ("KG + enrichment",      0.41, 0.20),
    ("KG (embedding)",       0.52, 0.16),
    ("Hybrid (embedding)",   0.72, 0.28),
    ("Hybrid (production)",  0.78, 0.52),
]
CUR = [  # (arm, coverage@3) — 18 curated design-intent queries; coverage@3 is the
    # set-valued counterpart of pass@3 (identical for the testbench's single-cell gold)
    ("Lexical",         0.44),
    ("KG (functional)",   0.32),
    ("KG + enrichment",   0.32),
    ("KG (embedding)",    0.68),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.9, 3.0))
ys = list(range(len(TB) - 1, -1, -1))
a1.scatter([r[1] for r in TB], ys, s=44, color=S.BLUE, edgecolor=S.INK, lw=0.5,
           zorder=3, label="named")
a1.scatter([r[2] for r in TB], ys, s=40, color=S.ORANGE, marker="s", edgecolor=S.INK,
           lw=0.5, zorder=3, label="functional")
a1.set_yticks(ys, [r[0] for r in TB], fontsize=8)
a1.set_xlabel("pass@3")
a1.set_title("Testbench (231 queries)", fontsize=9)
a1.set_xlim(0, 1.0)
a1.set_ylim(-0.6, len(TB) - 0.4)
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
          fontsize=8, handletextpad=0.3, columnspacing=1.2)
ys2 = list(range(len(CUR) - 1, -1, -1))
a2.scatter([r[1] for r in CUR], ys2, s=44, color=S.BLUE, edgecolor=S.INK, lw=0.5,
           zorder=3)
a2.set_yticks(ys2, [r[0] for r in CUR], fontsize=8)
a2.set_xlabel("coverage@3")
a2.set_title("Curated intents (18 queries)", fontsize=9)
a2.set_xlim(0, 1.0)
a2.set_ylim(-0.6, len(CUR) - 0.4)
for ax in (a1, a2):
    S.box(ax)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_e1_retrieval.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_e1_retrieval")

# ================= F6: A2 faithfulness by type, Wilson 95% =================
# counts reconstruct the appendix table tab:a2-relation (fraction, n): 140/195 overall
A2 = [("HAS_PROPERTY", 55, 69), ("BASED_ON_PRINCIPLE", 22, 29), ("RELATED_TO", 26, 37),
      ("CONTAINS_COMPONENT", 9, 14), ("PERFORMS_FUNCTION", 22, 36), ("USES_COMPONENT", 6, 10)]
assert sum(k for _, k, _ in A2) == 140 and sum(n for _, _, n in A2) == 195


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


print("F6 faithfulness + Wilson CI (check vs tab:a2-relation):")
fig, ax = plt.subplots(figsize=(5.4, 2.3))
ys = list(range(len(A2) - 1, -1, -1))
ax.axvline(140 / 195, color=S.REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
for y, (name, k, n) in zip(ys, A2):
    p, lo, hi = wilson(k, n)
    print(f"  {name:20} {p:.2f} [{lo:.2f},{hi:.2f}] n={n}")
    ax.plot([lo, hi], [y, y], color=S.INK, lw=1.0, zorder=2)
    ax.plot([lo, lo], [y - 0.14, y + 0.14], color=S.INK, lw=1.0, zorder=2)
    ax.plot([hi, hi], [y - 0.14, y + 0.14], color=S.INK, lw=1.0, zorder=2)
    ax.scatter([p], [y], s=44, color=S.BLUE, edgecolor=S.INK, lw=0.5, zorder=3)
ax.set_yticks(ys, [f"{n} (n={c})" for n, _, c in [(a, b, c) for a, b, c in A2]],
              fontfamily="monospace", fontsize=8)
ax.set_xlabel("fraction of committed edges supported")
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-0.6, len(A2) - 0.4)
S.box(ax, ticks_all_sides=False)
ax.tick_params(left=False)
ax.text(140 / 195 + 0.010, -0.52, "overall 0.72", fontsize=7.5, color=S.REF,
        va="bottom")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_a2_faithfulness.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_a2_faithfulness")

# ================= F7: A1 near-duplicate sweep =================
# values = appendix table tab:a1-dup; "other labels" = Design_Function+Architecture+Phys_Principle
TH = [0.95, 0.90, 0.85, 0.80]
A1 = [("Component (111 nodes)", [0, 2, 10, 26], S.BLUE),
      ("Property (162 nodes)", [0, 0, 6, 17], S.ORANGE),
      ("Other labels (141 nodes)", [0, 2, 10, 11], S.YELLOW)]
TOTAL = [0, 4, 26, 54]
fig, ax = plt.subplots(figsize=(5.4, 2.6))
x = list(range(len(TH)))
bottom = [0] * len(TH)
for name, vals, col in A1:
    ax.bar(x, vals, bottom=bottom, width=0.55, color=col, edgecolor=S.INK, lw=0.6,
           label=name)
    bottom = [b + v for b, v in zip(bottom, vals)]
for xi, tot in zip(x, TOTAL):  # total = stack height, labeled
    ax.text(xi, tot + 1.4, str(tot), ha="center", fontsize=8.5, color=S.INK,
            fontweight="bold")
ax.set_xticks(x, [f"$\\geq {t:.2f}$" for t in TH])
ax.set_xlabel("cosine-similarity threshold")
ax.set_ylabel("nodes in near-duplicate clusters")
ax.set_ylim(0, 62)
S.box(ax)
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_a1_sweep.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_a1_sweep")

# ================= F8: Qwen edge-F1 by arm x level =================
# values = appendix table tab:qwen-auto (_score_qwen_n24.py, failures scored 0)
Q = [("Rigid pipeline", 0.438, 0.024), ("Minimal agentic", 0.585, 0.363),
     ("+ grounding", 0.756, 0.366), ("+ gate", 0.609, 0.362),
     ("+ critic", 0.660, 0.389), ("Full", 0.621, 0.323)]
fig, ax = plt.subplots(figsize=(5.4, 2.6))
ys = list(range(len(Q) - 1, -1, -1))
ax.scatter([r[1] for r in Q], ys, s=44, color=S.BLUE, edgecolor=S.INK, lw=0.5,
           zorder=3, label="Level 3")
ax.scatter([r[2] for r in Q], ys, s=40, color=S.ORANGE, marker="s", edgecolor=S.INK,
           lw=0.5, zorder=3, label="Level 4")
ax.set_yticks(ys, [r[0] for r in Q])
ax.set_xlabel("edge-$F_1$ (failures scored 0)")
ax.set_xlim(0, 0.85)
ax.set_ylim(-0.6, len(Q) - 0.4)
S.box(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=8.5,
          handletextpad=0.3, columnspacing=1.4)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_qwen_levels.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_qwen_levels")

# F9 below is SUPERSEDED by the two-panel (Qwen + Nemotron) version in
# _gen_nemotron_trace_figs.py, which writes the same filename. Exit here so
# re-running this script cannot clobber it.
import sys
sys.exit(0)

# ================= F9: validation calls vs success, per arm (recomputed) =================
VALIDATE = {"validate_ports", "get_component_properties", "get_component_info",
            "get_pdk_cell_details", "get_pdk_implementations", "get_module_params",
            "resolve_function"}
LAB = json.load(open(ROOT / "benchmark/results/_qwen_review_input_review3_labels.json"))["annotations"]
per = defaultdict(lambda: {"val": [], "succ": []})
for tf in glob.glob(str(ROOT / "benchmark/results/qwen_traces/*.json")):
    d = json.load(open(tf))
    a2 = LAB.get(f"{d['arm']}::{d['prompt_id']}::rep{d['rep']}", {}).get("axes", {}).get("A2")
    if a2 is None:
        continue
    tools = [tc["name"] for s in d["trace"] for tc in s.get("tool_calls", [])]
    per[d["arm"]]["val"].append(sum(t in VALIDATE for t in tools))
    per[d["arm"]]["succ"].append(1.0 if a2 == "correct" else 0.0)
NAMES = {"base_agentic": "Minimal agentic", "base_plus_kg": "+ grounding",
         "base_plus_gate": "+ gate", "base_plus_critic": "+ critic", "full": "Full"}
OFF = {"Minimal agentic": (0.25, -0.9), "+ grounding": (0.25, 0.6), "+ gate": (0.25, -0.4),
       "+ critic": (-0.35, 1.0), "Full": (-0.3, -1.6)}
print("F9 per-arm validation/success (check vs tab:trace-critic):")
fig, ax = plt.subplots(figsize=(4.4, 2.8))
for k, disp in NAMES.items():
    xv = st.mean(per[k]["val"])
    yv = 100 * st.mean(per[k]["succ"])
    print(f"  {disp:16} val={xv:.1f}  success={yv:.0f}%")
    ax.scatter([xv], [yv], s=52, color=S.BLUE, edgecolor=S.INK, lw=0.5, zorder=3)
    dx, dy = OFF[disp]
    ax.annotate(disp, (xv, yv), xytext=(xv + dx, yv + dy), fontsize=8, color=S.INK,
                ha="left" if dx > 0 else "right")
ax.set_xlabel("validation calls per run")
ax.set_ylabel("runs graded connectivity-correct (%)")
ax.set_xlim(6, 14.5)
ax.set_ylim(34, 50)
S.box(ax)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_validation_scatter.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_validation_scatter")

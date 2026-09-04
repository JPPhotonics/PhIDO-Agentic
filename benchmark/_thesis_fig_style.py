"""Shared matplotlib style for the thesis result figures.

Matches the Chapter 4 benchmark figures (MATLAB-produced, reproduced from the published
study) so the Chapter 6 charts read as the same family: MATLAB default color order,
full box axes with inward ticks, no gridlines, black ink, bold sans titles/labels
(Liberation Sans = Arial-metric). Validated (dataviz six-checks, white surface):
[#0072BD, #D95319] and [#0072BD, #D95319, #7E2F8E] pass all gates; the MATLAB yellow
and green fail the 3:1 contrast floor on white and are not used for lines/marks.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MATLAB default color order (the Ch4 palette)
BLUE = "#0072BD"     # slot 1  (Ch4: S)
ORANGE = "#D95319"   # slot 2  (Ch4: L)
YELLOW = "#EDB120"   # slot 3  (Ch4: PC) — sub-3:1 on white; fills only, never lines
PURPLE = "#7E2F8E"   # slot 4  (Ch4: SG) — validated third series
GREEN = "#77AC30"    # slot 5  (Ch4: CS) — sub-3:1 on white; fills only
LBLUE = "#4DBEEE"    # slot 6  (Ch4: EE)
ORANGE_DK = "#A63E12"  # tone-on-tone hatch ink for the orange series

INK = "#000000"
REF = "#666666"      # dashed reference lines


def use_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK, "axes.labelweight": "bold",
        "axes.titleweight": "bold", "axes.titlesize": 10,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def box(ax, ticks_all_sides=True):
    """Full MATLAB-style box: all four spines, inward ticks on all sides, no grid."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(INK)
    if ticks_all_sides:
        ax.tick_params(top=True, right=True, direction="in", length=3)
    else:
        ax.tick_params(direction="in", length=3)
    ax.grid(False)

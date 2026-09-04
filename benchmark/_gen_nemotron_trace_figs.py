"""Two-model versions of the §6.6 trace figures (Qwen panel a, Nemotron panel b), written
to the SAME filenames the thesis includes: qwen_tool_mix, ch6_stage_reach,
ch6_validation_scatter. Qwen data as in _gen_qwen_figs/_gen_ch6_figs*; Nemotron joins
kept shard reps to traces by trace_idx (legacy first-run reps positionally)."""
import glob, json, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import _thesis_fig_style as S
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/home/tony/PhIDOv1/PhIDO-Release/thesis/figures")
S.use_style()
VALIDATE = {"validate_ports", "get_component_properties", "get_component_info",
            "get_pdk_cell_details", "get_pdk_implementations", "get_module_params", "resolve_function"}
STAGES = [("ExtractedConcepts", "Concepts"), ("RequirementManifest", "Requirements"),
          ("DesignIntent", "Design intent"), ("ComponentSelectionLLM", "Selection"),
          ("ComplianceVerdict", "Compliance"), ("CriticVerdict", "Critic")]

def qwen_runs():
    lab = json.load(open(ROOT/"benchmark/results/_qwen_review_input_review3_labels.json"))["annotations"]
    for tf in glob.glob(str(ROOT/"benchmark/results/qwen_traces/*.json")):
        d = json.load(open(tf))
        a2 = lab.get(f"{d['arm']}::{d['prompt_id']}::rep{d['rep']}", {}).get("axes", {}).get("A2")
        yield d["arm"], a2, d["trace"]

def nemo_runs():
    lab = json.load(open(ROOT/"benchmark/results/_nemotron_review_input_review3_labels.json"))["annotations"]
    shard = {}
    for sf in sorted(glob.glob(str(ROOT/"benchmark/results/nemotron_trace_ablation_w*.json"))):
        for arm, byp in json.load(open(sf)).items():
            for pid, reps in byp.items():
                shard.setdefault(arm, {}).setdefault(pid, []).extend(reps)
    for arm, byp in shard.items():
        for pid, reps in byp.items():
            for i, srec in enumerate(reps):
                ti = srec.get("trace_idx") or (i + 1)
                tf = ROOT / f"benchmark/results/nemotron_traces/{arm}__{pid}__rep{ti}.json"
                if not tf.exists():
                    continue
                a2 = lab.get(f"{arm}::{pid}::rep{i+1}", {}).get("axes", {}).get("A2")
                yield arm, a2, json.load(open(tf))["trace"]

MODELS = [("Qwen3.6-27B", list(qwen_runs())), ("Nemotron-3-Ultra-550B", list(nemo_runs()))]

# ============ tool mix, two stacked panels ============
# Tools are grouped exploration-then-validation (suffixes "(exp)"/"(val)"), the order of
# the shared tools is identical in both panels (pooled share, computed over both models),
# every bar carries its percentage label, and the sub-1.5% tail folds into "other".
counts = []
for name, runs in MODELS:
    succ, fail = Counter(), Counter()
    ns = nf = 0
    for arm, a2, trace in runs:
        tgt = succ if a2 == "correct" else (fail if a2 == "wrong" else None)
        if tgt is None: continue
        if a2 == "correct": ns += 1
        else: nf += 1
        for s in trace:
            for tc in s.get("tool_calls", []): tgt[tc["name"]] += 1
    counts.append((name, succ, fail, ns, nf))

def _main(succ, fail):
    s_tot, f_tot = sum(succ.values()), sum(fail.values())
    return {t for t in set(succ) | set(fail)
            if 100*succ[t]/s_tot >= 1.5 or 100*fail[t]/f_tot >= 1.5}

mains = [_main(s, f) for _, s, f, _, _ in counts]
pooled = Counter()
for _, succ, fail, _, _ in counts:
    s_tot, f_tot = sum(succ.values()), sum(fail.values())
    for t in set().union(*mains):
        pooled[t] += 100*succ[t]/s_tot + 100*fail[t]/f_tot
ORDER = sorted(set().union(*mains), key=lambda t: (t in VALIDATE, -pooled[t]))

fig, axes = plt.subplots(2, 1, figsize=(5.9, 7.0))
for ax, (name, succ, fail, ns, nf), main, tag in zip(axes, counts, mains, "ab"):
    s_tot, f_tot = sum(succ.values()), sum(fail.values())
    tools = [t for t in ORDER if t in main] + ["other"]
    s_pct = [100*succ[t]/s_tot for t in tools[:-1]] + [100*sum(v for k, v in succ.items() if k not in main)/s_tot]
    f_pct = [100*fail[t]/f_tot for t in tools[:-1]] + [100*sum(v for k, v in fail.items() if k not in main)/f_tot]
    labels = [f"{t} ({'val' if t in VALIDATE else 'exp'})" for t in tools[:-1]] + ["other"]
    print(name, "success:", {t: round(p) for t, p in zip(tools, s_pct)})
    print(name, "fail:   ", {t: round(p) for t, p in zip(tools, f_pct)})
    ys, y = [], 0.0  # extra gap where the exploration group ends and before "other"
    for i, t in enumerate(tools):
        if i and ((t in VALIDATE and tools[i-1] not in VALIDATE) or t == "other"):
            y -= 0.8
        ys.append(y); y -= 1.0
    H = 0.36
    ax.barh([y+0.20 for y in ys], s_pct, height=H, color=S.BLUE, edgecolor=S.INK, lw=0.4,
            label=f"graded correct (n={ns})")
    ax.barh([y-0.20 for y in ys], f_pct, height=H, color=S.ORANGE, hatch="////",
            edgecolor=S.ORANGE_DK, lw=0.4, label=f"graded wrong (n={nf})")
    for y, sp, fp in zip(ys, s_pct, f_pct):
        ax.text(sp+0.4, y+0.20, f"{sp:.0f}%", va="center", fontsize=6.8, color="#555555")
        ax.text(fp+0.4, y-0.20, f"{fp:.0f}%", va="center", fontsize=6.8, color="#555555")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontfamily="monospace", fontsize=8)
    ax.set_ylim(min(ys)-0.75, max(ys)+0.75)
    ax.set_xlim(0, max(max(s_pct), max(f_pct)) + 6)
    ax.set_title(f"({tag}) {name}", fontsize=9.5, fontweight="bold")
    S.box(ax); ax.tick_params(left=False)
    ax.legend(loc="lower right", fontsize=8)
axes[1].set_xlabel("share of the run's tool calls (%)")
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(OUT/f"qwen_tool_mix.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig); print("wrote qwen_tool_mix (2 panels)")

# ============ stage reach, two panels side by side ============
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.7))
for ax, (name, runs), tag in zip(axes, MODELS, "ab"):
    buckets = {"SUCCESS": [], "PARTIAL": [], "FAIL": []}
    for arm, a2, trace in runs:
        b = {"correct": "SUCCESS", "partial": "PARTIAL", "wrong": "FAIL"}.get(a2)
        if b: buckets[b].append({s.get("schema") for s in trace if s.get("kind") == "structured"})
    for b, col, mk, lw in [("SUCCESS", S.BLUE, "o", 1.6), ("PARTIAL", S.PURPLE, "^", 1.1),
                           ("FAIL", S.ORANGE, "s", 1.6)]:
        vals = [100*st.mean(sch in run for run in buckets[b]) for sch, _ in STAGES]
        ax.plot(range(len(STAGES)), vals, color=col, marker=mk, ms=5, lw=lw,
                markeredgecolor=S.INK, markeredgewidth=0.4, label=f"{b.lower()} (n={len(buckets[b])})")
    print(name, "reach:", {b: len(v) for b, v in buckets.items()})
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([l for _, l in STAGES], fontsize=7, rotation=30, ha="right")
    ax.set_title(f"({tag}) {name}", fontsize=9.5, fontweight="bold")
    ax.set_ylim(60 if tag == "a" else 25, 104); S.box(ax); ax.legend(loc="lower left", fontsize=7.5)
axes[0].set_ylabel("% of runs reaching stage")
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(OUT/f"ch6_stage_reach.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig); print("wrote ch6_stage_reach (2 panels)")

# ============ validation scatter, two panels ============
NAMES = {"base_agentic": "Minimal agentic", "base_plus_kg": "+ grounding",
         "base_plus_gate": "+ gate", "base_plus_critic": "+ critic", "full": "Full"}
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.9))
for ax, (name, runs), tag in zip(axes, MODELS, "ab"):
    per = defaultdict(lambda: {"val": [], "succ": []})
    for arm, a2, trace in runs:
        if a2 is None: continue
        tools = [tc["name"] for s in trace for tc in s.get("tool_calls", [])]
        per[arm]["val"].append(sum(t in VALIDATE for t in tools))
        per[arm]["succ"].append(1.0 if a2 == "correct" else 0.0)
    OFFSETS = {
        "a": {"Minimal agentic": (0.3, -1.2, "left"), "+ grounding": (0.3, 0.5, "left"),
              "+ gate": (0.3, -0.6, "left"), "+ critic": (-0.4, 1.0, "right"), "Full": (-0.4, -1.8, "right")},
        "b": {"Minimal agentic": (0.35, 0.6, "left"), "+ grounding": (0.35, -1.0, "left"),
              "+ gate": (-0.35, 0.6, "right"), "+ critic": (-0.4, -1.4, "right"), "Full": (-0.4, 1.0, "right")},
    }[tag]
    pts = []
    for k, disp in NAMES.items():
        xv = st.mean(per[k]["val"]); yv = 100*st.mean(per[k]["succ"]); pts.append((xv, yv, disp))
        ax.scatter([xv], [yv], s=48, color=S.BLUE, edgecolor=S.INK, lw=0.5, zorder=3)
    print(name, "scatter:", [(d, round(x,1), round(y)) for x, y, d in pts])
    xs = [p[0] for p in pts]; ys_ = [p[1] for p in pts]
    xpad = (max(xs)-min(xs))*0.30 + 0.5; ypad = (max(ys_)-min(ys_))*0.35 + 1.6
    ax.set_xlim(min(xs)-xpad, max(xs)+xpad); ax.set_ylim(min(ys_)-ypad, max(ys_)+ypad)
    for xv, yv, disp in pts:
        dx, dy, ha = OFFSETS[disp]
        ax.annotate(disp, (xv, yv), xytext=(xv+dx, yv+dy), fontsize=7.5, color=S.INK, ha=ha)
    ax.set_title(f"({tag}) {name}", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("validation calls per run")
    S.box(ax)
axes[0].set_ylabel("connectivity-correct (%)")
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(OUT/f"ch6_validation_scatter.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig); print("wrote ch6_validation_scatter (2 panels)")

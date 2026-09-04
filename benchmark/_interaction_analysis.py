"""Feature-interaction analysis for the E2 ablation.

Two independent estimators of whether the three features combine additively:

  (1) SUPERPOSITION RESIDUAL (both models). Under a no-interaction (additive) model the full arm
      should score  base + d_kg + d_gate + d_critic  where d_X = base_plus_X - base. The residual
      (observed full - additive prediction) is the three-way interaction. Needs only the additive
      arms, so it works for Qwen, which never ran leave-one-out arms.
  (2) LOO-vs-ADDITIVE GAP (gpt-5.4 only). For each feature, contribution measured in context
      (full_minus_X -> full) minus contribution measured on the bare scaffold
      (base -> base_plus_X). A non-zero gap means that feature's value depends on the others.

Per-prompt residuals are ranked so the strongest interaction prompts can be traced individually.
"""
import json
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results"
DB = json.load(open(RES / "_transition_cases.json"))
ALL = DB["all_reps"]
GOLD = DB["gold"]
PROMPTS = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
FEATS = ["kg", "gate", "critic"]


import os

# UNCONDITIONAL=1 reproduces the convention §6.6 of the thesis uses (a run that errors or
# produces no netlist scores 0) rather than excluding those reps from the mean. Default is 0
# (exclude), matching the E2 tables of §6.5.
UNCOND = os.getenv("UNCONDITIONAL") == "1"
CONV = "unconditional (failures scored 0)" if UNCOND else "conditional (failures excluded)"


def pmeans(model, arm):
    """{pid: mean edge-F1} for one arm, under the selected scoring convention."""
    out = {}
    for pid, reps in ALL.get(model, {}).get(arm, {}).items():
        v = []
        for r in reps:
            f1 = r.get("edgeF1")
            if isinstance(f1, (int, float)):
                v.append(f1)
            elif UNCOND and str(r.get("status")) not in ("ok", "scored"):
                v.append(0.0)
        if v:
            out[pid] = st.mean(v)
    return out


def wil(v):
    try:
        return wilcoxon(v)[1] if any(abs(x) > 1e-9 for x in v) else 1.0
    except ValueError:
        return float("nan")


report = ["# Feature-interaction analysis (E2 ablation, edge-F1 vs b3_gold_v2)\n"]
R = report.append
R(f"**Scoring convention: {CONV}.** Re-run with `UNCONDITIONAL=1` for the other. "
  "Section 6.6 of the thesis scores unconditionally; the E2 tables of Section 6.5 condition.\n")
R("Do the knowledge graph, the topology gate and the critic combine additively? Two estimators, "
  "one usable on both models and one on gpt-5.4 only, are reported. Both are computed from "
  "per-prompt mean edge-F1 over scoreable reps (K=3), paired across prompts.\n")

# ---------------------------------------------------------------- (1) superposition residual
R("\n## 1. Superposition residual (both models)\n")
R("Under additivity the full arm should reach `base + d_kg + d_gate + d_critic`, i.e. "
  "`kg + gate + critic - 2*base`. The residual `full - prediction` is the three-way interaction: "
  "negative means the features **interfere** (the stack delivers less than the sum of its parts).\n")
R("| model | base | +kg | +gate | +critic | full observed | full predicted (additive) | residual | n | p |")
R("|---|---|---|---|---|---|---|---|---|---|")
resid_by_prompt = {}
for model in ("qwen", "gpt54"):
    base_arm = "base_agentic"
    b, k, g, c, f = (pmeans(model, base_arm), pmeans(model, "base_plus_kg"),
                     pmeans(model, "base_plus_gate"), pmeans(model, "base_plus_critic"),
                     pmeans(model, "full"))
    ps = sorted(set(b) & set(k) & set(g) & set(c) & set(f))
    pred = {p: k[p] + g[p] + c[p] - 2 * b[p] for p in ps}
    res = {p: f[p] - pred[p] for p in ps}
    resid_by_prompt[model] = res
    v = [res[p] for p in ps]
    R(f"| {model} | {st.mean([b[p] for p in ps]):.3f} | {st.mean([k[p] for p in ps]):.3f} | "
      f"{st.mean([g[p] for p in ps]):.3f} | {st.mean([c[p] for p in ps]):.3f} | "
      f"{st.mean([f[p] for p in ps]):.3f} | {st.mean([pred[p] for p in ps]):.3f} | "
      f"**{st.mean(v):+.3f}** | {len(ps)} | {wil(v):.4f} |")
R("\nThe additive prediction is unclamped and can exceed the [0,1] range of the metric; where it "
  "does, part of the residual is ceiling compression rather than true interference. The count of "
  "such prompts is given below.\n")
for model in ("qwen", "gpt54"):
    b, k, g, c = (pmeans(model, "base_agentic"), pmeans(model, "base_plus_kg"),
                  pmeans(model, "base_plus_gate"), pmeans(model, "base_plus_critic"))
    ps = sorted(resid_by_prompt[model])
    over = [p for p in ps if (k[p] + g[p] + c[p] - 2 * b[p]) > 1.0]
    R(f"- {model}: additive prediction exceeds 1.0 on {len(over)}/{len(ps)} prompts"
      + (f" ({', '.join(over)})" if over else ""))
    sub = [p for p in ps if p not in over]
    if sub:
        vv = [resid_by_prompt[model][p] for p in sub]
        R(f"  - residual restricted to the {len(sub)} prompts without ceiling compression: "
          f"**{st.mean(vv):+.3f}** (p={wil(vv):.4f})")

# ------------------------------------------------------------- (2) LOO vs additive (gpt-5.4)
R("\n## 2. In-context vs bare-scaffold contribution (gpt-5.4 only)\n")
R("Qwen never ran the `full_minus_*` arms, so this estimator is available for gpt-5.4 alone. "
  "A feature whose two contributions disagree is one whose value depends on what else is running.\n")
R("| feature | additive Δ | LOO Δ | interaction (LOO − additive) | n | p(additive) | p(LOO) |")
R("|---|---|---|---|---|---|---|")
for feat in FEATS:
    b, x = pmeans("gpt54", "base_agentic"), pmeans("gpt54", f"base_plus_{feat}")
    fm, fu = pmeans("gpt54", f"full_minus_{feat}"), pmeans("gpt54", "full")
    ps = sorted(set(b) & set(x) & set(fm) & set(fu))
    da = [x[p] - b[p] for p in ps]
    dl = [fu[p] - fm[p] for p in ps]
    inter = [dl[i] - da[i] for i in range(len(ps))]
    R(f"| {feat} | {st.mean(da):+.3f} | {st.mean(dl):+.3f} | **{st.mean(inter):+.3f}** | "
      f"{len(ps)} | {wil(da):.3f} | {wil(dl):.3f} |")

# ------------------------------------------------------- per-prompt ranking for case tracing
R("\n## 3. Prompts ranked by interaction magnitude\n")
R("The prompts where the stack departs most from the sum of its parts — the candidates for "
  "per-run tracing. `full` and the single-feature arms' per-prompt means are shown so the "
  "direction is readable.\n")
for model in ("qwen", "gpt54"):
    R(f"\n### {model}\n")
    b, k, g, c, f = (pmeans(model, "base_agentic"), pmeans(model, "base_plus_kg"),
                     pmeans(model, "base_plus_gate"), pmeans(model, "base_plus_critic"),
                     pmeans(model, "full"))
    rows = sorted(resid_by_prompt[model].items(), key=lambda kv: kv[1])
    R("| prompt | lvl | base | +kg | +gate | +critic | full | additive pred | residual |")
    R("|---|---|---|---|---|---|---|---|---|")
    for p, r in rows:
        R(f"| {p} | {GOLD[p]['level_paper']} | {b[p]:.2f} | {k[p]:.2f} | {g[p]:.2f} | "
          f"{c[p]:.2f} | {f[p]:.2f} | {k[p]+g[p]+c[p]-2*b[p]:.2f} | **{r:+.2f}** |")

# ------------------------------------------------- does the stack beat its best single feature?
R("\n## 4. Does the full stack beat its best single feature?\n")
R("A practical framing of the same question: per prompt, compare the full stack against the best "
  "of the three single-feature arms (an oracle that picks the right feature per prompt).\n")
R("Two versions. The **per-prompt oracle** takes the best of the three feature arms on each "
  "prompt; it is not achievable in practice (you cannot know which feature will win) and is an "
  "upper bound. The **best fixed arm** is the single feature arm with the highest mean over all "
  "prompts, which is a choice a designer could actually make.\n")
R("| model | full | best fixed single arm | Δ vs fixed | per-prompt oracle | Δ vs oracle | full wins/ties/loses vs oracle |")
R("|---|---|---|---|---|---|---|")
for model in ("qwen", "gpt54"):
    k, g, c, f = (pmeans(model, "base_plus_kg"), pmeans(model, "base_plus_gate"),
                  pmeans(model, "base_plus_critic"), pmeans(model, "full"))
    ps = sorted(set(k) & set(g) & set(c) & set(f))
    arms = {"+kg": k, "+gate": g, "+critic": c}
    fixed_name, fixed = max(arms.items(), key=lambda kv: st.mean([kv[1][p] for p in ps]))
    fx = [f[p] - fixed[p] for p in ps]
    best = {p: max(k[p], g[p], c[p]) for p in ps}
    d = [f[p] - best[p] for p in ps]
    R(f"| {model} | {st.mean([f[p] for p in ps]):.3f} | {fixed_name} "
      f"{st.mean([fixed[p] for p in ps]):.3f} | **{st.mean(fx):+.3f}** (p={wil(fx):.3f}) | "
      f"{st.mean([best[p] for p in ps]):.3f} | **{st.mean(d):+.3f}** | "
      f"{sum(1 for x in d if x > 1e-9)} / {sum(1 for x in d if abs(x) <= 1e-9)} / "
      f"{sum(1 for x in d if x < -1e-9)} |")

(RES / "_interaction_analysis.md").write_text("\n".join(report))
print("\n".join(report))

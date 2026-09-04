"""Deterministic structural classification of every differing (model, transition, prompt) cell.

No LLM judgement anywhere: each rep gets a mode from mutually-exclusive rules over its stored
status / node multiset / edge multiset, and each case gets a transition signature from the
per-arm mode distributions plus component-F1 movement. Emits a full human-readable forensics
report (complete matrices, mode frequencies, structural census, 158-case appendix).

Inputs : results/_transition_cases.json, results/_transition_noise_flags.json
Outputs: results/_transition_forensics.md, results/_transition_classified.json
"""
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results"
DB = json.load(open(RES / "_transition_cases.json"))
CASES, GOLD = DB["cases"], DB["gold"]
NOISE = json.load(open(RES / "_transition_noise_flags.json"))

PROMPTS = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
TRS = ["rigid_baseline->base_agentic", "base_agentic->base_plus_kg",
       "base_agentic->base_plus_gate", "base_agentic->base_plus_critic",
       "base_agentic->full",
       "full_minus_kg->full", "full_minus_gate->full", "full_minus_critic->full"]
SHORT = {"rigid_baseline->base_agentic": "rigid→base", "base_agentic->base_plus_kg": "base→+kg",
         "base_agentic->base_plus_gate": "base→+gate",
         "base_agentic->base_plus_critic": "base→+critic", "base_agentic->full": "base→full",
         "full_minus_kg->full": "LOO kg", "full_minus_gate->full": "LOO gate",
         "full_minus_critic->full": "LOO critic"}
# LOO arms exist only in the gpt-5.4 store; matrices render per-model over transitions that
# actually have cases, so Qwen is not padded with empty LOO columns.
FEATURES = [("kg", "base_agentic->base_plus_kg", "full_minus_kg->full"),
            ("gate", "base_agentic->base_plus_gate", "full_minus_gate->full"),
            ("critic", "base_agentic->base_plus_critic", "full_minus_critic->full")]
EPS = 1e-9
OK = ("ok", "scored")


# ------------------------------------------------------------------ rep-level modes
def multiset(pairs):
    return Counter({k: v for k, v in (pairs or [])})


def mset_f1(pred, gold):
    tp = sum((pred & gold).values())
    fp, fn = sum((pred - gold).values()), sum((gold - pred).values())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def rep_mode(rep, pid):
    """One deterministic label per rep. Priority order is fixed and mutually exclusive."""
    s = str(rep.get("status"))
    if s.startswith("error:"):
        return "infra-error", s.split("error:")[1]
    if s.startswith("failed:"):
        return "pipeline-failure", s.split("failed:")[1]
    if s not in OK:
        return "unknown-status", s
    if not rep.get("n_nodes"):
        return "empty-netlist", ""
    if not rep.get("n_edges"):
        return "edgeless-netlist", ""
    if rep.get("edgeF1") is None:
        return "unscored-edgeless-gold", ""
    if rep["edgeF1"] >= 1.0 - EPS:
        return "exact-match", ""
    gc = multiset(GOLD[pid]["gold_classes"])
    pc = multiset(rep.get("pred_classes"))
    if pc == gc:
        return "wiring-error", f"{len(rep.get('missing') or [])}miss/{len(rep.get('extra') or [])}extra"
    npred, ngold = sum(pc.values()), sum(gc.values())
    if npred < ngold:
        sub = "deficit"
    elif npred > ngold:
        sub = "surplus"
    else:
        sub = "substitution"
    return f"component-{sub}", f"{npred}v{ngold}comp"


def classify_reps(reps, pid):
    out = []
    for r in reps:
        m, d = rep_mode(r, pid)
        out.append({"mode": m, "detail": d, "edgeF1": r.get("edgeF1"),
                    "status": r.get("status"), "n_nodes": r.get("n_nodes"),
                    "n_edges": r.get("n_edges")})
    return out


# ------------------------------------------------------- case-level (transition) signature
def scoreable(rm):
    return [r for r in rm if isinstance(r["edgeF1"], (int, float))]


def comp_f1_mean(reps, pid):
    gc = multiset(GOLD[pid]["gold_classes"])
    v = [mset_f1(multiset(r.get("pred_classes")), gc) for r in reps
         if str(r.get("status")) in OK and r.get("pred_classes")]
    return st.mean(v) if v else None


def comp_sets(reps):
    """distinct component multisets produced by an arm (as sorted tuples)."""
    return {tuple(sorted((k, v) for k, v in (r.get("pred_classes") or [])))
            for r in reps if str(r.get("status")) in OK and r.get("pred_classes")}


def case_signature(c):
    pid = c["prompt"]
    ra, rb = classify_reps(c["reps_a"], pid), classify_reps(c["reps_b"], pid)
    ca, cb = Counter(r["mode"] for r in ra), Counter(r["mode"] for r in rb)
    n_sa, n_sb = len(scoreable(ra)), len(scoreable(rb))
    edgeless_d = cb["edgeless-netlist"] - ca["edgeless-netlist"]
    exact_d = cb["exact-match"] - ca["exact-match"]
    broke_a = ca["infra-error"] + ca["pipeline-failure"] + ca["empty-netlist"]
    broke_b = cb["infra-error"] + cb["pipeline-failure"] + cb["empty-netlist"]
    fa, fb = comp_f1_mean(c["reps_a"], pid), comp_f1_mean(c["reps_b"], pid)
    comp_moved = None if (fa is None or fb is None) else fb - fa
    sets_a, sets_b = comp_sets(c["reps_a"]), comp_sets(c["reps_b"])
    comp_changed = bool(sets_a ^ sets_b)

    tags = []
    if broke_b != broke_a:
        tags.append("reachability-loss" if broke_b > broke_a else "reachability-gain")
    if edgeless_d:
        tags.append("edgeless-induced" if edgeless_d > 0 else "edgeless-rescue")
    if comp_changed:
        if comp_moved is not None and abs(comp_moved) > 0.01:
            tags.append("component-shift-toward-gold" if comp_moved > 0
                        else "component-shift-away-from-gold")
        else:
            tags.append("component-shift-lateral")
    if not comp_changed and not edgeless_d and abs(c["delta"]) > EPS:
        tags.append("wiring-shift-improved" if c["delta"] > 0 else "wiring-shift-regressed")
    if exact_d:
        tags.append("exact-gained" if exact_d > 0 else "exact-lost")
    if not tags:
        tags.append("no-structural-change")

    key = f"{c['model']}|{c['transition']}|{pid}"
    return {"model": c["model"], "transition": c["transition"], "prompt": pid,
            "delta": c["delta"], "mean_a": c["mean_a"], "mean_b": c["mean_b"],
            "level_paper": GOLD[pid]["level_paper"], "n_comp_gold": GOLD[pid]["n_comp"],
            "modes_a": dict(ca), "modes_b": dict(cb), "reps_a": ra, "reps_b": rb,
            "n_scoreable_a": n_sa, "n_scoreable_b": n_sb,
            "compF1_a": fa, "compF1_b": fb, "comp_moved": comp_moved,
            "comp_changed": comp_changed, "edgeless_delta": edgeless_d,
            "exact_delta": exact_d, "tags": tags,
            "delta_basis": c.get("delta_basis", "both-scored"),
            "in_noise_band": bool(NOISE.get(key, False))}


def fmt_reps(rs):
    """'mode[detail](f1)' per rep, comma-joined."""
    out = []
    for r in rs:
        det = f"[{r['detail']}]" if r["detail"] else ""
        f1 = "—" if not isinstance(r["edgeF1"], (int, float)) else f"{r['edgeF1']:.2f}"
        out.append(f"{r['mode']}{det}({f1})")
    return ", ".join(out)


def verdict(s):
    """Deterministic one-sentence reading, generated from the signature only."""
    a, b = s["transition"].split("->")
    d = s["delta"]
    dirn = "improved" if d > EPS else ("regressed" if d < -EPS else "scored identically")
    bits = []
    if "reachability-gain" in s["tags"]:
        bits.append(f"{b} produced a scoreable netlist where {a} did not "
                    f"({s['n_scoreable_a']}→{s['n_scoreable_b']} scoreable reps)")
    if "reachability-loss" in s["tags"]:
        bits.append(f"{b} lost runs to errors/failures that {a} completed "
                    f"({s['n_scoreable_a']}→{s['n_scoreable_b']} scoreable reps)")
    if "edgeless-rescue" in s["tags"]:
        bits.append(f"{b} emitted {abs(s['edgeless_delta'])} fewer wiring-free (edgeless) netlist(s)")
    if "edgeless-induced" in s["tags"]:
        bits.append(f"{b} emitted {s['edgeless_delta']} more wiring-free (edgeless) netlist(s)")
    for t in s["tags"]:
        if t.startswith("component-shift"):
            mv = s["comp_moved"]
            bits.append(f"component selection changed ({t.split('component-shift-')[1]}"
                        + (f", comp-F1 {mv:+.2f}" if mv is not None else "") + ")")
    for t in s["tags"]:
        if t.startswith("wiring-shift"):
            bits.append("the same components were wired differently")
    if not bits:
        bits.append("no structural difference is visible in the stored artifacts")
    tail = " [inside within-arm rep-noise band]" if s["in_noise_band"] else ""
    if s["delta_basis"] != "both-scored":
        arm = a if s["delta_basis"] == "a-unscoreable" else b
        tail += (f" [Δ is NOT a clean edge-F1 comparison: {arm} produced no scoreable rep for "
                 f"this prompt, so its mean is substituted with 0]")
    return f"{dirn} (Δ={d:+.3f}): " + "; ".join(bits) + "." + tail


SIGS = [case_signature(c) for c in CASES]
for s in SIGS:
    s["verdict"] = verdict(s)
SIGS.sort(key=lambda s: (s["model"], TRS.index(s["transition"]), PROMPTS.index(s["prompt"])))

# ------------------------------------------------------------------------ report
L = []
W = L.append
W("# Transition forensics: every performance delta, classified\n")
W("Complete-coverage structural analysis of the Qwen3.6-27B and gpt-5.4 ablations. "
  "Every (model, transition, prompt) cell whose two arms differ is included — the case list is "
  "generated mechanically from the data, so no case selection by judgement enters anywhere.\n")
W("## Method\n")
W("For each of the 24 prompts and each of the 5 arm transitions, per-prompt mean edge-F1 "
  "(over `status=ok` reps, K=3) is compared between the two arms. A cell is a **case** if the "
  "means differ *or* the per-rep status sets differ. Each rep then receives one deterministic "
  "mode from mutually-exclusive rules; each case receives a transition signature derived from "
  "the two arms' mode distributions and their component-F1 movement against gold. "
  "Scoring is against the revised `b3_gold_v2` (24 prompts).\n")
W("**Rep modes** (priority order, mutually exclusive): `infra-error` (status `error:*`) · "
  "`pipeline-failure` (status `failed:*`) · `empty-netlist` (no components) · "
  "`edgeless-netlist` (components built, zero connections) · `unscored-edgeless-gold` "
  "(gold itself has no edges, edge-F1 undefined) · `exact-match` (edge-F1 = 1.0) · "
  "`wiring-error` (component multiset equals gold, connections wrong) · "
  "`component-deficit` / `component-surplus` / `component-substitution` (component multiset "
  "differs from gold).\n")
W("**Case tags**: `reachability-gain|loss` (change in how many reps yielded a scoreable "
  "netlist) · `edgeless-rescue|induced` · `component-shift-toward-gold|away-from-gold|lateral` "
  "· `wiring-shift-improved|regressed` (identical components, different connections) · "
  "`exact-gained|lost` · `no-structural-change`. A case is additionally flagged "
  "**in-noise-band** when |Δ| does not exceed the combined standard error of the two arms' rep "
  "means, i.e. the delta is not distinguishable from within-arm repeat variance.\n")

W("## 1. Complete delta matrices\n")
W("Δ = mean edge-F1(arm B) − mean edge-F1(arm A); blank = the two arms scored identically and "
  "had identical status sets; `±0` = identical mean but differing statuses; "
  "`~` marks a delta inside the rep-noise band.\n")
by = {(s["model"], s["transition"], s["prompt"]): s for s in SIGS}
model_trs = {m: [t for t in TRS if any(s["model"] == m and s["transition"] == t for s in SIGS)]
             for m in ("qwen", "gpt54")}
for model, name in (("qwen", "Qwen3.6-27B"), ("gpt54", "gpt-5.4")):
    W(f"\n### {name}\n")
    if model == "qwen":
        W("*(no leave-one-out columns: the `full_minus_*` arms were never run for Qwen.)*\n")
    W("| prompt | lvl | " + " | ".join(SHORT[t] for t in model_trs[model]) + " |")
    W("|---|---|" + "---|" * len(model_trs[model]))
    for p in PROMPTS:
        cells = []
        for t in model_trs[model]:
            s = by.get((model, t, p))
            if not s:
                cells.append("")
            elif abs(s["delta"]) <= EPS:
                cells.append("±0")
            else:
                cells.append(f"{s['delta']:+.3f}" + ("~" if s["in_noise_band"] else ""))
        lvl = GOLD[p]["level_paper"]
        W(f"| {p} | {lvl} | " + " | ".join(cells) + " |")

W("\n## 2. Case census and noise accounting\n")
W("`unscoreable-arm` counts cases where one arm produced no scoreable rep for that prompt, so its "
  "mean is substituted with 0 — a reachability difference, not a clean edge-F1 comparison. Those "
  "cases are flagged individually in the appendix.\n")
W("| model | transition | cases | improved | regressed | status-only | in noise band | unscoreable-arm |")
W("|---|---|---|---|---|---|---|---|")
for model in ("qwen", "gpt54"):
    for t in TRS:
        cs = [s for s in SIGS if s["model"] == model and s["transition"] == t]
        if not cs:
            continue
        W(f"| {model} | {SHORT[t]} | {len(cs)} | "
          f"{sum(1 for s in cs if s['delta'] > EPS)} | "
          f"{sum(1 for s in cs if s['delta'] < -EPS)} | "
          f"{sum(1 for s in cs if abs(s['delta']) <= EPS)} | "
          f"{sum(1 for s in cs if s['in_noise_band'])} | "
          f"{sum(1 for s in cs if s['delta_basis'] != 'both-scored')} |")

W("\n## 2b. Survivorship sensitivity: excluded-failure vs failure-as-zero\n")
W("The headline edge-F1 averages a prompt over its `status=ok` reps only, so a rep that crashed "
  "or produced no design is **dropped** rather than scored. That convention answers *\"when it "
  "produces a design, how good is it?\"*, and it flatters whichever arm fails most often — for "
  "Qwen the rigid pipeline, which loses 25 of 72 reps to infrastructure errors. The engineering "
  "question is closer to *\"how often do you get a correct design?\"*, which scores a failed rep "
  "as 0. Both are reported here; the scaffold uplift is larger under the second, so the headline "
  "number is the conservative one.\n")
ARMS_ORDER = ["rigid_baseline", "baseline", "base_agentic", "base_plus_kg", "base_plus_gate",
              "base_plus_critic", "full"]
W("\n| model | arm | mean edge-F1 (ok reps only) | mean edge-F1 (failure = 0) | reps scored / total |")
W("|---|---|---|---|---|")
surv = {}
for model, byarm in DB["all_reps"].items():
    for arm in ARMS_ORDER:
        if arm not in byarm:
            continue
        pp_ok, pp_zero, n_ok, n_tot = [], [], 0, 0
        for pid, reps in byarm[arm].items():
            vok, vz = [], []
            for r in reps:
                n_tot += 1
                f1 = r.get("edgeF1")
                if isinstance(f1, (int, float)):
                    vok.append(f1); vz.append(f1); n_ok += 1
                elif str(r.get("status")) not in OK:
                    vz.append(0.0)          # crashed / produced nothing -> counts as wrong
            if vok:
                pp_ok.append(st.mean(vok))
            if vz:
                pp_zero.append(st.mean(vz))
        mo = st.mean(pp_ok) if pp_ok else float("nan")
        mz = st.mean(pp_zero) if pp_zero else float("nan")
        surv[(model, arm)] = (mo, mz)
        W(f"| {model} | {arm} | {mo:.3f} | {mz:.3f} | {n_ok}/{n_tot} |")
W("")
for model, rigid in (("qwen", "rigid_baseline"), ("gpt54", "baseline")):
    if (model, rigid) in surv and (model, "base_agentic") in surv:
        ro, rz = surv[(model, rigid)]
        ao, az = surv[(model, "base_agentic")]
        W(f"- **{model} scaffold uplift** (rigid → base_agentic): "
          f"{ao - ro:+.3f} excluding failures, **{az - rz:+.3f} counting failures as 0**.")
W("\nThese arm-mean differences are not identical to the paired Wilcoxon deltas reported "
  "elsewhere (e.g. Qwen rigid→base_agentic +0.195): the paired statistic uses only prompts where "
  "*both* arms yielded a scoreable rep, whereas each column above averages each arm over every "
  "prompt it scored. The gap between the two is itself a survivorship effect and moves in the "
  "same direction.\n")

W("\n## 3. Rep-mode frequencies per arm (complete corpus)\n")
W("Every rep of every arm over **all 24 prompts** — not restricted to the differing cells — so "
  "the columns are directly comparable. This is the census that carries the cross-model "
  "mechanism.\n")
arm_modes = defaultdict(Counter)
for model, byarm in DB["all_reps"].items():
    for arm, byp in byarm.items():
        for pid, reps in byp.items():
            for r in reps:
                arm_modes[(model, arm)][rep_mode(r, pid)[0]] += 1
ALLM = ["exact-match", "wiring-error", "component-deficit", "component-surplus",
        "component-substitution", "edgeless-netlist", "unscored-edgeless-gold",
        "empty-netlist", "pipeline-failure", "infra-error"]
for model in ("qwen", "gpt54"):
    W(f"\n### {model}\n")
    arms = [a for (m, a) in arm_modes if m == model]
    order = [a for a in ["rigid_baseline", "baseline", "base_agentic", "base_plus_kg",
                         "base_plus_gate", "base_plus_critic", "full"] if a in arms]
    W("| mode | " + " | ".join(order) + " |")
    W("|---|" + "---|" * len(order))
    for m in ALLM:
        row = [str(arm_modes[(model, a)][m]) for a in order]
        if any(x != "0" for x in row):
            W(f"| {m} | " + " | ".join(row) + " |")
    W("| **total reps** | " + " | ".join(str(sum(arm_modes[(model, a)].values()))
                                          for a in order) + " |")

W("\n## 4. Transition-signature frequencies\n")
W("How often each structural mechanism appears, per transition. A case may carry several tags.\n")
for model in ("qwen", "gpt54"):
    W(f"\n### {model}\n")
    tagset = sorted({t for s in SIGS if s["model"] == model for t in s["tags"]})
    W("| tag | " + " | ".join(SHORT[t] for t in model_trs[model]) + " |")
    W("|---|" + "---|" * len(model_trs[model]))
    for tag in tagset:
        row = [str(sum(1 for s in SIGS if s["model"] == model and s["transition"] == t
                       and tag in s["tags"])) for t in model_trs[model]]
        W(f"| {tag} | " + " | ".join(row) + " |")

W("\n## 4b. Additive vs leave-one-out contribution per feature (gpt-5.4)\n")
W("The additive contrast asks what a feature adds to the bare scaffold "
  "(`base_agentic` → `base_plus_X`). The leave-one-out contrast asks what the same feature adds "
  "to the rest of the stack (`full_minus_X` → `full`) — the in-context question, which can differ "
  "if features interact. Both are oriented as the **contribution of X**, so a negative number "
  "means the feature costs edge-F1. Paired over the prompts where both arms scored; Wilcoxon "
  "signed-rank, uncorrected (three features × two contrasts = six tests, so treat single "
  "asterisks with the multiplicity caution already applied elsewhere in the thesis).\n")


def arm_prompt_means(model, arm):
    byp = DB["all_reps"].get(model, {}).get(arm, {})
    out = {}
    for pid, reps in byp.items():
        v = [r["edgeF1"] for r in reps if isinstance(r.get("edgeF1"), (int, float))]
        if v:
            out[pid] = st.mean(v)
    return out


def paired_delta(model, arm_a, arm_b):
    """Δ = mean(arm_b) − mean(arm_a), paired over prompts scored by both."""
    A, B = arm_prompt_means(model, arm_a), arm_prompt_means(model, arm_b)
    ps = sorted(set(A) & set(B))
    if not ps:
        return None
    diff = [B[p] - A[p] for p in ps]
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(diff)[1] if any(diff) else 1.0
    except Exception:  # noqa: BLE001
        p = float("nan")
    return (st.mean(diff), p, len(ps),
            sum(1 for d in diff if d > EPS), sum(1 for d in diff if d < -EPS))


W("| feature | additive Δ (base→base+X) | n, w/l, p | LOO Δ (full−X→full) | n, w/l, p | agree? |")
W("|---|---|---|---|---|---|")
for feat, add_t, loo_t in FEATURES:
    a_arm, b_arm = add_t.split("->")
    la_arm, lb_arm = loo_t.split("->")
    ad = paired_delta("gpt54", a_arm, b_arm)
    ld = paired_delta("gpt54", la_arm, lb_arm)
    def cell(x):
        return "—" if x is None else f"{x[0]:+.3f}"
    def stat(x):
        return "—" if x is None else f"n={x[2]}, {x[3]}/{x[4]}, p={x[1]:.3f}{'*' if x[1] < 0.05 else ''}"
    agree = "—"
    if ad and ld:
        agree = "yes" if (ad[0] > 0) == (ld[0] > 0) else "**no — sign flips**"
    W(f"| {feat} | {cell(ad)} | {stat(ad)} | {cell(ld)} | {stat(ld)} | {agree} |")
W("")

W("\n## 5. Case appendix — all "
  f"{len(SIGS)} cases\n")
W("`A:`/`B:` list the per-rep modes of the earlier and later arm; `(f1)` is that rep's edge-F1.\n")
cur = None
for s in SIGS:
    hdr = (s["model"], s["transition"])
    if hdr != cur:
        cur = hdr
        a, b = s["transition"].split("->")
        W(f"\n### {s['model']} · {a} → {b}\n")

    W(f"**{s['prompt']}** (paper level {s['level_paper']}, gold {s['n_comp_gold']} components) — "
      f"Δ={s['delta']:+.3f}  ·  tags: {', '.join(s['tags'])}"
      + ("  ·  ⚠ in-noise-band" if s["in_noise_band"] else ""))
    W(f"- A: {fmt_reps(s['reps_a'])}")
    W(f"- B: {fmt_reps(s['reps_b'])}")
    W(f"- {s['verdict']}\n")

(RES / "_transition_forensics.md").write_text("\n".join(L))
json.dump(SIGS, open(RES / "_transition_classified.json", "w"), indent=1)
print(f"classified {len(SIGS)} cases -> results/_transition_forensics.md "
      f"(+ _transition_classified.json)")
print(f"in-noise-band: {sum(1 for s in SIGS if s['in_noise_band'])}")
print("tag totals:", dict(Counter(t for s in SIGS for t in s["tags"]).most_common()))

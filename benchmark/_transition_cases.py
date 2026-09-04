"""Extract EVERY (model, transition, prompt) cell where per-prompt mean edge-F1 differs
between the two arms of a transition, for Qwen3.6-27B and gpt-5.4. No sampling, no
representative-case selection: the case list is defined mechanically by the data.

For each case, emit per-rep: status, edgeF1, typed-edge diff vs gold (missing/extra),
predicted component classes, and artifact pointers (Qwen agent trace / gpt-5.4 DOT file).
Output: results/_transition_cases.json + a delta-matrix summary to stdout.
"""
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results"
os.environ["E2_MODEL"] = "qwen/qwen3.6-27b"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")

import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402

PROMPTS = [f"L3_{i:02d}" for i in range(1, 13)] + [f"L4_{i:02d}" for i in range(1, 13)]
# Additive transitions (both models) then leave-one-out transitions (gpt-5.4 only — the LOO
# arms were never run for Qwen). LOO is oriented full_minus_X -> full so its delta is the
# CONTRIBUTION of feature X to the rest of the stack, the same sign convention as the additive
# base_agentic -> base_plus_X delta, making the two directly comparable.
TRANSITIONS = [("rigid_baseline", "base_agentic"),
               ("base_agentic", "base_plus_kg"),
               ("base_agentic", "base_plus_gate"),
               ("base_agentic", "base_plus_critic"),
               ("base_agentic", "full"),
               ("full_minus_kg", "full"),
               ("full_minus_gate", "full"),
               ("full_minus_critic", "full")]
EPS = 1e-9


def parse_edges(edge_list):
    out = []
    for e in edge_list:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2:
            out.append(frozenset(eps))
    return out


def gold_typed_edges(pid):
    return SC._entry_to_topology(SC.GOLD[pid]).typed_edges()


def _estr(e):
    """typed-edge element -> stable string (elements may be frozensets/tuples)."""
    if isinstance(e, (frozenset, set, tuple, list)):
        return " -- ".join(sorted(str(x) for x in e))
    return str(e)


def edge_diff(pred_topo, pid):
    """typed-edge multiset diff after the same accept/class-map the scorer applies."""
    g = SC.GOLD[pid]
    pt = SC._apply_accept(pred_topo, g).typed_edges()
    gt = gold_typed_edges(pid)
    missing = sorted(_estr(e) for e in (gt - pt).elements())
    extra = sorted(_estr(e) for e in (pt - gt).elements())
    return missing, extra


def qwen_rep(rec, pid, arm, rep_i):
    out = {"status": rec.get("status"), "edgeF1": None, "missing": None, "extra": None,
           "pred_classes": None, "n_llm_calls": rec.get("n_llm_calls"),
           "n_nodes": len(rec.get("nodes") or {}), "n_edges": len(rec.get("edges") or []),
           "trace": f"qwen_traces/{arm}__{pid}__rep{rep_i}.json" if arm != "rigid_baseline" else None}
    if rec.get("status") != "ok" or not rec.get("nodes"):
        return out
    topo = Topology(nodes=dict(rec["nodes"]), edges=parse_edges(rec.get("edges", [])), external={})
    try:
        s = SC._score_topo(topo, pid)
    except Exception as ex:  # noqa: BLE001
        out["status"] = f"score_error:{type(ex).__name__}"
        return out
    out["edgeF1"] = s["edgeF1"]
    out["pred_classes"] = sorted(Counter(rec["nodes"].values()).items())
    try:
        out["missing"], out["extra"] = edge_diff(topo, pid)
    except Exception:  # noqa: BLE001
        pass
    return out


def load_qwen():
    """{arm: {pid: [rep dicts]}} — shards w0..w3 are authoritative; warn if the old
    no-suffix store holds anything the shards don't."""
    raw = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(RES / "qwen_trace_ablation_w*.json"))):
        for arm, byp in json.load(open(f)).items():
            for pid, reps in byp.items():
                raw[arm][pid].extend(reps)
    old = json.load(open(RES / "qwen_trace_ablation.json"))
    extra = sum(len(reps) for byp in old.values() for reps in byp.values())
    if extra:
        print(f"NOTE: legacy qwen_trace_ablation.json holds {extra} reps (NOT merged; "
              f"shards are authoritative — verify overlap if this is unexpected)")
    for f in sorted(glob.glob(str(RES / "qwen_rigid_baseline*.json"))):
        if f.endswith(".bak") or ".pre_shard" in f:
            continue
        for arm, byp in json.load(open(f)).items():
            for pid, reps in byp.items():
                raw[arm][pid].extend(reps)
    out = defaultdict(lambda: defaultdict(list))
    for arm, byp in raw.items():
        for pid, reps in byp.items():
            for i, rec in enumerate(reps, 1):
                out[arm][pid].append(qwen_rep(rec, pid, arm, i))
    return out


def g54_rep(rec, pid, arm, rep_i):
    out = {"status": rec.get("status"), "edgeF1": None, "missing": None, "extra": None,
           "pred_classes": None, "hcollapsed": rec.get("hcollapsed"),
           "n_nodes": len(rec.get("pred_node_map") or {}),
           "n_edges": len(rec.get("pred_edges") or []),
           "dot": f"_ablation_correctness_gpt54_v2_dots/{arm}/{pid}_rep{rep_i}.dot"}
    if not isinstance(rec.get("edgeF1"), (int, float)):
        return out
    out["edgeF1"] = rec["edgeF1"]
    nm = rec.get("pred_node_map") or {}
    out["pred_classes"] = sorted(Counter(nm.values()).items())
    pe = rec.get("pred_edges") or []
    if nm:
        topo = Topology(nodes=dict(nm), edges=[frozenset(e) for e in pe if len(e) == 2],
                        external={})
        try:
            out["missing"], out["extra"] = edge_diff(topo, pid)
        except Exception:  # noqa: BLE001
            pass
    return out


def load_g54():
    d = json.load(open(RES / "_ablation_correctness_gpt54_v2.json"))
    out = defaultdict(lambda: defaultdict(list))
    for arm, byp in d.items():
        for pid, reps in byp.items():
            for i, rec in enumerate(reps, 1):
                out[arm][pid].append(g54_rep(rec, pid, arm, i))
    return out


def mean_f1(reps):
    v = [r["edgeF1"] for r in reps if isinstance(r["edgeF1"], (int, float))]
    return st.mean(v) if v else None


MODELS = {"qwen": (load_qwen(), {}),
          "gpt54": (load_g54(), {"rigid_baseline": "baseline"})}

cases = []
matrix = defaultdict(dict)
for model, (data, keymap) in MODELS.items():
    real_arms = set(data)          # capture BEFORE any defaultdict access materialises a key
    for a, b in TRANSITIONS:
        ka, kb = keymap.get(a, a), keymap.get(b, b)
        if ka not in real_arms or kb not in real_arms:
            continue               # e.g. Qwen never ran the full_minus_* (LOO) arms
        tname = f"{a}->{b}"
        for pid in PROMPTS:
            ra, rb = data[ka].get(pid, []), data[kb].get(pid, [])
            ma, mb = mean_f1(ra), mean_f1(rb)
            if ma is None and mb is None:
                continue
            delta = (mb if mb is not None else 0.0) - (ma if ma is not None else 0.0)
            matrix[(model, tname)][pid] = round(delta, 3)
            statuses_differ = ({r["status"] for r in ra} != {r["status"] for r in rb})
            # When an arm produced no scoreable rep for this prompt its mean is undefined; the
            # delta below substitutes 0 for it, which is the failure-as-zero convention rather
            # than the headline exclude-failures one. Flag it so downstream never reads such a
            # cell as a clean edge-F1 comparison.
            basis = ("both-scored" if (ma is not None and mb is not None)
                     else ("a-unscoreable" if ma is None else "b-unscoreable"))
            if abs(delta) > EPS or ma is None or mb is None or statuses_differ:
                cases.append({"model": model, "transition": tname, "prompt": pid,
                              "mean_a": ma, "mean_b": mb, "delta": round(delta, 3),
                              "delta_basis": basis,
                              "direction": "improved" if delta > EPS else ("regressed" if delta < -EPS else "status-only"),
                              "reps_a": ra, "reps_b": rb})

cases.sort(key=lambda c: -abs(c["delta"]))
gold_summary = {pid: {"n_comp": SC.GOLD[pid]["n_comp"], "level_paper": SC.GOLD[pid]["level_paper"],
                      "gold_classes": sorted(SC._entry_to_topology(SC.GOLD[pid]).component_counts().items()),
                      "gold_edges": sorted(_estr(e) for e in gold_typed_edges(pid).elements())} for pid in PROMPTS}
# Full per-arm rep corpus (ALL prompts, ALL arms) so downstream census tables are comparable
# across arms rather than restricted to the prompts that happened to differ.
all_reps = {model: {arm: {pid: reps for pid, reps in byp.items()}
                    for arm, byp in data.items()}
            for model, (data, _) in MODELS.items()}
json.dump({"cases": cases, "gold": gold_summary, "all_reps": all_reps},
          open(RES / "_transition_cases.json", "w"), indent=1)

print(f"\nTOTAL differing cases: {len(cases)} "
      f"(of {len(MODELS) * len(TRANSITIONS) * len(PROMPTS)} cells)")
for model in MODELS:
    for a, b in TRANSITIONS:
        t = f"{a}->{b}"
        row = matrix[(model, t)]
        diffs = {p: d for p, d in row.items() if abs(d) > EPS}
        pos = sum(1 for d in diffs.values() if d > 0)
        neg = len(diffs) - pos
        print(f"\n{model:6} {t:34} differing prompts: {len(diffs):2} (+{pos}/-{neg})")
        for p in PROMPTS:
            if p in diffs:
                print(f"    {p}  Δ={diffs[p]:+.3f}")

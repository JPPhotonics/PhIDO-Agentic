"""Topology-CORRECTNESS eval (not funnel): score each arm's generated netlist against the
independent b3_gold topologies (functional-class nodes + typed-edge multiset F1). This measures
"did it wire the right kinds of blocks together" — the thing the funnel (buildable+simulatable)
cannot see. Topology-only: stops at the netlist, skips layout/sim/DRC (faster + no SAX/DRC).

Arms: rigid_baseline (build_rigid_netlist -> circuit_dsl) and agentic single_shot
(explore -> finalize -> gf_netlist_yaml).

K reps per prompt (E2_K, default 3) to average over the known agentic run-to-run variance.
Each rep is tagged with a STATUS so the aggregate can separate three distinct outcomes that
compF1=0 used to conflate:
  - "scored"        : a non-empty netlist was produced and scored against gold
  - "empty_netlist" : the arm produced a netlist with zero component instances
  - "failed:<gate>" : the pipeline aborted (e.g. the empty_netlist guard, a validation gate)
  - "error:<Exc>"   : an exception was raised
Correctness (compF1/edgeF1/GED) is aggregated ONLY over "scored" reps; netlist-PRODUCTION
rate is reported separately. Resumable via incremental JSON (reps are appended; a killed run
is continued by re-invoking — it tops each prompt back up to K). Scratch.
"""
import json
import os
from pathlib import Path

import yaml
from b3_eval import _apply_accept, _entry_to_topology, load_gold
from topology_eval import Topology, score_topology

MODEL = os.getenv("E2_MODEL", "o3-mini")
K = int(os.getenv("E2_K", "3"))
OUT = os.getenv("E2_CORRECTNESS_OUT", f"results/_correctness_k{K}.json")
# When set, save each agentic rep's DOT schematic (.dot source + .png rendered exactly as the
# Streamlit app does) under DOT_DIR/<arm>/<pid>_rep<k>.{dot,png}. Unset -> no files written.
DOT_DIR = os.getenv("E2_DOT_DIR")
# Gold + paired prompts default to the original 24-prompt draft; override BOTH together to run a
# different set (e.g. E2_GOLD=b3_gold_v2.json E2_PROMPTS=e2_prompts_v2.json for the 12-L3+12-L4 set).
_GOLD_PATH = os.getenv("E2_GOLD")
meta, GOLD = load_gold(Path(_GOLD_PATH) if _GOLD_PATH else None)
CM = meta["class_map"]
prompts = {p["id"]: p for p in json.load(open(os.getenv("E2_PROMPTS", "e2_prompts.json")))["prompts"]}


def _endpoint(nodes, ep):
    # Return a RAW instance-id endpoint ("C1.o3"). Do NOT pre-resolve to the
    # component class here: Topology.typed_endpoint() already maps id->class via
    # the nodes dict (same single-resolution path the gold uses). Pre-resolving
    # to class made typed_endpoint re-resolve "MMI_1x2.o3" -> "?.o3" (class not a
    # key in nodes), which zeroed typed-edge F1 for every prompt.
    inst, port = ep.split(",", 1)
    return f"{inst.strip()}.{port.strip()}"


def topo_from_gf(netlist_yaml):
    n = yaml.safe_load(netlist_yaml) or {}
    nodes = {k: v.get("component") for k, v in (n.get("instances") or {}).items()}
    cls = {nid: CM.get(c, c) for nid, c in nodes.items()}
    edges = []
    for rd in (n.get("routes") or {}).values():
        for a, b in (rd.get("links") or {}).items():
            edges.append(frozenset((_endpoint(nodes, a), _endpoint(nodes, b))))
    return Topology(nodes=cls, edges=edges, external={})


def topo_from_dsl(circuit_dsl):
    nodes = {k: (v.get("component") if isinstance(v, dict) else v)
             for k, v in (circuit_dsl.get("nodes") or {}).items()}
    cls = {nid: CM.get(c, c) for nid, c in nodes.items()}
    edges = []
    raw = circuit_dsl.get("edges")
    if isinstance(raw, dict):
        for spec in raw.values():
            link = spec.get("link", "") if isinstance(spec, dict) else ""
            if ":" in link:
                a, b = link.split(":", 1) if link.count(":") == 1 else link.split(": ", 1)
                if "," in a and "," in b:
                    edges.append(frozenset((_endpoint(nodes, a), _endpoint(nodes, b))))
    return Topology(nodes=cls, edges=edges, external={})


def _score_topo(pred, gid):
    """Score a NON-empty predicted topology against gold.

    When ``PHIDO_HCOLLAPSE=1``, apply the edge-verified hierarchical collapse (hierarchical_collapse.
    collapse) to the prediction BEFORE scoring, so a correct primitive decomposition of a composite
    block (e.g. 2 couplers + phase-per-arm) ties the atomic library-cell realisation. The RAW
    (pre-collapse) wiring is always persisted (pred_node_map / pred_edges) so the choice is auditable
    and re-scorable both ways.
    """
    g = GOLD[gid]
    raw_node_map = dict(pred.nodes)
    raw_edges = [sorted(e) for e in pred.edges]
    collapsed = False
    if os.getenv("PHIDO_HCOLLAPSE") == "1":
        from hierarchical_collapse import collapse
        before = len(pred.nodes)
        pred = collapse(pred)
        collapsed = len(pred.nodes) != before
    s = score_topology(_apply_accept(pred, g), _entry_to_topology(g))
    return {"status": "scored",
            "compF1": round(s["component_prf"]["f1"], 3),
            "edgeF1": (None if s["typed_edge_prf"]["f1"] is None
                       else round(s["typed_edge_prf"]["f1"], 3)),
            "ged": s["approx_ged"]["ged"], "level": g["level"],
            "level_paper": g["level_paper"], "n_comp": g["n_comp"],
            "certain": g["topology_certain"],
            "pred_nodes": list(pred.nodes.values()),  # post-collapse when HCOLLAPSE on
            "hcollapsed": collapsed,
            # RAW predicted wiring (pre-collapse, post class-map). node_map is instance-id -> class;
            # edges are sorted [ "id.port", "id.port" ] pairs. Lets a later re-score recompute
            # edgeF1/GED and re-run WIRING-VERIFIED collapse independently of this run's flag.
            "pred_node_map": raw_node_map,
            "pred_edges": raw_edges}


def run_baseline(prompt):
    import baseline_runner as B
    topo = topo_from_dsl(B.build_rigid_netlist(prompt, MODEL))
    # Third element is the agentic app's rendered DOT string; the rigid baseline has none.
    return topo, ("ok" if topo.nodes else "empty_netlist"), None


def _run_agentic_cfg(prompt, *, kg=True, routing=False, gate=True, critic=True,
                     disable_enforcement=False):
    """Agentic arm parameterized by the 4 ablation FEATURE bits (same mapping as
    ``e2_runner.RunConfig`` so the ablation grid and this correctness harness stay in sync):
      kg      -> explore grounding_rounds=5 + RETRIEVAL_BACKEND=hybrid (else 0 / lexical)
      routing -> extraction_mode="iterative" (else "single_shot")
      gate    -> enable_topology_gate (the Clingo TOPOLOGY gate)
      critic  -> max_critic_rounds=2 (else 0)
    The app default = (kg=1, routing=0, gate=1, critic=1) == the ablation's ``full``.
    NOTE: the iterative-``routing`` feature is treated as unimplemented and is held OFF across the
    entire ablation grid (no routing arms) — see the E2_ARMS=ablation grid below.

    ``disable_enforcement`` is a SEPARATE axis from the ``gate`` feature: it toggles the
    simulatable SELECTION gate (``PHIDO_DISABLE_SELECTION_GATE``) + retriever filter
    (``RETRIEVAL_ENFORCE_SIMULATABLE=0``). Held ON (enforcement OFF) for the ablation arms so the
    known gate/pre-filter asymmetry doesn't confound the feature attribution. Env set/restored
    per call so arms don't leak.
    """
    from mcp_servers.pipeline_orchestrator import explore_and_ask, run_pipeline_finalize
    env = {"RETRIEVAL_BACKEND": "hybrid" if kg else "lexical"}
    if disable_enforcement:
        env["PHIDO_DISABLE_SELECTION_GATE"] = "1"
        env["RETRIEVAL_ENFORCE_SIMULATABLE"] = "0"
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        est = None
        for ev in explore_and_ask(prompt, MODEL, 15, 5 if kg else 0):
            if ev.get("type") == "clarification":
                est = ev
        pdone = None
        aborted = None
        for ev in run_pipeline_finalize(explore_state=est, user_prompt=prompt, model=MODEL,
                                        max_critic_rounds=2 if critic else 0,
                                        enable_topology_gate=bool(gate),
                                        enable_ar_gate=False,
                                        extraction_mode="iterative" if routing else "single_shot"):
            t = ev.get("type")
            if t == "pipeline_done":
                pdone = ev["result"]
            elif t in ("validation_failed", "error"):
                msg = ev.get("gate") or ev.get("message") or "error"
                # The orchestrator's generic "Interpreter failed to produce DesignIntent"
                # fires right AFTER the specific extract-failure event ("LLM refused ...:
                # <exc>"); last-wins masked the actual exception in the banked status
                # (2026-08-28 Nemotron rerun forensics) — keep the specific one.
                if not (msg.startswith("Interpreter failed to produce DesignIntent") and aborted):
                    aborted = msg[:160]
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    if pdone is None:
        return Topology(nodes={}, edges=[]), (f"failed:{aborted}" if aborted else "no_pipeline_done"), None
    topo = topo_from_gf(pdone.get("gf_netlist_yaml") or "")
    # dot_string is the exact DOT the Streamlit app renders (LLM-routed, port records + layout).
    return topo, ("ok" if topo.nodes else "empty_netlist"), pdone.get("dot_string")


def _run_agentic(prompt, disable_enforcement=False):
    """App-default agentic config (kg on, single_shot, topology-gate on, critic on)."""
    return _run_agentic_cfg(prompt, kg=True, routing=False, gate=True, critic=True,
                            disable_enforcement=disable_enforcement)


def run_agentic(prompt):
    return _run_agentic(prompt, disable_enforcement=False)


def run_agentic_nogate(prompt):
    return _run_agentic(prompt, disable_enforcement=True)


def rep(fn, prompt, gid):
    g = GOLD[gid]
    try:
        topo, status, dot = fn(prompt)
    except Exception as e:  # noqa: BLE001
        return {"status": f"error:{type(e).__name__}", "error": str(e)[:150],
                "level": g["level"], "level_paper": g["level_paper"],
                "n_comp": g["n_comp"], "certain": g["topology_certain"]}
    if status == "ok":
        r = _score_topo(topo, gid)
    else:
        r = {"status": status, "level": g["level"], "level_paper": g["level_paper"],
             "n_comp": g["n_comp"], "certain": g["topology_certain"], "pred_nodes": []}
    # Persist the agentic app's DOT schematic verbatim (None for the rigid baseline).
    if dot:
        r["dot_string"] = dot
    return r


ARMS = {"baseline": run_baseline, "agentic": run_agentic,
        "agentic_nogate": run_agentic_nogate}

# E2_ARMS=ablation swaps the 3-arm headline set for the feature-attribution grid, scored on
# the SAME correctness metric + v2 gold. Feature bits = (kg, routing, gate, critic); all agentic
# arms run enforcement-OFF (disable_enforcement) so the selection-gate confound is held constant.
# LOO arms = full - X ; additive arms = base + X. base_agentic=(0,0,0,0).
# ROUTING (iterative extraction) is treated as UNIMPLEMENTED: the routing bit is held 0 across
# every arm, so there are no routing arms. "full" is therefore kg+gate+critic (== app default,
# formerly "full_minus_routing"). Grid = baseline + 8 agentic arms.
if os.getenv("E2_ARMS") == "ablation":
    _BITS = {
        "base_agentic": (0, 0, 0, 0),
        "full": (1, 0, 1, 1),                 # kg+gate+critic == app-default agentic
        "full_minus_kg": (0, 0, 1, 1),
        "full_minus_gate": (1, 0, 0, 1),
        "full_minus_critic": (1, 0, 1, 0),
        "base_plus_kg": (1, 0, 0, 0),
        "base_plus_gate": (0, 0, 1, 0),
        "base_plus_critic": (0, 0, 0, 1),
    }

    def _mk_arm(bits):
        kg, routing, gate, critic = bits
        return lambda prompt: _run_agentic_cfg(
            prompt, kg=bool(kg), routing=bool(routing), gate=bool(gate),
            critic=bool(critic), disable_enforcement=True)

    ARMS = {"baseline": run_baseline}
    for _name, _bits in _BITS.items():
        ARMS[_name] = _mk_arm(_bits)

# Arms compared against baseline in the correctness headline (everything except baseline).
AGENTIC_ARMS = [a for a in ARMS if a != "baseline"]

# E2_PROMPT_SUBSET (comma-sep gold ids) restricts the run to a subset (e.g. a cheap screening
# pilot); unset -> all prompts. Applied in main() and report().
_subset = os.getenv("E2_PROMPT_SUBSET")
PROMPT_IDS = [p.strip() for p in _subset.split(",")] if _subset else list(prompts)


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def report(out):
    """Emit a Markdown report separating netlist-PRODUCTION from topology CORRECTNESS."""
    lines = [f"# E2 topology-correctness (K={K}, model={MODEL})", ""]
    lines.append("Correctness (compF1/edgeF1/GED) is over **scored** reps only; production "
                 "rate counts reps that yielded a non-empty netlist. Empty/failed reps are "
                 "NOT scored as correctness=0 (they are a pipeline-reliability signal).")
    lines.append("")
    # netlist-production rate
    lines.append("## Netlist-production rate (reps that produced a non-empty circuit)")
    lines.append("| arm | scored | empty | failed | error | total | production rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        reps = [r for pid in PROMPT_IDS for r in out.get(arm, {}).get(pid, [])]
        sc = sum(r["status"] == "scored" for r in reps)
        em = sum(r["status"] == "empty_netlist" for r in reps)
        fa = sum(r["status"].startswith("failed") or r["status"] == "no_pipeline_done" for r in reps)
        er = sum(r["status"].startswith("error") for r in reps)
        tot = len(reps)
        pr = round(sc / tot, 3) if tot else None
        lines.append(f"| {arm} | {sc} | {em} | {fa} | {er} | {tot} | {pr} |")
    lines.append("")
    # per-prompt mean compF1 (scored reps only)
    def pmean(arm, pid, key):
        vals = [r[key] for r in out.get(arm, {}).get(pid, []) if r["status"] == "scored"]
        return _mean(vals)
    # Correctness headline: each agentic-like arm vs baseline, over prompts where BOTH
    # that arm and baseline have >=1 scored rep (pairing recomputed per arm).
    for arm in AGENTIC_ARMS:
        paired = [pid for pid in PROMPT_IDS
                  if pmean("baseline", pid, "compF1") is not None
                  and pmean(arm, pid, "compF1") is not None]
        lines.append(f"## Correctness headline: {arm} vs baseline "
                     f"(paired n={len(paired)}, both scored >=1 rep)")
        lines.append(f"| metric | baseline | {arm} | Δ({arm}-baseline) |")
        lines.append("|---|---|---|---|")
        for key, lo_better in (("compF1", False), ("edgeF1", False), ("ged", True)):
            b = _mean([pmean("baseline", pid, key) for pid in paired])
            a = _mean([pmean(arm, pid, key) for pid in paired])
            d = round(a - b, 3) if (a is not None and b is not None) else None
            note = " (lower better)" if lo_better else ""
            lines.append(f"| {key}{note} | {b} | {a} | {d} |")
        lines.append("")
        # Stratify by the paper's Table 1 level (component count), NOT the repo's unreviewed
        # tags. Repo tags are shown as a second table for comparison only.
        for axis, label in (("level_paper", "paper Table 1 / component-count"),
                            ("level", "repo tags — UNREVIEWED, for comparison only")):
            lines.append(f"### {arm}: compF1 by level ({label}; paired, scored reps)")
            lines.append(f"| level | baseline | {arm} |")
            lines.append("|---|---|---|")
            for L in (1, 2, 3, 4):
                ks = [pid for pid in paired if GOLD[pid][axis] == L]
                b = _mean([pmean("baseline", pid, "compF1") for pid in ks])
                a = _mean([pmean(arm, pid, "compF1") for pid in ks])
                lines.append(f"| L{L} (n={len(ks)}) | {b} | {a} |")
            lines.append("")
    # empty/failed prompt list per arm (reliability detail)
    for arm in ARMS:
        bad = [pid for pid in PROMPT_IDS
               if any(r["status"] != "scored" for r in out.get(arm, {}).get(pid, []))]
        lines.append(f"- **{arm}** prompts with >=1 non-scored rep: {bad}")
    path = OUT.rsplit(".", 1)[0] + ".md"
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"report -> {path}", flush=True)


def _save_dot(arm, pid, k, dot):
    """Save the agentic DOT schematic as .dot source + .png (app's own renderer)."""
    from mcp_servers.pipeline_orchestrator import _render_dot_to_png
    d = os.path.join(DOT_DIR, arm)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{pid}_rep{k}")
    with open(stem + ".dot", "w") as fh:
        fh.write(dot)
    png = _render_dot_to_png(dot)
    if png:
        with open(stem + ".png", "wb") as fh:
            fh.write(png)
    return stem + (".png" if png else ".dot")


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm, fn in ARMS.items():
        out.setdefault(arm, {})
        for pid in PROMPT_IDS:
            out[arm].setdefault(pid, [])
            while len(out[arm][pid]) < K:
                r = rep(fn, prompts[pid]["prompt"], pid)
                out[arm][pid].append(r)
                k = len(out[arm][pid])
                saved = ""
                if DOT_DIR and r.get("dot_string"):
                    saved = " dot->" + _save_dot(arm, pid, k, r["dot_string"])
                json.dump(out, open(OUT, "w"), indent=1)
                print(f"[{arm}/{pid} rep{k}/{K}] "
                      f"{r['status']} compF1={r.get('compF1')}{saved}", flush=True)
    report(out)
    print("DONE", flush=True)


# Guard the run loop so the module can be imported (e.g. for a re-score/smoke test) without
# executing the full sweep — the launcher runs `python _score_correctness.py` (__name__ ==
# "__main__"), so its behaviour is unchanged.
if __name__ == "__main__":
    main()

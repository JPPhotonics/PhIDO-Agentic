"""Aggregate the post-hoc layout/DRC pass (``_layout_from_dot.py``) into funnel tables + report.

Joins each layout record (arm, pid, rep) back to its original netlist-level record (status,
compF1/edgeF1, gold level) and reports, per configuration and per level:

  * the five-stage funnel (instantiate -> routing_ok -> models -> sim_success -> drc_clean),
    UNCONDITIONAL (a rep that produced no netlist fails every stage) and CONDITIONAL on a
    netlist having been produced;
  * paired feature contrasts (McNemar on drc_clean and on sim_success) — additive vs
    base_agentic, leave-one-out vs full — via e2_funnel.compare_arms;
  * DRC attribution: share of reps whose design contains a library cell that violates DRC at its
    own defaults (``_pdk_component_drc.json``), and drc_clean restricted to designs free of them;
  * SAX attribution: share of sim failures that are the closed-circuit "no ports given" case.

Usage: SOURCE=gpt54|qwen PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv-layout/bin/python benchmark/_layout_report.py
Writes results/_layout_<SOURCE>_report.md and results/_layout_<SOURCE>_merged.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import e2_funnel as F  # noqa: E402

SOURCE = os.getenv("SOURCE", "gpt54")
RES = ROOT / "benchmark" / "results"
STAGES = F.STAGES
GOLD = {e["id"]: e for e in json.load(open(ROOT / "benchmark" / "b3_gold_v2.json"))["gold"]}

if SOURCE == "gpt54":
    ARMS = ["base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic",
            "full", "full_minus_kg", "full_minus_gate", "full_minus_critic"]
else:
    ARMS = ["base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic", "full"]
ADDITIVE = [("kg", "base_agentic", "base_plus_kg"), ("gate", "base_agentic", "base_plus_gate"),
            ("critic", "base_agentic", "base_plus_critic")]
LOO = [("kg", "full_minus_kg", "full"), ("gate", "full_minus_gate", "full"),
       ("critic", "full_minus_critic", "full")]


def load_layout() -> list[dict]:
    recs: list[dict] = []
    for f in sorted(glob.glob(str(RES / f"_layout_{SOURCE}_w*.json"))) + [str(RES / f"_layout_{SOURCE}.json")]:
        if Path(f).exists():
            recs += json.load(open(f))
    seen, out = set(), []
    for r in recs:  # dedupe (a rep can appear in an unsharded smoke file and a shard)
        k = (r["arm"], r["pid"], r["rep"], r.get("shard"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def load_original() -> dict:
    """(arm, pid, rep) -> original netlist-level record (status, compF1, edgeF1)."""
    out = {}
    if SOURCE == "gpt54":
        d = json.load(open(RES / "_ablation_correctness_gpt54_v2.json"))
        for arm, byp in d.items():
            for pid, rs in byp.items():
                for k, r in enumerate(rs, 1):
                    out[(arm, pid, k, None)] = r
    else:
        # Qwen records hold the raw topology (nodes/edges), not scores: recompute edgeF1 exactly as
        # _score_qwen_n24.py does (unscoreable/non-ok -> None here; the funnel handles failures).
        os.environ.setdefault("E2_MODEL", "qwen/qwen3.6-27b")
        os.environ.setdefault("E2_GOLD", str(ROOT / "benchmark" / "b3_gold_v2.json"))
        os.environ.setdefault("E2_PROMPTS", str(ROOT / "benchmark" / "e2_prompts_v2.json"))
        import re as _re
        import _score_correctness as SC
        from topology_eval import Topology

        def _edges(el):
            o = []
            for e in el:
                eps = _re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
                if len(eps) == 2:
                    o.append(frozenset(eps))
            return o

        for f in sorted(glob.glob(str(RES / "qwen_trace_ablation_w*.json"))):
            for arm, byp in json.load(open(f)).items():
                for pid, rs in byp.items():
                    for k, r in enumerate(rs, 1):
                        r = dict(r)
                        if r.get("status") == "ok" and r.get("nodes"):
                            try:
                                sc = SC._score_topo(Topology(nodes=dict(r["nodes"]), edges=_edges(r.get("edges", [])),
                                                             external={}), pid)
                                r["edgeF1"], r["compF1"] = sc["edgeF1"], sc["compF1"]
                            except Exception:  # noqa: BLE001
                                pass
                        out[(arm, pid, int(r.get("trace_idx") or k), Path(f).stem)] = r
    return out


def load_dirty_cells() -> set[str]:
    p = RES / "_pdk_component_drc.json"
    if not p.exists():
        return set()
    return {r["module"] for r in json.load(open(p)) if r.get("drc_clean") is False}


def rate(xs):
    return f"{(sum(xs) / len(xs)):.2f} ({sum(xs)}/{len(xs)})" if xs else "—"


def funnel_row(recs: list[dict], unconditional: bool) -> str:
    recs = [r for r in recs if not r.get("dot_unusable")]
    rs = recs if unconditional else [r for r in recs if r.get("recon") not in (None, "no_dot")]
    if not rs:
        return " | ".join(["—"] * len(STAGES)) + " | 0"
    cells = []
    for s in STAGES:
        cells.append(rate([F.normalize(r["stages"])[s] for r in rs]))
    return " | ".join(cells) + f" | {len(rs)}"


def main() -> None:
    lay = load_layout()
    orig = load_original()
    dirty = load_dirty_cells()
    for r in lay:
        o = orig.get((r["arm"], r["pid"], r["rep"], r.get("shard")), {})
        r["orig_status"] = o.get("status")
        r["compF1"], r["edgeF1"] = o.get("compF1"), o.get("edgeF1")
        r["level"] = GOLD.get(r["pid"], {}).get("level")
        mods = set((r.get("diag") or {}).get("modules", {}).values())
        r["has_dirty_cell"] = bool(mods & dirty)
        r["dirty_cells"] = sorted(mods & dirty)
        se = (r.get("diag") or {}).get("sim_error") or ""
        r["sim_no_ports"] = "no ports given" in se
        # Stored DOT is not a schematic at all (LLM edge-router returned prose / an empty graph while the
        # netlist was still exported from the builder DSL) -> the rep cannot be reconstructed; it is
        # EXCLUDED from funnel denominators rather than counted as a layout failure.
        r["dot_unusable"] = (r.get("recon") == "rebuild_failed"
                             and "no nodes parsed" in ((r.get("diag") or {}).get("rebuild_error") or ""))
    (RES / f"_layout_{SOURCE}_merged.json").write_text(json.dumps(lay, indent=1, default=str))

    by_arm = defaultdict(list)
    for r in lay:
        by_arm[r["arm"]].append(r)
    expected = {a: len([k for k in orig if k[0] == a]) for a in ARMS}

    L = [f"# Post-hoc layout + DRC funnel — {SOURCE}", "",
         "Netlists rebuilt from each rep's stored DOT (modules + port-level wiring, PDK-default parameters), "
         "placed with the pipeline's own Graphviz placement, then GDS -> SAX -> KLayout DRC "
         "(`_layout_from_dot.py`; env `.venv-layout`, kfactory 0.21.7). Rigid baseline is absent: its "
         "reps stored class-mapped nodes only, so its netlists cannot be rebuilt.", ""]
    L += ["## Coverage", "", "| configuration | original reps | layout records | rebuilt ok | no netlist (orig) | DOT unusable (excluded) | rebuild failed (other) | layout error | timeout/error | funnel n |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        rs = by_arm.get(a, [])
        c = Counter(r.get("recon") for r in rs)
        du = sum(r["dot_unusable"] for r in rs)
        L.append(f"| {a} | {expected.get(a, 0)} | {len(rs)} | {c.get('ok', 0)} | {c.get('no_dot', 0)} | {du} | "
                 f"{c.get('rebuild_failed', 0) - du} | {c.get('layout_error', 0)} | "
                 f"{sum(v for k, v in c.items() if k and (k.startswith('error') or k == 'timeout'))} | {len(rs) - du} |")
    hdr = "| configuration | " + " | ".join(STAGES) + " | n |"
    sep = "|---" * (len(STAGES) + 2) + "|"
    for label, uncond in (("UNCONDITIONAL (no-netlist reps fail every stage)", True),
                          ("CONDITIONAL on a netlist having been produced", False)):
        L += ["", f"## Funnel, {label}", "", hdr, sep]
        for a in ARMS:
            L.append(f"| {a} | {funnel_row(by_arm.get(a, []), uncond)} |")
        for lvl in (3, 4):
            L += ["", f"### Level {lvl} only", "", hdr, sep]
            for a in ARMS:
                L.append(f"| {a} | {funnel_row([r for r in by_arm.get(a, []) if r['level'] == lvl], uncond)} |")

    # paired feature contrasts on the funnel (unconditional; e2_funnel pairs on prompt_id and
    # majority-votes reps -> feed one aggregated record per (arm, prompt))
    def per_prompt(arm):
        byp = defaultdict(list)
        for r in by_arm.get(arm, []):
            if r.get("dot_unusable"):
                continue
            byp[r["pid"]].append(F.normalize(r["stages"]))
        out = []
        for pid, sts in byp.items():
            out.append({"prompt_id": pid, "level": GOLD[pid]["level"], "arm": arm,
                        "stages": {s: sum(x[s] for x in sts) > len(sts) / 2 for s in STAGES}})
        return out

    L += ["", "## Paired feature contrasts (per-prompt majority over reps; McNemar)", "",
          "| contrast | OFF arm | ON arm | Δ drc_clean | p (McNemar) | Δ sim_success | p | Δ routing_ok | p |", "|---|---|---|---|---|---|---|---|---|"]
    for fam, rows in (("additive", ADDITIVE), ("leave-one-out", LOO)):
        for feat, off, on in rows:
            if off not in by_arm or on not in by_arm:
                continue
            cmp = F.compare_arms(per_prompt(off), per_prompt(on))
            def g(s, k):
                b = cmp["by_stage"][s]
                return f"{b['delta']:+.2f}" if k == "d" else f"{b['mcnemar']['p_value']:.3f}"
            L.append(f"| {fam} {feat} | {off} | {on} | {g('drc_clean','d')} | {g('drc_clean','p')} | "
                     f"{g('sim_success','d')} | {g('sim_success','p')} | {g('routing_ok','d')} | {g('routing_ok','p')} |")

    # attribution
    L += ["", "## DRC attribution: library cells that violate DRC at their own defaults", "",
          f"Intrinsically violating cells (from `_pdk_component_drc.md`): {sorted(dirty)}", "",
          "| configuration | rebuilt reps | contain a violating cell | drc_clean, all | drc_clean, designs free of violating cells | violation categories (all reps) |",
          "|---|---|---|---|---|---|"]
    for a in ARMS:
        rs = [r for r in by_arm.get(a, []) if r.get("recon") == "ok"]
        clean = [F.normalize(r["stages"])["drc_clean"] for r in rs]
        free = [F.normalize(r["stages"])["drc_clean"] for r in rs if not r["has_dirty_cell"]]
        dc = Counter()
        for r in rs:
            for c in r.get("dirty_cells", []):
                dc[c] += 1
        L.append(f"| {a} | {len(rs)} | {rate([r['has_dirty_cell'] for r in rs])} | {rate(clean)} | {rate(free)} | {dict(dc)} |")
    L += ["", "## SAX attribution", "", "| configuration | reps reaching routing_ok | sim_success | sim failures that are closed circuits (no top-level port) | missing-model failures |", "|---|---|---|---|---|"]
    for a in ARMS:
        rs = [r for r in by_arm.get(a, []) if F.normalize(r["stages"])["routing_ok"]]
        fails = [r for r in rs if not F.normalize(r["stages"])["sim_success"]]
        mm = [r for r in fails if any(not str(m).startswith("SAX_BUILD_ERROR") and not str(m).startswith("Unnamed")
                                      for m in ((r.get("diag") or {}).get("missing_models") or []))]
        L.append(f"| {a} | {len(rs)} | {rate([F.normalize(r['stages'])['sim_success'] for r in rs])} | "
                 f"{rate([r['sim_no_ports'] for r in fails])} | {len(mm)} |")

    # instantiate-failure attribution (GDS build exceptions -> offending library cell)
    def _err_cell(r):
        msg = (r.get("diag") or {}).get("layout_error") or ""
        mods = set((r.get("diag") or {}).get("modules", {}).values())
        if "wdm_mzi1x4" in msg or ("route_width must be provided" in msg and "wdm_mzi1x4" in mods):
            return "wdm_mzi1x4 (route_single ValueError)"
        if "More than two connected optical ports" in msg:
            import re as _re
            # message lists the offending ports; an instance port (e.g. 'C6,o2') means the DESIGN
            # wired that port into more than one link; cell-internal names only means the cell itself
            if _re.search(r"\['C\d+,o\d+'", msg):
                return "design: a component port wired into >1 link"
            if "mzi_2x2_pn_diode" in mods:
                return "mzi_2x2_pn_diode (cell-internal >2 connected ports)"
        return "other: " + msg[:80]
    L += ["", "## Instantiate failures (GDS build exceptions), attributed to library cells", "",
          "| configuration | layout_error reps | by cause |", "|---|---|---|"]
    for a in ARMS:
        rs = [r for r in by_arm.get(a, []) if r.get("recon") == "layout_error"]
        L.append(f"| {a} | {len(rs)} | {dict(Counter(_err_cell(r) for r in rs))} |")

    L += ["", "## Rebuild failures (stored DOT could not be turned into a netlist)", "",
          "| configuration | rebuild_failed reps | by cause |", "|---|---|---|"]
    for a in ARMS:
        rs = [r for r in by_arm.get(a, []) if r.get("recon") == "rebuild_failed"]
        L.append(f"| {a} | {len(rs)} | {dict(Counter(((r.get('diag') or {}).get('rebuild_error') or '')[:60] for r in rs))} |")

    # correctness x manufacturability cross-tab
    L += ["", "## Topology correctness (edgeF1 = 1) vs layout outcome, rebuilt reps", "",
          "| configuration | edgeF1=1 & drc_clean | edgeF1=1 & not clean | edgeF1<1 & drc_clean | edgeF1<1 & not clean |", "|---|---|---|---|---|"]
    for a in ARMS:
        rs = [r for r in by_arm.get(a, []) if r.get("recon") == "ok" and r.get("edgeF1") is not None]
        c = Counter((r["edgeF1"] == 1.0, F.normalize(r["stages"])["drc_clean"]) for r in rs)
        L.append(f"| {a} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} |")

    (RES / f"_layout_{SOURCE}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

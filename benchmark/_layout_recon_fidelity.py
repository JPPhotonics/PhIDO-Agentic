"""Fidelity check for ``_layout_from_dot.py`` against funnel-era pipeline artifacts.

The July E2 funnel runs (``results/e2_k3_artifacts``) saved the pipeline's OWN ``gf_netlist_yaml``
plus the funnel stages it reached at the time. For N of them this script checks:

  (1) ENV REPLAY   original netlist -> run_layout_simulation (this venv) -> stages, vs the stored
                   stages. Tests that the layout/SAX/DRC tail in the .venv-layout environment
                   reproduces the July results (kfactory 0.21.7 era).
  (2) REBUILD      original netlist -> the app-format DOT it implies (circuit_dsl_to_dot + edges,
                   labels carrying "(module)") -> _layout_from_dot.rebuild_netlist -> netlist'.
                   Structural diff vs the original: instances/modules, port config, route links,
                   placements, top-level ports, and settings (settings are EXPECTED to differ
                   wherever the original run applied LLM parameter overrides).
  (3) STAGE MATCH  netlist' -> stages, vs (1). Tests that the reconstruction, at default
                   parameters, lands in the same funnel stage as the pipeline's own netlist.

Usage:  N=30 PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv-layout/bin/python benchmark/_layout_recon_fidelity.py
Writes results/_layout_recon_fidelity.{json,md}.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

import _layout_from_dot as LD  # noqa: E402
import e2_funnel as F  # noqa: E402
import e2_runner as R  # noqa: E402

N = int(os.getenv("N", "30"))
ART = ROOT / "benchmark" / "results" / "e2_k3_artifacts"
OUT = ROOT / "benchmark" / "results" / "_layout_recon_fidelity"


def netlist_to_dot(net: dict) -> str | None:
    """Emit the DOT the app would have produced for this netlist (record nodes + port edges)."""
    from mcp_servers.pipeline_orchestrator import _authoritative_port_config
    from mcp_servers.schematic_builder_server import circuit_dsl_to_dot

    nodes = {}
    for nid, inst in (net.get("instances") or {}).items():
        mod = inst.get("component")
        ports = _authoritative_port_config(mod)
        if not ports:
            return None  # cannot express this module in the app's DOT format
        nodes[nid] = {"component": mod, "properties": {"ports": ports},
                      "label": f"{nid}: X\\n({mod})"}
    dsl = {"doc": {"name": "fidelity"}, "nodes": nodes}
    dot = circuit_dsl_to_dot(json.dumps(dsl))
    if dot.startswith("{"):
        return None
    edge_lines = []
    for rd in (net.get("routes") or {}).values():
        for a, b in (rd.get("links") or {}).items():
            (an, ap), (bn, bp) = a.split(","), b.split(",")
            edge_lines.append(f"  {an.strip()}:{ap.strip()} -- {bn.strip()}:{bp.strip()};")
    return dot.rstrip().rstrip("}").rstrip() + "\n" + "\n".join(edge_lines) + "\n}"


def _links(net: dict) -> set[tuple[str, str]]:
    out = set()
    for rd in (net.get("routes") or {}).values():
        for a, b in (rd.get("links") or {}).items():
            out.add(tuple(sorted((a.replace(" ", ""), b.replace(" ", "")))))
    return out


def structural_diff(orig: dict, recon: dict) -> dict:
    oi, ri = orig.get("instances") or {}, recon.get("instances") or {}
    d: dict = {}
    d["same_instance_ids"] = set(oi) == set(ri)
    d["same_modules"] = all(oi[k].get("component") == ri.get(k, {}).get("component") for k in oi)
    d["same_port_cfg"] = all((oi[k].get("info") or {}).get("ports") == (ri.get(k, {}).get("info") or {}).get("ports")
                             for k in oi)
    d["same_links"] = _links(orig) == _links(recon)
    op, rp = orig.get("placements") or {}, recon.get("placements") or {}
    d["same_placements"] = all(
        k in rp and all(abs(float(op[k].get(a, 0)) - float(rp[k].get(a, 0))) < 1e-6 for a in ("x", "y", "rotation"))
        for k in op) and set(op) == set(rp)
    d["same_ports"] = (orig.get("ports") or {}) == (recon.get("ports") or {})
    diffs = {}
    for k in oi:
        os_, rs_ = oi[k].get("settings") or {}, (ri.get(k) or {}).get("settings") or {}
        changed = {p: (os_.get(p), rs_.get(p)) for p in set(os_) | set(rs_) if os_.get(p) != rs_.get(p)}
        if changed:
            diffs[k] = changed
    d["settings_diff"] = diffs
    d["n_settings_overridden_instances"] = len(diffs)
    return d


def stages_of(netlist_yaml: str) -> dict:
    ls, err = LD.layout_sim(netlist_yaml)
    st = R.funnel_from_results({"gf_netlist_yaml": netlist_yaml}, ls)
    return {"stages": st, "layout_error": err,
            "missing_models": (ls or {}).get("missing_models"),
            "drc_violations": (ls or {}).get("drc_violations")}


def main() -> None:
    LD._patch_build_dir("fidelity")
    files = sorted(glob.glob(str(ART / "*.json")))
    # take a spread: alternate arms, skip runs with no saved netlist
    picked = []
    for f in files:
        y = Path(f).with_suffix("").with_suffix(".netlist.yml")
        if not y.exists():
            y = Path(str(f)[:-5] + ".netlist.yml")
        if y.exists():
            picked.append((f, y))
    step = max(1, len(picked) // N)
    picked = picked[::step][:N]
    rows = []
    for f, y in picked:
        meta = json.load(open(f))
        orig_yaml = y.read_text()
        orig = yaml.safe_load(orig_yaml) or {}
        row = {"file": Path(f).name, "arm": meta.get("arm"), "prompt_id": meta.get("prompt_id"),
               "stored_stages": meta.get("stages")}
        try:
            row["replay"] = stages_of(orig_yaml)
        except Exception as e:  # noqa: BLE001
            row["replay"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        dot = netlist_to_dot(orig)
        if dot is None:
            row["rebuild"] = {"skipped": "module without NxM docstring ports"}
        else:
            recon_yaml, diag = LD.rebuild_netlist(dot)
            if recon_yaml is None:
                row["rebuild"] = {"error": diag.get("rebuild_error")}
            else:
                recon = yaml.safe_load(recon_yaml) or {}
                row["rebuild"] = structural_diff(orig, recon)
                try:
                    row["recon_stages"] = stages_of(recon_yaml)
                except Exception as e:  # noqa: BLE001
                    row["recon_stages"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        rows.append(row)
        OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))  # incremental
        rs = row.get("replay", {}).get("stages", {})
        cs = row.get("recon_stages", {}).get("stages", {})
        print(f"{row['arm']:9} {row['prompt_id']:6} stored={F.furthest_stage(row['stored_stages'] or {}):12} "
              f"replay={F.furthest_stage(rs) if rs else 'ERR':12} recon={F.furthest_stage(cs) if cs else '-':12} "
              f"struct={ {k: v for k, v in row.get('rebuild', {}).items() if k.startswith('same_')} }",
              flush=True)
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))

    # summary
    n = len(rows)
    rep_ok = [r for r in rows if "stages" in r.get("replay", {})]
    same_stage = sum(F.furthest_stage(r["stored_stages"]) == F.furthest_stage(r["replay"]["stages"]) for r in rep_ok)
    same_drc = sum(bool(r["stored_stages"].get("drc_clean")) == bool(r["replay"]["stages"].get("drc_clean")) for r in rep_ok)
    rb = [r for r in rows if "same_links" in r.get("rebuild", {})]
    keys = ["same_instance_ids", "same_modules", "same_port_cfg", "same_links", "same_placements", "same_ports"]
    lines = [f"# Reconstruction fidelity (N={n} funnel-era artifacts)", "",
             "## (1) Environment replay: original netlist through this venv's layout tail",
             f"- replayed without error: {len(rep_ok)}/{n}",
             f"- same furthest funnel stage as stored (July): {same_stage}/{len(rep_ok)}",
             f"- same drc_clean verdict as stored: {same_drc}/{len(rep_ok)}", "",
             "## (2) DOT-based rebuild vs original netlist (structural)",
             f"- rebuilt: {len(rb)}/{n}"]
    for k in keys:
        lines.append(f"- {k}: {sum(bool(r['rebuild'][k]) for r in rb)}/{len(rb)}")
    ov = sum(r["rebuild"]["n_settings_overridden_instances"] > 0 for r in rb)
    lines += [f"- netlists where the original carried non-default settings (LLM overrides): {ov}/{len(rb)}", ""]
    both = [r for r in rb if "stages" in r.get("recon_stages", {}) and "stages" in r.get("replay", {})]
    lines += ["## (3) Reconstruction stages vs replay stages (same venv)",
              f"- same furthest stage: {sum(F.furthest_stage(r['recon_stages']['stages']) == F.furthest_stage(r['replay']['stages']) for r in both)}/{len(both)}",
              f"- same drc_clean: {sum(bool(r['recon_stages']['stages'].get('drc_clean')) == bool(r['replay']['stages'].get('drc_clean')) for r in both)}/{len(both)}",
              ""]
    mism = [r for r in both if F.furthest_stage(r["recon_stages"]["stages"]) != F.furthest_stage(r["replay"]["stages"])]
    if mism:
        lines.append("### stage mismatches (recon vs replay)")
        for r in mism:
            lines.append(f"- {r['file']}: replay={F.furthest_stage(r['replay']['stages'])} recon={F.furthest_stage(r['recon_stages']['stages'])} "
                         f"settings_diff_instances={list(r['rebuild']['settings_diff'])}")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

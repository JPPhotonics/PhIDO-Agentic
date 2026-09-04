"""Per-component DRC baseline: is each DesignLibrary cell DRC-clean at its own default settings?

Interpreting a circuit-level ``drc_clean`` verdict requires knowing which violations a design
inherits from the library cells themselves (a grating coupler whose default tooth width is below
the Si minimum width fails DRC in every circuit that contains it, regardless of the design). This
script instantiates every top-level PDK module standalone, writes its GDS, runs the same KLayout
DRC script the pipeline uses, and tabulates violations by category.

Usage: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv-layout/bin/python benchmark/_pdk_component_drc.py
Writes results/_pdk_component_drc.{json,md}.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "benchmark" / "results" / "_pdk_component_drc"
BUILD = ROOT / "benchmark" / "results" / "_layout_build_pdk"
BUILD.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from mcp_servers import layout_sim_server as L
    L.BUILD_DIR = BUILD
    L._ensure_pdk()
    from PhotonicsAI.Photon.DemoPDK import list_of_cnames

    rows = []
    for mod in sorted(list_of_cnames):
        row = {"module": mod}
        try:
            c = L._pdk.get_component(mod)
            gds = BUILD / f"{mod}.gds"
            c.write_gds(str(gds))
            row["dx_um"], row["dy_um"] = float(c.dxsize), float(c.dysize)
            r = L.run_drc_check(str(gds))
            row.update({k: r.get(k) for k in ("drc_ran", "drc_clean", "violations", "error")})
            if r.get("report_path") and Path(r["report_path"]).exists():
                t = ET.parse(r["report_path"])
                row["by_category"] = dict(Counter(i.findtext("category") for i in t.iter("item")))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
        print(f"{mod:36} clean={row.get('drc_clean')} viol={row.get('violations')} "
              f"{row.get('by_category', '')} {row.get('error') or ''}", flush=True)
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))
    lines = ["# Per-component DRC at default settings", "",
             "| module | dx (um) | dy (um) | DRC clean | violations | by category |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['module']} | {r.get('dx_um', ''):.6} | {r.get('dy_um', ''):.6} | {r.get('drc_clean')} | "
                     f"{r.get('violations')} | {r.get('by_category', r.get('error', ''))} |"
                     if isinstance(r.get("dx_um"), float) else
                     f"| {r['module']} | | | {r.get('drc_clean')} | {r.get('violations')} | {r.get('error', '')} |")
    clean = [r["module"] for r in rows if r.get("drc_clean") is True]
    dirty = [r["module"] for r in rows if r.get("drc_clean") is False]
    lines += ["", f"- clean at defaults ({len(clean)}): {clean}", f"- violating at defaults ({len(dirty)}): {dirty}"]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))


if __name__ == "__main__":
    main()

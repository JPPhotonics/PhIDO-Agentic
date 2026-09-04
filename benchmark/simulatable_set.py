"""Compute the authoritative *simulatable* PDK subset.

The E2 ``models`` stage fails whenever a built component's required SAX leaf models
are not all present in ``DemoPDK.all_models``. The selectable pool (DesignLibrary, 34
modules) is therefore NOT the same as the set we can actually simulate. This script
resolves the true set by, for each module:

  1. instantiating the ``@gf.cell`` component (default args),
  2. expanding it to its recursive gdsfactory netlist,
  3. computing ``sax.get_required_circuit_models`` (the LEAF models the circuit needs),
  4. checking ``required ⊆ all_models`` and that ``sax.circuit`` actually builds.

A module is ``simulatable`` iff it builds AND every required leaf model is covered.
Writes ``benchmark/results/simulatable_set.json`` — the buildable+simulatable whitelist
the retrieval/selection enforcement should constrain to.

Run:
  PYTHONPATH=/home/tony/PhIDOv1/wt-graphrag \
    /home/tony/PhIDOv1/wt-graphrag/.venv/bin/python benchmark/simulatable_set.py
"""

from __future__ import annotations

import json
import pathlib

import sax

from PhotonicsAI.Photon.DemoPDK import all_models, cells

OUT = pathlib.Path(__file__).parent / "results" / "simulatable_set.json"


def probe(name: str) -> dict:
    """Build one module and report its leaf-model coverage."""
    rec: dict = {"module": name}
    try:
        comp = cells[name]()
    except Exception as e:  # noqa: BLE001 — record, don't crash the sweep
        rec.update(buildable=False, error=f"instantiate: {type(e).__name__}: {e}"[:200])
        return rec
    rec["buildable"] = True
    try:
        recnet = comp.get_netlist(recursive=True)
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"netlist: {type(e).__name__}: {e}"[:200], required=[], missing=[])
        return rec
    try:
        required = sorted(sax.get_required_circuit_models(recnet))
    except StopIteration:
        # Leaf primitive (no sub-instances): it contributes its OWN model. By construction
        # all_models is the union of every module's get_model(), so a leaf's own keys are
        # always covered — resolve them directly instead of via circuit expansion.
        import importlib

        gm = importlib.import_module(
            f"PhotonicsAI.KnowledgeBase.DesignLibrary.{name}"
        ).get_model()
        required = sorted(gm.keys())
        rec["leaf"] = True
    missing = [m for m in required if m not in all_models]
    rec["required"] = required
    rec["missing"] = missing
    try:
        sax.circuit(recnet, all_models, backend="default")
        rec["sax_builds"] = True
    except Exception as e:  # noqa: BLE001 — leaf-only netlists (no instances) can't form a circuit
        rec["sax_builds"] = False
        rec["sax_error"] = f"{type(e).__name__}: {e}"[:200]
    # Simulatable in a circuit = it instantiates AND every leaf model it introduces is in
    # all_models. sax_builds is required only for composites (a bare leaf has no circuit to form).
    rec["simulatable"] = rec["buildable"] and not missing and (
        rec.get("leaf", False) or rec.get("sax_builds", False)
    )
    return rec


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    rows = [probe(n) for n in sorted(cells)]
    sim = sorted(r["module"] for r in rows if r.get("simulatable"))
    notsim = sorted(r["module"] for r in rows if not r.get("simulatable"))
    payload = {
        "n_selectable": len(rows),
        "n_simulatable": len(sim),
        "simulatable": sim,
        "not_simulatable": notsim,
        "all_models_keys": sorted(all_models),
        "per_module": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"selectable={len(rows)}  simulatable={len(sim)}  not_simulatable={len(notsim)}")
    print("\nSIMULATABLE:", sim)
    print("\nNOT simulatable (would die at E2 'models'):")
    for r in rows:
        if not r.get("simulatable"):
            why = r.get("error") or (f"missing leaf models: {r['missing']}" if r.get("missing")
                                      else r.get("sax_error", "?"))
            print(f"  {r['module']:32} {why}")
    print(f"\n[ok] -> {OUT}")


if __name__ == "__main__":
    main()

"""Generate an expanded E1 query set from the 103 Testbench design prompts (plan path 1).

For each real design prompt, an LLM extracts the component sub-intents as (phrase, type),
where `type` is constrained to a FIXED functional vocabulary (TYPE_TO_MODULES below). Gold is
then assigned PURELY from that dictionary — gold is NOT the LLM's module pick and NOT the
pipeline's selection, so it stays arm-independent. The only LLM-dependent step is phrase->type
classification, which is cached to results/e1_testbench_extraction.json for audit.

TYPE_TO_MODULES is the domain gold table (covers the 34-cell DemoPDK). It is a **DRAFT** and the
authoritative artifact the Poon group must sign off on; overlaps (e.g. the several MZI variants)
and judgment calls (VOA->pin diode, modulator families) are intentional and flagged for review.

Output: benchmark/e1_queries_testbench.json  (consumed by run_e1_retrieval.py <file>).
Run:    CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/build_e1_queries_from_testbench.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
from PhotonicsAI.Photon import llm_api  # noqa: E402

TESTBENCH = HERE.parent / "Testbench_modified.csv"
OUT = HERE / "e1_queries_testbench.json"
CACHE = HERE / "results" / "e1_testbench_extraction.json"

# ---- DRAFT domain gold table: functional type -> acceptable PDK module(s) (34-cell DemoPDK) ----
TYPE_TO_MODULES: dict[str, list[str]] = {
    "directional_coupler": ["_directional_coupler", "_directional_coupler_adiabatic"],
    "mmi_1x2": ["_mmi1x2"],
    "mmi_2x2": ["_mmi2x2"],
    "mzi_2x2": ["mzi_2x2_heater_tin_cband", "mzi_2x2_pindiode_cband", "mzi_2x2_pn_diode"],
    "mzi_1x2": ["mzi_1x2_pindiode_cband"],
    "mzi_1x1": ["mzi_1x1_heater_doped_si_cband", "mzi_1x1_pindiode_cband", "mzi1"],
    "mach_zehnder_modulator": ["tw_mzm", "mzi_2x2_pn_diode", "mzi_2x2_pindiode_cband",
                               "mzi_1x2_pindiode_cband", "mzi_1x1_pindiode_cband"],
    "thermo_optic_phase_shifter": ["heater_tin_cband", "heater_doped_si_cband"],
    "phase_shifter": ["heater_tin_cband", "heater_doped_si_cband", "pndiode", "pindiode_cband"],
    "variable_optical_attenuator": ["pindiode_cband"],
    "grating_coupler": ["_gc"],
    "edge_coupler": ["edge_coupler"],
    "ring_resonator": ["mrr_1x1", "mrr_1x1_heater_tin", "mrr_2x2", "coupler_ring"],
    "microring_modulator": ["mrm_1x1_pndiode"],
    "wavelength_division_multiplexer": ["wdm_mzi1x4"],
    "crossing": ["crossing"],
    "photodetector": ["photodetector"],
    "laser": ["laser"],
    "polarization_splitter_rotator": ["polarization_splitter_rotator"],
    "waveguide_bend": ["bend_euler", "_bend_s", "_bezier_curve"],
    "straight_waveguide": ["straight"],
    "taper_mode_converter": ["mode_converter"],
}
VOCAB = sorted(TYPE_TO_MODULES)


class Component(BaseModel):
    phrase: str   # short verbatim/paraphrased description of ONE component need from the prompt
    type: str     # one of VOCAB, or "other"


class Extraction(BaseModel):
    components: List[Component]


SYS = (
    "You decompose a photonic circuit design request into its individual component needs. "
    "For each distinct component the request asks for, output a short phrase describing that "
    "component's need (as a user would say it) and classify it into exactly one type from the "
    "allowed list. If a component does not fit any allowed type, use type 'other'. Do not invent "
    "components not implied by the request."
)


def extract(prompt: str) -> List[Component]:
    user = (f"Allowed types: {', '.join(VOCAB)}, other\n\n"
            f"Design request:\n{prompt}\n\n"
            "Return JSON {\"components\": [{\"phrase\": ..., \"type\": ...}, ...]}.")
    try:
        res = llm_api.callgoogle_pydantic(user, SYS, Extraction)
        return res.components if res and hasattr(res, "components") else []
    except Exception as e:
        print(f"  [warn] extraction failed: {e}")
        return []


def main() -> None:
    prompts = [r[0] for r in csv.reader(open(TESTBENCH, newline="")) if r and r[0].strip()]
    print(f"Testbench prompts: {len(prompts)}")

    cache, queries = [], []
    seen_ids: dict[str, int] = {}
    for pi, prompt in enumerate(prompts):
        comps = extract(prompt)
        rec = {"prompt_index": pi, "prompt": prompt,
               "components": [{"phrase": c.phrase, "type": c.type} for c in comps]}
        cache.append(rec)
        for c in comps:
            gold = TYPE_TO_MODULES.get(c.type, [])
            if not gold:                       # 'other' or unknown type -> not gradable
                continue
            base = c.type
            seen_ids[base] = seen_ids.get(base, 0) + 1
            queries.append({
                "id": f"tb{pi:03d}_{base}_{seen_ids[base]}",
                "kind": "intent",
                "query": c.phrase,
                "gold": gold,
                "type": c.type,
                "source_prompt_index": pi,
            })
        if (pi + 1) % 20 == 0:
            print(f"  ...{pi+1}/{len(prompts)} prompts, {len(queries)} gradable queries so far")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    OUT.write_text(json.dumps({
        "_about": "Expanded E1 queries from Testbench prompts. Query=LLM-extracted component phrase; "
                  "gold=TYPE_TO_MODULES[type] (arm-independent). DRAFT — audit the type dict + the "
                  "phrase->type assignments in results/e1_testbench_extraction.json before final use.",
        "queries": queries,
    }, indent=2))
    n_other = sum(1 for r in cache for c in r["components"] if c["type"] not in TYPE_TO_MODULES)
    print(f"\n[done] {len(queries)} gradable queries -> {OUT}")
    print(f"       ({n_other} component mentions typed 'other'/unknown and dropped)")
    print(f"       extraction cache -> {CACHE}")


if __name__ == "__main__":
    main()

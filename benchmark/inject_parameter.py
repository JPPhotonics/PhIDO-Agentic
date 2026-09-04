"""B5 — parameter-error taxonomy + synthetic injector.

Takes known-valid component parameter sets (bounded by the enumerable PDK) and produces
labeled-invalid variants, one taxonomy-spanning mutation each. Feeds the AR (Automated
Reasoning) parameter-gate's per-error-class catch-rate eval via ``gate_eval``.

A spec per param: ``{"type": float, "min": .., "max": .., "grid": .., "required": True}``.
Pairwise/relational constraints live in ``constraints`` as predicates over the param dict.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ParamSpec:
    specs: dict[str, dict]                                  # param -> {type,min,max,grid,required}
    constraints: list[tuple[str, Callable[[dict], bool]]] = field(default_factory=list)  # (name, ok?)


# Minimal PDK-like specs (extend from the real DesignLibrary Args docstrings later).
SPECS: dict[str, ParamSpec] = {
    "ring": ParamSpec(
        specs={
            "radius_um": {"type": float, "min": 5.0, "max": 100.0, "grid": 0.001, "required": True},
            "gap_um": {"type": float, "min": 0.1, "max": 2.0, "required": True},
            "coupling_length_um": {"type": float, "min": 0.0, "max": 50.0, "required": True},
        },
        # a narrow gap with a long coupler over-couples -> physically incompatible pairing
        constraints=[("gap_coupling", lambda p: not (p["gap_um"] < 0.2 and p["coupling_length_um"] > 30))],
    ),
    "straight": ParamSpec(specs={"length_um": {"type": float, "min": 0.0, "max": 1000.0, "required": True}}),
    "mmi": ParamSpec(specs={"width_um": {"type": float, "min": 0.5, "max": 10.0, "grid": 0.005, "required": True}}),
}

VALID_SETS: dict[str, dict] = {
    "ring": {"radius_um": 10.0, "gap_um": 0.3, "coupling_length_um": 5.0},
    "straight": {"length_um": 100.0},
    "mmi": {"width_um": 2.0},
}


# ---- injectors: each returns (mutated params) or None if inapplicable ----
def _out_of_range(params: dict, spec: ParamSpec, rng: random.Random) -> dict | None:
    cands = [k for k, s in spec.specs.items() if "max" in s]
    if not cands:
        return None
    p = copy.deepcopy(params)
    k = rng.choice(cands)
    p[k] = spec.specs[k]["max"] * 10  # well above the bound
    return p


def _wrong_units(params: dict, spec: ParamSpec, rng: random.Random) -> dict | None:
    cands = [k for k in params if k.endswith("_um")]
    if not cands:
        return None
    p = copy.deepcopy(params)
    k = rng.choice(cands)
    p[k] = params[k] * 1000  # nm value entered into a µm field
    return p


def _type_mismatch(params: dict, spec: ParamSpec, rng: random.Random) -> dict | None:
    k = rng.choice(list(params))
    p = copy.deepcopy(params)
    p[k] = f"{params[k]}um"  # string where a float is required
    return p


def _missing_required(params: dict, spec: ParamSpec, rng: random.Random) -> dict | None:
    req = [k for k, s in spec.specs.items() if s.get("required")]
    if not req:
        return None
    p = copy.deepcopy(params)
    del p[rng.choice(req)]
    return p


def _grid_violation(params: dict, spec: ParamSpec, rng: random.Random) -> dict | None:
    cands = [k for k, s in spec.specs.items() if "grid" in s and k in params]
    if not cands:
        return None
    p = copy.deepcopy(params)
    k = rng.choice(cands)
    p[k] = params[k] + spec.specs[k]["grid"] / 3  # off the fab grid
    return p


def _incompatible_pairing(params: dict, spec: ParamSpec, rng: random.Random) -> dict | None:
    if not spec.constraints:
        return None
    p = copy.deepcopy(params)
    # force the ring gap/coupling constraint to fail
    if "gap_um" in p and "coupling_length_um" in p:
        p["gap_um"], p["coupling_length_um"] = 0.15, 40.0
        return p
    return None


PARAM_INJECTORS = {
    "out_of_range": _out_of_range,
    "wrong_units": _wrong_units,
    "type_mismatch": _type_mismatch,
    "missing_required": _missing_required,
    "grid_violation": _grid_violation,
    "incompatible_pairing": _incompatible_pairing,
}


def inject_invalids(classes: list[str] | None = None, seed: int = 0) -> list[dict]:
    """Labeled-invalid parameter records over all components × applicable error classes."""
    classes = classes if classes is not None else list(PARAM_INJECTORS)
    rng = random.Random(seed)
    out: list[dict] = []
    for comp, params in VALID_SETS.items():
        for cls in classes:
            mutated = PARAM_INJECTORS[cls](params, SPECS[comp], rng)
            if mutated is not None:
                out.append({"id": f"{comp}-{cls}", "component": comp, "params": mutated,
                            "label": "invalid", "error_class": cls})
    return out


def valid_records() -> list[dict]:
    return [{"id": f"valid-{c}", "component": c, "params": copy.deepcopy(p), "label": "valid"}
            for c, p in VALID_SETS.items()]

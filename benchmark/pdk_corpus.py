"""GDSFactory Phase -1 — build the E1 candidate corpus from the generic PDK.

The E1 retrieval corpus is two tiers (KB_BENCHMARK_PLAN.md §4 E1):
* **distractor tier** — the ~260 generic gdsfactory cells, with **signature-grounded**
  descriptions (derived deterministically from each factory's name + parameters, NOT
  free-form LLM invention — reproducible and non-circular);
* **modeled tier** — the DesignLibrary components (retrieval targets, E2-capable), parsed
  from their structured docstrings.

The merged corpus is exactly the candidate-dict list that ``e1_retrieval`` consumes:
``{"id", "name", "description", "function", "tier", "params"}`` — identical across all
three arms, so it's shared infra, not the treatment.
"""

from __future__ import annotations

import inspect
from pathlib import Path

# coarse function tag inferred from the cell name (keeps descriptions grounded + searchable)
_KIND = [
    ("mmi", "splits or combines optical power via multimode interference"),
    ("directional_coupler", "couples light between two waveguides"),
    ("coupler", "couples optical power between waveguides"),
    ("ring", "resonates light for wavelength filtering"),
    ("mzi", "interferes two paths for modulation or filtering"),
    ("grating_coupler", "couples light between fiber and chip"),
    ("gc", "couples light between fiber and chip"),
    ("bend", "routes a waveguide around a turn"),
    ("taper", "transitions between waveguide widths"),
    ("straight", "guides light along a straight waveguide"),
    ("crossing", "lets two waveguides cross with low coupling"),
    ("splitter", "splits optical power"),
    ("heater", "applies thermo-optic phase tuning"),
    ("ring_single", "single-ring wavelength filter"),
    ("pad", "provides an electrical contact pad"),
    ("via", "connects metal layers"),
    ("spiral", "provides a compact long delay line"),
]


def _function_of(name: str) -> str:
    low = name.lower()
    for key, desc in _KIND:
        if key in low:
            return desc
    return "a photonic integrated component"


def _signature_params(factory) -> list[str]:
    try:
        return [p for p in inspect.signature(factory).parameters
                if p not in ("self", "args", "kwargs")]
    except (ValueError, TypeError):
        return []


def _describe(name: str, params: list[str]) -> str:
    pretty = name.replace("_", " ").strip()
    func = _function_of(name)
    tail = f" Parameters: {', '.join(params[:8])}." if params else ""
    return f"{pretty}: {func}.{tail}"


def build_distractor_candidates(limit: int | None = None, include_ports: bool = False) -> list[dict]:
    """Enumerate the generic gdsfactory PDK into signature-grounded candidate dicts."""
    import gdsfactory as gf

    pdk = gf.get_active_pdk()
    out: list[dict] = []
    for name, factory in sorted(pdk.cells.items()):
        params = _signature_params(factory)
        cand = {
            "id": name, "name": name.replace("_", " "),
            "description": _describe(name, params),
            "function": _function_of(name), "tier": "distractor", "params": params,
        }
        if include_ports:
            try:
                cand["ports"] = list(factory().ports.keys())   # builds geometry; may fail
            except Exception:                                  # noqa: BLE001
                cand["ports"] = []
        out.append(cand)
        if limit and len(out) >= limit:
            break
    return out


def build_modeled_candidates(design_library_dir: Path | None = None) -> list[dict]:
    """Parse DesignLibrary docstrings into candidate dicts (retrieval targets, modeled tier).

    Best-effort: uses utils.search_directory_for_docstrings if importable; else returns [].
    """
    try:
        from PhotonicsAI.Photon import utils  # type: ignore
        root = design_library_dir or Path(utils.__file__).resolve().parents[1] / "KnowledgeBase" / "DesignLibrary"
        docs = utils.search_directory_for_docstrings(str(root))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] modeled tier unavailable ({type(exc).__name__}); distractor-only corpus")
        return []
    out = []
    for name, doc in (docs.items() if isinstance(docs, dict) else []):
        text = doc if isinstance(doc, str) else str(doc)
        out.append({"id": name, "name": name.replace("_", " "), "description": text,
                    "function": "", "tier": "modeled", "params": []})
    return out


def build_corpus(include_ports: bool = False) -> list[dict]:
    """Merged E1 candidate corpus: modeled targets + generic distractors (deduped by id)."""
    modeled = build_modeled_candidates()
    seen = {c["id"] for c in modeled}
    distractors = [c for c in build_distractor_candidates(include_ports=include_ports) if c["id"] not in seen]
    return modeled + distractors

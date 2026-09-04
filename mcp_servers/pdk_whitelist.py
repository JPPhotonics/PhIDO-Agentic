"""Authoritative buildable+simulatable PDK whitelist for selection enforcement.

The DesignLibrary exposes 34 modules, but only a subset is **selectable** (top-level, not
``_``-prefixed internal building blocks) AND **simulatable** (instantiates with default args and
every leaf SAX model it introduces is present in ``DemoPDK.all_models``). Selecting outside this
set is the asymmetry that let the agentic E2 arm die at ``instantiate``/``models`` on components
the rigid baseline could never pick. This module is the single source of truth that both the
retriever (``component_retrieval``) and the selection gate (``pipeline_orchestrator``) enforce.

The simulatable set is computed lazily and cached — building 34 cells + sax probing costs a few
seconds, once per process. If the probe itself errors (missing PDK, import failure), it degrades
to the selectable-but-unverified set so retrieval never hard-breaks. The logic mirrors
``benchmark/simulatable_set.probe`` inline (production code must not import from ``benchmark/``).
"""

from __future__ import annotations

from functools import lru_cache

from mcp_servers.pdk_catalog_server import CATALOG

_INTERNAL_PREFIX = "_"

# Passive photonic DEVICES that are ``_``-prefixed (so used as building blocks inside
# composite cells) but are ALSO legitimate standalone components a user can request —
# a bare MMI power splitter, a directional coupler, a grating coupler. The blanket
# ``_``-exclusion conflated "internal implementation detail" with "not a valid device"
# and made these unselectable, so any prompt asking for a bare passive splitter had
# nothing to map to → the correct pick (e.g. ``_mmi1x2`` for "a 1x2 power splitter")
# was unmapped by the selection gate → empty netlist. These all build and are leaf
# primitives whose own SAX model is in ``all_models`` (verified simulatable), and the
# b3_gold class_map treats them as valid standalone answers (_mmi1x2->MMI_1x2, etc.).
# NOT exposed: pure routing geometry (``_bend_s``, ``_bezier_curve``) — not devices.
_EXPOSED_PRIMITIVES = frozenset({
    "_mmi1x2", "_mmi2x2", "_directional_coupler",
    "_directional_coupler_adiabatic", "_gc",
})


def selectable_modules() -> frozenset[str]:
    """Top-level catalog modules + exposed passive-device primitives.

    Excludes ``_``-prefixed internal building blocks EXCEPT the passive photonic
    devices in ``_EXPOSED_PRIMITIVES`` (MMIs, directional couplers, grating coupler),
    which are valid standalone components despite being reused inside composite cells.
    """
    return frozenset(
        c["module_name"] for c in CATALOG
        if not c["module_name"].startswith(_INTERNAL_PREFIX)
        or c["module_name"] in _EXPOSED_PRIMITIVES
    )


@lru_cache(maxsize=1)
def simulatable_modules() -> frozenset[str]:
    """Selectable modules that also build and have full leaf-model coverage.

    A module qualifies iff it instantiates with default args and every leaf model its
    recursive netlist requires is in ``all_models`` (a leaf primitive trivially qualifies —
    it contributes its own model, which is in ``all_models`` by construction). Never returns
    empty: on total probe failure it falls back to ``selectable_modules()``.
    """
    sel = selectable_modules()
    try:
        import sax

        from PhotonicsAI.Photon.DemoPDK import all_models, cells
    except Exception:
        return sel  # PDK/sax unavailable — don't over-restrict, degrade to selectable
    ok: set[str] = set()
    for name in sel:
        try:
            recnet = cells[name]().get_netlist(recursive=True)
        except Exception:
            continue  # fails to instantiate/expand → would die at E2 'instantiate'
        try:
            required = sax.get_required_circuit_models(recnet)
        except StopIteration:
            ok.add(name)  # leaf primitive: own model is in all_models by construction
            continue
        except Exception:
            continue
        if all(m in all_models for m in required):
            ok.add(name)
    return frozenset(ok) or sel  # never empty


def is_simulatable(name: str) -> bool:
    """True iff ``name`` is in the buildable+simulatable whitelist."""
    return name in simulatable_modules()

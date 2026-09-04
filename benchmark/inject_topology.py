"""B4 — topology-error taxonomy + synthetic injector.

Takes known-valid circuit topologies and produces *labeled-invalid* variants by applying
one taxonomy-spanning mutation each — so the label is known by construction (no expert).
These feed the Clingo topology gate's per-error-class catch-rate eval via ``gate_eval``.

A ``Topology`` is an abstract netlist: component instances, port-to-port connections, and
each component type's port kinds (``opt_bi`` / ``opt_in`` / ``opt_out`` / ``elec``). It is
*not* a gdsfactory netlist — it's the minimal structure the gate's rules reason over.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

# Port kinds per component type (minimal fixtures; extend as the gate's scope grows).
COMPONENTS: dict[str, dict[str, str]] = {
    "gc": {"o1": "opt_bi"},                                   # grating coupler (chip I/O)
    "mmi1x2": {"o1": "opt_bi", "o2": "opt_bi", "o3": "opt_bi"},
    "straight": {"o1": "opt_bi", "o2": "opt_bi"},
    "mzi": {"o1": "opt_bi", "o2": "opt_bi"},
    "pd": {"o1": "opt_in", "e1": "elec"},                     # photodiode
    "heater": {"e1": "elec", "e2": "elec"},
}


@dataclass
class Topology:
    instances: dict[str, str]                       # name -> component type
    connections: list[tuple[str, str]]              # ("inst.port", "inst.port")
    external: dict[str, str] = field(default_factory=dict)

    def port_kind(self, ref: str) -> str | None:
        inst, _, port = ref.partition(".")
        ctype = self.instances.get(inst)
        return COMPONENTS.get(ctype, {}).get(port) if ctype else None


def seed_topologies() -> list[Topology]:
    """A couple of valid seeds. (Curated gold topologies from GETTING_STARTED plug in here later.)"""
    return [
        Topology(  # MZI between two grating couplers
            instances={"in_gc": "gc", "mzi": "mzi", "out_gc": "gc"},
            connections=[("in_gc.o1", "mzi.o1"), ("mzi.o2", "out_gc.o1")],
        ),
        Topology(  # 1x2 splitter feeding two straights
            instances={"in_gc": "gc", "spl": "mmi1x2", "s_top": "straight", "s_bot": "straight"},
            connections=[("in_gc.o1", "spl.o1"), ("spl.o2", "s_top.o1"), ("spl.o3", "s_bot.o1")],
        ),
    ]


# ---- injectors: each returns a mutated (invalid) copy, or None if inapplicable ----
def _dangling_port(t: Topology, rng: random.Random) -> Topology | None:
    if not t.connections:
        return None
    t = copy.deepcopy(t)
    t.connections.pop(rng.randrange(len(t.connections)))  # leaves both ports open
    return t


def _nonexistent_ref(t: Topology, rng: random.Random) -> Topology | None:
    if not t.connections:
        return None
    t = copy.deepcopy(t)
    i = rng.randrange(len(t.connections))
    a, _ = t.connections[i]
    t.connections[i] = ("ghost.o1", a)  # references an instance that does not exist
    return t


def _self_loop(t: Topology, rng: random.Random) -> Topology | None:
    inst = rng.choice(list(t.instances))
    port = next(iter(COMPONENTS[t.instances[inst]]))
    t = copy.deepcopy(t)
    t.connections.append((f"{inst}.{port}", f"{inst}.{port}"))  # port tied to itself
    return t


def _double_connect(t: Topology, rng: random.Random) -> Topology | None:
    if not t.connections:
        return None
    t = copy.deepcopy(t)
    t.connections.append(t.connections[rng.randrange(len(t.connections))])  # illegal fan-out
    return t


def _floating_instance(t: Topology, rng: random.Random) -> Topology | None:
    t = copy.deepcopy(t)
    t.instances["orphan"] = "straight"  # added but never connected
    return t


def _type_mismatch(t: Topology, rng: random.Random) -> Topology | None:
    opt_ports = [f"{i}.{p}" for i, c in t.instances.items() for p, k in COMPONENTS[c].items() if k.startswith("opt")]
    if not opt_ports:
        return None
    t = copy.deepcopy(t)
    t.instances["h"] = "heater"
    t.connections.append(("h.e1", rng.choice(opt_ports)))  # electrical tied to optical
    return t


TOPOLOGY_INJECTORS = {
    "dangling_port": _dangling_port,
    "nonexistent_ref": _nonexistent_ref,
    "self_loop": _self_loop,
    "double_connect": _double_connect,
    "floating_instance": _floating_instance,
    "type_mismatch": _type_mismatch,
}


def inject_invalids(
    seeds: list[Topology] | None = None,
    classes: list[str] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Produce labeled-invalid records (label known by construction) over all seeds×classes."""
    seeds = seeds if seeds is not None else seed_topologies()
    classes = classes if classes is not None else list(TOPOLOGY_INJECTORS)
    rng = random.Random(seed)
    out: list[dict] = []
    for si, base in enumerate(seeds):
        for cls in classes:
            mutated = TOPOLOGY_INJECTORS[cls](base, rng)
            if mutated is not None:
                out.append({"id": f"s{si}-{cls}", "topology": mutated, "label": "invalid", "error_class": cls})
    return out


def valid_records(seeds: list[Topology] | None = None) -> list[dict]:
    seeds = seeds if seeds is not None else seed_topologies()
    return [{"id": f"valid-{i}", "topology": t, "label": "valid"} for i, t in enumerate(seeds)]

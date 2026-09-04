"""B3 — gold-topology set + topology-correctness scorer.

Gold topologies are parsed from the real ``GETTING_STARTED_EXAMPLE_OUTPUTS/Level N/4_SG.txt``
schematic-generation netlists (the format a pipeline arm also emits): ``nodes`` (id ->
component), ``edges`` (``E*: {link: 'N1,o3: N2,o2'}``), and external ``ports``.

Scoring is **node-id-agnostic**: a pipeline may name instances differently than the gold,
so we compare *typed* connectivity — each endpoint ``Ni,port`` is mapped to
``<component>,<port>`` and edges/nodes are compared as **multisets** (trees repeat
identical typed edges). This answers "did it wire the right *kinds* of components
together?" Exact instance-level isomorphism (distinguishing the 7 identical MZIs in a
tree) is a harder match left for later; ``approx_ged`` (from metrics) is reported as a
secondary set-based signal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from metrics import approx_ged, prf

GOLD_ROOT = Path(__file__).resolve().parent.parent / "GETTING_STARTED_EXAMPLE_OUTPUTS"


@dataclass
class Topology:
    nodes: dict[str, str]                              # instance id -> component type
    edges: list[frozenset[str]]                        # each: {"N1.o3", "N2.o2"} (undirected link)
    external: dict[str, str] = field(default_factory=dict)  # ext port -> "Ni.port"

    def typed_endpoint(self, ref: str) -> str:
        """'N1.o3' -> '<component>.o3' (node-id-agnostic)."""
        inst, _, port = ref.partition(".")
        return f"{self.nodes.get(inst, '?')}.{port}"

    def typed_edges(self) -> Counter:
        return Counter(frozenset(self.typed_endpoint(r) for r in e) for e in self.edges)

    def component_counts(self) -> Counter:
        return Counter(self.nodes.values())


def _endpoint(token: str) -> str:
    """'N1,o3' -> 'N1.o3' (tolerates spaces)."""
    return token.strip().replace(",", ".", 1)


def parse_sg_netlist(text: str) -> Topology:
    """Parse a 4_SG.txt schematic netlist into a Topology."""
    data = yaml.safe_load(text) or {}
    nodes = {nid: (spec or {}).get("component", "?") for nid, spec in (data.get("nodes") or {}).items()}
    edges: list[frozenset[str]] = []
    raw_edges = data.get("edges")
    if isinstance(raw_edges, dict):                    # {'E1': {'link': 'N1,o3: N2,o2'}}
        for spec in raw_edges.values():
            link = (spec or {}).get("link") if isinstance(spec, dict) else None
            if link and ":" in link:
                a, b = link.split(":", 1)
                edges.append(frozenset((_endpoint(a), _endpoint(b))))
    external = {}
    for ext, ref in (data.get("ports") or {}).items():
        external[ext] = _endpoint(ref) if isinstance(ref, str) else ref
    return Topology(nodes=nodes, edges=edges, external=external)


def load_gold_topologies(root: Path | None = None) -> dict[str, Topology]:
    """Parse the four GETTING_STARTED levels into gold topologies keyed by 'Level N'."""
    root = root or GOLD_ROOT
    out: dict[str, Topology] = {}
    for level_dir in sorted(root.glob("Level * Prompt")):
        sg = level_dir / "4_SG.txt"
        if sg.exists():
            key = level_dir.name.replace(" Prompt", "")
            out[key] = parse_sg_netlist(sg.read_text(encoding="utf-8"))
    return out


def _multiset_prf(pred: Counter, gold: Counter) -> tuple[float, float, float]:
    tp = sum((pred & gold).values())
    return prf(tp, sum((pred - gold).values()), sum((gold - pred).values()))


def validity(t: Topology) -> dict:
    """Structural sanity: every edge/port endpoint references an existing node."""
    bad_edges = [e for e in t.edges for r in e if r.split(".")[0] not in t.nodes]
    bad_ports = [p for p, r in t.external.items()
                 if isinstance(r, str) and r.split(".")[0] not in t.nodes]
    connected = {r.split(".")[0] for e in t.edges for r in e} | {
        r.split(".")[0] for r in t.external.values() if isinstance(r, str)}
    floating = [n for n in t.nodes if n not in connected] if len(t.nodes) > 1 else []
    return {"all_refs_exist": not bad_edges and not bad_ports,
            "n_dangling_refs": len(bad_edges) + len(bad_ports),
            "floating_instances": floating}


def score_topology(pred: Topology, gold: Topology) -> dict:
    """Topology correctness of a predicted netlist vs gold (node-id-agnostic)."""
    gold_edges = gold.typed_edges()
    if gold_edges:
        ep, er, ef = _multiset_prf(pred.typed_edges(), gold_edges)
    else:
        # Edgeless gold (single-component design) makes typed-edge F1 undefined
        # (recall = 0/0). Report N/A so these prompts drop out of the edge-F1 mean
        # rather than counting as 0.0, which otherwise dominates and spuriously
        # depresses the headline (9/24 b3_gold prompts are edgeless).
        ep = er = ef = None
    cp, cr, cf = _multiset_prf(pred.component_counts(), gold.component_counts())
    # set-based GED surrogate over component-typed nodes/edges
    pred_n = {f"n{i}": c for i, c in enumerate(pred.nodes.values())}
    gold_n = {f"n{i}": c for i, c in enumerate(gold.nodes.values())}
    ged = approx_ged(pred_n, set(pred.typed_edges()), gold_n, set(gold.typed_edges()))
    return {
        "typed_edge_prf": {"precision": ep, "recall": er, "f1": ef},
        "component_prf": {"precision": cp, "recall": cr, "f1": cf},
        "approx_ged": ged,
        "n_edges": {"pred": len(pred.edges), "gold": len(gold.edges)},
        "n_nodes": {"pred": len(pred.nodes), "gold": len(gold.nodes)},
        "pred_validity": validity(pred),
    }

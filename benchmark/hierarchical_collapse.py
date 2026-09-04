"""Edge-verified hierarchical collapse for topology-correctness scoring.

WHY. The E2 correctness metric compares a predicted circuit topology against a CLASS-level gold
(b3_gold) where composite blocks are ATOMIC nodes (MZI_2x2 / MZI_1x1). The rigid baseline picks
those atomic PDK cells; the agentic builder instead emits the physically-correct PRIMITIVE
decomposition (two 2x2 couplers + a phase element / heater in each arm). The scorer's 1-to-1
class relabel (`b3_eval._apply_accept`) cannot see that a wired {2 couplers + phase-per-arm}
subgraph IS an MZI, so a correct decomposition scores compF1=0 — the entire L3 deficit in
[[e2-o1-correctness-3cond-run-2026-07-09]]. This module recognises such subgraphs and CONTRACTS
them to the atomic node BEFORE scoring, so a decomposition and a library-cell realisation compare
equal.

EDGE-VERIFIED (not node-multiset). Unlike the node-count prototype (`_rescore_hcollapse.py`), a
collapse fires ONLY when the WIRING forms the composite: two couplers joined by exactly two
interferometer arms. Two unrelated couplers plus a stray heater do NOT collapse. This requires the
predicted edges — persisted per rep as `pred_edges` in `_score_correctness.py` (added after the o1
3-cond run, so that run's stored reps have no edges to verify against; this engine is validated on
constructed fixtures + gold topologies here and calibrated against real `pred_edges` on the first
Opus run).

SCOPE / STATUS. v1 recognises the MZI composite (the only decomposition the agentic builder was
observed to emit). It is deliberately conservative — an unrecognised or ambiguous arrangement is
left untouched and scores as before, so the rule can only ever HELP a correct decomposition, never
manufacture a false match. It is NOT comprehensive (rings/CROWs/cascades/lattice filters are not
handled); the final correctness numbers still require a human evaluator to adjudicate the residual
(see [[e2-o1-correctness-3cond-run-2026-07-09]]).
"""
from __future__ import annotations

from topology_eval import Topology

COUPLER_2x2 = {"MMI_2x2", "DC_2x2"}
# 2-port elements that can sit INLINE on an interferometer arm (optical in -> out).
ARM_INLINE = {"HEATER", "_ELECTRICAL", "STRAIGHT", "PHASE_SHIFTER", "PIN"}
# elements that may instead be ELECTRICAL TAPS on an arm (attached, not inline optical).
ARM_DECOR = {"HEATER", "_ELECTRICAL", "PHASE_SHIFTER", "PIN"}


def _split(ref: str) -> tuple[str, str]:
    inst, _, port = ref.partition(".")
    return inst, port


def _adjacency(topo: Topology) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """node -> port -> list of (neighbour_node, neighbour_port)."""
    adj: dict[str, dict[str, list[tuple[str, str]]]] = {n: {} for n in topo.nodes}
    for e in topo.edges:
        ends = list(e)
        if len(ends) != 2:
            continue  # ignore self-loops / malformed edges
        (a, ap), (b, bp) = _split(ends[0]), _split(ends[1])
        if a in adj:
            adj[a].setdefault(ap, []).append((b, bp))
        if b in adj:
            adj[b].setdefault(bp, []).append((a, ap))
    return adj


def _degree(adj, nid) -> int:
    return sum(len(v) for v in adj.get(nid, {}).values())


def _find_arm(adj, nodes, start_coupler, start_port, target_coupler, matched):
    """Trace one arm from start_coupler.start_port toward target_coupler.

    An arm is a direct coupler-coupler edge, or a chain through inline 2-port arm elements
    (HEATER/_ELECTRICAL/STRAIGHT/...). Returns (target_port, [arm_node_ids]) or None.
    """
    node, port = start_coupler, start_port
    arm_nodes: list[str] = []
    for _ in range(len(nodes) + 1):  # bounded: no cycles through distinct nodes
        nbrs = adj[node].get(port, [])
        if len(nbrs) != 1:
            return None
        nxt, nxt_port = nbrs[0]
        if nxt == target_coupler:
            return nxt_port, arm_nodes
        if nxt in matched or nodes.get(nxt) not in ARM_INLINE or _degree(adj, nxt) != 2:
            return None  # dead end, revisits the match, or not a clean inline 2-port arm element
        # step through the inline element to its OTHER port
        other = [p for p in adj[nxt] if p != nxt_port]
        if len(other) != 1:
            return None
        arm_nodes.append(nxt)
        node, port = nxt, other[0]
    return None


def _try_collapse_one(topo: Topology) -> Topology | None:
    """Find one primitive-MZI subgraph and contract it. Return new Topology, or None if none."""
    nodes = topo.nodes
    adj = _adjacency(topo)
    couplers = [n for n, c in nodes.items() if c in COUPLER_2x2]

    for i in range(len(couplers)):
        for j in range(i + 1, len(couplers)):
            c1, c2 = couplers[i], couplers[j]
            matched = {c1, c2}
            arms: list[tuple[str, str, list[str]]] = []  # (c1_port, c2_port, arm_nodes)
            for p in list(adj[c1]):
                res = _find_arm(adj, nodes, c1, p, c2, matched | {c1})
                if res is not None:
                    tgt_port, arm_nodes = res
                    if not (set(arm_nodes) & {a for _, _, an in arms for a in an}):
                        arms.append((p, tgt_port, arm_nodes))
            if len(arms) != 2:
                continue  # an MZI has exactly two arms between the coupler pair

            arm_node_ids = {a for _, _, an in arms for a in an}
            matched |= arm_node_ids
            # Absorb electrical-tap decorators: HEATER/_ELECTRICAL whose every neighbour is in the
            # match (they annotate an arm but aren't inline). Keeps the "active MZI" signature.
            for n, c in nodes.items():
                if n in matched or c not in ARM_DECOR:
                    continue
                nbr_nodes = {b for pv in adj[n].values() for (b, _) in pv}
                if nbr_nodes and nbr_nodes <= matched:
                    matched.add(n)

            # Require at least one phase element (inline arm occupant or absorbed tap) — otherwise
            # two couplers joined by two bare waveguides is a coupler pair, not an MZI worth naming.
            has_phase = any(nodes[a] in (ARM_DECOR | {"STRAIGHT"}) for a in arm_node_ids) or any(
                nodes[n] in ARM_DECOR for n in matched - {c1, c2} - arm_node_ids)
            arm_ports = {(c1, arms[0][0]), (c1, arms[1][0]), (c2, arms[0][1]), (c2, arms[1][1])}

            # External interface = coupler ports NOT facing the arms (either declared external or
            # linked to a node outside the match, or simply unused/dangling).
            def ext_ports(coupler):
                used = {p for (cc, p) in arm_ports if cc == coupler}
                out = []
                for p in adj[coupler]:
                    if p in used:
                        continue
                    if any(b not in matched for (b, _) in adj[coupler][p]):
                        out.append(p)
                # also count declared external ports on this coupler
                for ext, ref in topo.external.items():
                    inst, port = _split(ref) if isinstance(ref, str) else ("", "")
                    if inst == coupler and port not in used and port not in out:
                        out.append(port)
                return out

            ext1, ext2 = ext_ports(c1), ext_ports(c2)
            # Per-side effective arity: trust the wiring when a side has ANY wired/declared external
            # port, else fall back to the coupler's PORT CAPACITY minus its arm ports. This is what
            # distinguishes a bare decomposition (dangling outer ports -> capacity -> 2 per side ->
            # MZI_2x2, e.g. L3_1) from an actually-1x1 interface (each side wires exactly one outer
            # port, e.g. L3_4's DC couplers each feeding one GC). Without the capacity fallback a
            # dangling 2x2 MZI mis-collapses to MZI_1x1 (observed on the real o1 pilot wiring).
            cap = 4  # MMI_2x2 / DC_2x2 are 4-port
            arms1 = len({p for (cc, p) in arm_ports if cc == c1})
            arms2 = len({p for (cc, p) in arm_ports if cc == c2})
            eff1 = len(ext1) if ext1 else cap - arms1
            eff2 = len(ext2) if ext2 else cap - arms2
            n_ext = eff1 + eff2
            if n_ext >= 3:
                mzi_class, port_map = "MZI_2x2", _assign_ports(ext1, ext2, ["o1", "o2"], ["o3", "o4"])
            elif n_ext == 2:
                mzi_class, port_map = "MZI_1x1", _assign_ports(ext1[:1], ext2[:1], ["o1"], ["o2"])
            else:
                continue  # unexpected arity — leave untouched
            if not has_phase and mzi_class == "MZI_2x2":
                continue

            return _contract(topo, matched, c1, c2, mzi_class, port_map)
    return None


def _assign_ports(ext1, ext2, labels1, labels2):
    """Map (coupler, coupler_port) -> new MZI port label, deterministically (sorted)."""
    pm = {}
    for p, lab in zip(sorted(ext1), labels1):
        pm[("C1", p)] = lab
    for p, lab in zip(sorted(ext2), labels2):
        pm[("C2", p)] = lab
    return pm


def _contract(topo, matched, c1, c2, mzi_class, port_map):
    """Build a new Topology with `matched` replaced by a single MZI node 'MZI'."""
    new_nodes = {n: c for n, c in topo.nodes.items() if n not in matched}
    new_nodes["MZI"] = mzi_class

    def remap(ref):
        inst, port = _split(ref)
        if inst not in matched:
            return ref
        key = ("C1", port) if inst == c1 else ("C2", port) if inst == c2 else None
        return f"MZI.{port_map[key]}" if key in port_map else None

    new_edges = []
    for e in topo.edges:
        ends = [remap(r) for r in e]
        if None in ends:
            continue  # internal edge of the collapsed subgraph — drop it
        if _split(ends[0])[0] == _split(ends[1])[0]:
            continue  # both endpoints landed on MZI — internal, drop
        new_edges.append(frozenset(ends))
    new_external = {}
    for ext, ref in topo.external.items():
        r = remap(ref) if isinstance(ref, str) else ref
        if r is not None:
            new_external[ext] = r
    return Topology(nodes=new_nodes, edges=new_edges, external=new_external)


def collapse(topo: Topology, max_iters: int = 8) -> Topology:
    """Iteratively contract every recognised primitive-MZI subgraph. Idempotent when none remain."""
    for _ in range(max_iters):
        nxt = _try_collapse_one(topo)
        if nxt is None:
            return topo
        topo = nxt
    return topo

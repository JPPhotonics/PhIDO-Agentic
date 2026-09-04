"""Generator for large (paper-L4) B3 gold topologies.

Hand-wiring 100+ edges for a 1x64 tree / 8x8 mesh / Spanke fabric is error-prone; instead we
CONSTRUCT each canonical architecture algorithmically so the node set and typed-edge multiset (the
quantities `topology_eval.score_topology` compares) are provably the intended ones. Every builder
returns a gold-entry dict compatible with `b3_gold.json` (`nodes`/`edges`/`external`), authored
from circuit semantics ONLY — no pipeline artifact is consulted (preserves B3 independence).

Port conventions follow b3_gold `_meta.port_conventions`:
  2x2 (MZI_2x2, DC_2x2, MMI_2x2): o1,o2 = left/in ; o3,o4 = right/out
  1x2 (MMI_1x2):                  o1 = common/in  ; o2,o3 = split/out
  1x1 (HEATER, STRAIGHT, MZI_1x1): o1 = in ; o2 = out
  GC (1x0):                       o1 = the on-chip optical port

Run `python benchmark/gold_gen.py --validate` to build + self-check node/edge counts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Edge = list  # ["Ni.oX", "Nj.oY"]


def _ids(prefix: str, n: int, start: int = 1) -> list[str]:
    return [f"{prefix}{i}" for i in range(start, start + n)]


# ---------------------------------------------------------------------------
# Binary 1xN trees (splitter / MZI)
# ---------------------------------------------------------------------------

def binary_tree_1xN(n_out: int, elem_class: str, prefix: str = "N") -> dict:
    """Complete binary tree of 1x2 `elem_class` blocks driving n_out outputs.

    n_out must be a power of two. Uses n_out-1 internal 1x2 blocks (level-order ids), n_out-2
    inter-stage links... actually (n_out-1) blocks with (n_out-2) internal parent->child links
    when leaves are the last-stage OUTPUT ports. Returns nodes, edges, and the list of external
    output refs (so callers can terminate them with GC/HEATER/etc.).
    """
    assert n_out >= 2 and (n_out & (n_out - 1)) == 0, "n_out must be a power of 2"
    n_int = n_out - 1
    nodes = {nid: elem_class for nid in _ids(prefix, n_int)}
    edges: list[Edge] = []
    # 1-indexed complete binary tree: node i has children 2i, 2i+1.
    out_refs: list[str] = []
    for i in range(1, n_int + 1):
        left, right = 2 * i, 2 * i + 1
        # parent split ports o2 (left child) and o3 (right child)
        for child, pport in ((left, "o2"), (right, "o3")):
            if child <= n_int:
                edges.append([f"{prefix}{i}.{pport}", f"{prefix}{child}.o1"])
            else:
                out_refs.append(f"{prefix}{i}.{pport}")  # last-stage output
    return {"nodes": nodes, "edges": edges, "out_refs": out_refs, "in_ref": f"{prefix}1.o1"}


def gold_tree_mzi_1x64() -> dict:
    """TB040: 1x64 tree of 1x2 MZIs, GC at input + each of 64 outputs, plus a
    separate loopback (2 GC + 1 bend)."""
    t = binary_tree_1xN(64, "MZI_1x2", prefix="M")
    nodes = dict(t["nodes"])
    edges = list(t["edges"])
    # input GC
    gc_in = "GIN"
    nodes[gc_in] = "GC"
    edges.append([f"{gc_in}.o1", t["in_ref"]])
    # 64 output GCs
    for k, ref in enumerate(t["out_refs"], 1):
        gid = f"GO{k}"
        nodes[gid] = "GC"
        edges.append([ref, f"{gid}.o1"])
    # separate loopback: GC - BEND - GC (disconnected sub-circuit)
    nodes["LGC1"] = "GC"; nodes["LB"] = "BEND"; nodes["LGC2"] = "GC"
    edges.append(["LGC1.o1", "LB.o1"]); edges.append(["LB.o2", "LGC2.o1"])
    external: dict[str, str] = {}  # all optical I/O terminates in GCs -> no on-chip external
    return {"nodes": nodes, "edges": edges, "external": external}


def gold_tree_mzi_1x32() -> dict:
    """NEW: 1x32 tree of 1x2 MZIs (31 blocks). Outputs left as external ports."""
    t = binary_tree_1xN(32, "MZI_1x2", prefix="M")
    external = {"o1": t["in_ref"]}
    for k, ref in enumerate(t["out_refs"], 2):
        external[f"o{k}"] = ref
    return {"nodes": t["nodes"], "edges": t["edges"], "external": external}


def gold_opa_1x16() -> dict:
    """NEW: 1x16 optical phased array = 1x16 MMI splitter tree -> HEATER per arm -> GC emitter."""
    t = binary_tree_1xN(16, "MMI_1x2", prefix="S")
    nodes = dict(t["nodes"]); edges = list(t["edges"])
    for k, ref in enumerate(t["out_refs"], 1):
        h, g = f"H{k}", f"E{k}"
        nodes[h] = "HEATER"; nodes[g] = "GC"
        edges.append([ref, f"{h}.o1"])
        edges.append([f"{h}.o2", f"{g}.o1"])
    external = {"o1": t["in_ref"]}  # single common input
    return {"nodes": nodes, "edges": edges, "external": external}


# ---------------------------------------------------------------------------
# Crossbar N x N (grid of 2x2 switches)
# ---------------------------------------------------------------------------

def gold_crossbar(N: int) -> dict:
    """N^2 MZI_2x2 switches in an NxN grid. Each switch links to its right neighbor (through,
    o3->o1) and its downward neighbor (cross, o4->o2). Row inputs enter col 0 (o1); column
    outputs leave row N-1 (o4); the far-right through ports (o3) and bottom... conventions vary
    -> topology_certain=false. Node/edge multiset is the scored, invariant quantity."""
    def nid(i, j):
        return f"X{i}_{j}"
    nodes = {nid(i, j): "MZI_2x2" for i in range(N) for j in range(N)}
    edges: list[Edge] = []
    for i in range(N):
        for j in range(N):
            if j + 1 < N:                          # through to right neighbor
                edges.append([f"{nid(i,j)}.o3", f"{nid(i,j+1)}.o1"])
            if i + 1 < N:                          # cross down to next row
                edges.append([f"{nid(i,j)}.o4", f"{nid(i+1,j)}.o2"])
    external: dict[str, str] = {}
    for i in range(N):                             # row optical inputs at left column
        external[f"in{i+1}"] = f"{nid(i,0)}.o1"
    for j in range(N):                             # column outputs at bottom row
        external[f"out{j+1}"] = f"{nid(N-1,j)}.o4"
    for i in range(N):                             # right-edge through outputs
        external[f"thru{i+1}"] = f"{nid(i,N-1)}.o3"
    for j in range(N):                             # top-edge cross inputs
        external[f"cin{j+1}"] = f"{nid(0,j)}.o2"
    return {"nodes": nodes, "edges": edges, "external": external}


# ---------------------------------------------------------------------------
# Benes N x N (recursive rearrangeable network of 2x2 switches)
# ---------------------------------------------------------------------------

def _benes_build(inputs: list[str], counter: dict, nodes: dict, edges: list) -> list[str]:
    """Recursively wire a Benes network over the given `inputs` (list of source refs, each
    'Ni.oX'). Returns the list of output refs. Each switch is an MZI_2x2 (o1,o2 in / o3,o4 out)."""
    N = len(inputs)
    if N == 1:
        return inputs
    if N == 2:
        counter["k"] += 1
        s = f"B{counter['k']}"
        nodes[s] = "MZI_2x2"
        edges.append([inputs[0], f"{s}.o1"])
        edges.append([inputs[1], f"{s}.o2"])
        return [f"{s}.o3", f"{s}.o4"]
    half = N // 2
    # input stage: N/2 switches, switch m takes inputs[m] and inputs[m+half] (shuffle)
    top_in, bot_in = [], []
    for m in range(half):
        counter["k"] += 1
        s = f"B{counter['k']}"
        nodes[s] = "MZI_2x2"
        edges.append([inputs[m], f"{s}.o1"])
        edges.append([inputs[m + half], f"{s}.o2"])
        top_in.append(f"{s}.o3")               # one output to top subnet
        bot_in.append(f"{s}.o4")               # other to bottom subnet
    top_out = _benes_build(top_in, counter, nodes, edges)
    bot_out = _benes_build(bot_in, counter, nodes, edges)
    # output stage: N/2 switches recombining top/bottom subnet outputs
    outs: list[str] = []
    for m in range(half):
        counter["k"] += 1
        s = f"B{counter['k']}"
        nodes[s] = "MZI_2x2"
        edges.append([top_out[m], f"{s}.o1"])
        edges.append([bot_out[m], f"{s}.o2"])
        outs.append(f"{s}.o3"); outs.append(f"{s}.o4")
    return outs[:N]


def gold_benes(N: int) -> dict:
    """NxN Benes network of MZI_2x2 switches. Switch count = N*(2*log2(N)-1)/2."""
    counter = {"k": 0}
    nodes: dict[str, str] = {}
    edges: list[Edge] = []
    inputs = [f"IN{i+1}.o1" for i in range(N)]           # placeholder input refs
    # Realize inputs as external ports directly on the first-stage switches: rebuild without
    # placeholder input nodes by feeding external names.
    nodes.clear(); edges.clear()
    outs = _benes_build([f"__IN{i}" for i in range(N)], counter, nodes, edges)
    # convert placeholder input endpoints into external ports
    external: dict[str, str] = {}
    fixed_edges: list[Edge] = []
    in_map: dict[str, str] = {}
    for e in edges:
        a, b = e
        na = a if not a.startswith("__IN") else None
        nb = b if not b.startswith("__IN") else None
        if a.startswith("__IN"):
            # b is 'Bk.oX' -> make external input
            idx = int(a[4:]) + 1
            external[f"in{idx}"] = b
            continue
        if b.startswith("__IN"):
            idx = int(b[4:]) + 1
            external[f"in{idx}"] = a
            continue
        fixed_edges.append(e)
    for k, ref in enumerate(outs, 1):
        external[f"out{k}"] = ref
    return {"nodes": nodes, "edges": fixed_edges, "external": external}


# ---------------------------------------------------------------------------
# Rail-mesh (Clements rectangular / Reck triangular) of 2x2 MZIs
# ---------------------------------------------------------------------------

def _rail_mesh(placements: list[tuple[int, int]], N: int, prefix: str = "M") -> dict:
    """Build an MZI_2x2 mesh from `placements` = [(column, top_rail), ...]; each MZI spans rails
    (top_rail, top_rail+1). Two MZIs adjacent in column order on a SHARED rail are joined by a
    waveguide (earlier MZI's output port on that rail -> later MZI's input port on that rail).
    The first MZI on a rail exposes an external input; the last exposes an external output.
    Port map per MZI: top rail -> in o1 / out o3 ; bottom rail -> in o2 / out o4."""
    placements = sorted(placements)  # by column, then rail
    nid = {(c, r): f"{prefix}{i+1}" for i, (c, r) in enumerate(placements)}
    nodes = {v: "MZI_2x2" for v in nid.values()}
    # rail -> ordered list of (column, mzi_top_rail, role) touching it
    rail_touch: dict[int, list[tuple[int, int, str]]] = {r: [] for r in range(N)}
    for (c, r) in placements:
        rail_touch[r].append((c, r, "top"))
        rail_touch[r + 1].append((c, r, "bot"))
    def in_port(role):  return "o1" if role == "top" else "o2"
    def out_port(role): return "o3" if role == "top" else "o4"
    edges: list[Edge] = []
    external: dict[str, str] = {}
    for rail in range(N):
        seq = sorted(rail_touch[rail])  # by column
        if not seq:
            external[f"pass{rail+1}"] = None  # (no MZI on this rail — rare)
            continue
        # external input into first MZI on this rail
        (c0, r0, role0) = seq[0]
        external[f"in{rail+1}"] = f"{nid[(c0, r0)]}.{in_port(role0)}"
        # chain consecutive MZIs on this rail
        for (ca, ra, rolea), (cb, rb, roleb) in zip(seq, seq[1:]):
            edges.append([f"{nid[(ca, ra)]}.{out_port(rolea)}",
                          f"{nid[(cb, rb)]}.{in_port(roleb)}"])
        # external output from last MZI on this rail
        (cl, rl, rolel) = seq[-1]
        external[f"out{rail+1}"] = f"{nid[(cl, rl)]}.{out_port(rolel)}"
    external = {k: v for k, v in external.items() if v is not None}
    return {"nodes": nodes, "edges": edges, "external": external}


def gold_clements(N: int) -> dict:
    """NxN Clements rectangular mesh, N columns; even columns pair rails (0,1),(2,3),...,
    odd columns pair (1,2),(3,4),... -> N(N-1)/2 MZIs total."""
    placements = []
    for c in range(N):
        start = 0 if c % 2 == 0 else 1
        for r in range(start, N - 1, 2):
            placements.append((c, r))
    return _rail_mesh(placements, N)


def gold_reck(N: int) -> dict:
    """NxN Reck triangular mesh: N(N-1)/2 MZIs in N-1 layers (sizes N-1, N-2, ..., 1). Laid out
    so layer l (0-indexed) occupies a descending diagonal band -> triangular fill. Connectivity
    via the same shared-rail adjacency as Clements. topology_certain=false (canonical port
    indexing per Reck's diagram is a convention; the node/typed-edge multiset is invariant)."""
    placements = []
    col = 0
    for l in range(N - 1):                      # layer l has (N-1-l) MZIs
        size = N - 1 - l
        for k in range(size):
            top_rail = (N - 2) - k              # fill from the bottom rails upward
            placements.append((col, top_rail))
            col += 1
    return _rail_mesh(placements, N)


# ---------------------------------------------------------------------------
# Spanke fabric (input 1xN switch trees fully crossed into output Nx1 trees)
# ---------------------------------------------------------------------------

def gold_spanke(N: int) -> dict:
    """NxN Spanke: N input 1xN switch-trees + N output Nx1 switch-trees, each built from 1x2
    switch elements (MMI_1x2 class). 1xN tree = N-1 elements; total 2*N*(N-1) elements. Input
    tree i output j crosses to output tree j input i (N^2 crossovers)."""
    nodes: dict[str, str] = {}
    edges: list[Edge] = []
    external: dict[str, str] = {}
    in_out_refs: dict[int, list[str]] = {}      # input tree i -> its N output refs
    out_in_refs: dict[int, list[str]] = {}      # output tree j -> its N input refs
    # input 1xN trees
    for i in range(N):
        t = binary_tree_1xN(N, "MMI_1x2", prefix=f"I{i}_")
        nodes.update(t["nodes"]); edges.extend(t["edges"])
        external[f"in{i+1}"] = t["in_ref"]
        in_out_refs[i] = t["out_refs"]
    # output Nx1 trees (mirror of a 1xN tree; its "in_ref" is the single common OUTPUT port,
    # its "out_refs" are the N branch ports that receive the crossovers)
    for j in range(N):
        t = binary_tree_1xN(N, "MMI_1x2", prefix=f"O{j}_")
        nodes.update(t["nodes"]); edges.extend(t["edges"])
        external[f"out{j+1}"] = t["in_ref"]
        out_in_refs[j] = t["out_refs"]
    # crossovers: input tree i, output j  ->  output tree j, slot i
    for i in range(N):
        for j in range(N):
            edges.append([in_out_refs[i][j], out_in_refs[j][i]])
    return {"nodes": nodes, "edges": edges, "external": external}


TB = {  # prompt text for the reused/added testbench + synthetic prompts (kept verbatim)
    "TB040": "Design a 1×64 Mach-Zehnder tree using six stages of 1x2 MZIs. Each MZI has integrated thermo-optic phase shifters and MMI couplers. Use edge couplers at the single input and at each of the 64 outputs for fiber connection. Additionally, include a separate waveguide loopback structure, containing two edge couplers, connected to a 180-degree bend.",
    "TB099": "Design an 8x8 optical switching network based on the Spanke architecture, ensuring strict-sense non-blocking switching between 8 input and 8 output ports. The network should use 112 built-in Multimode Interferometer (MMI) components and consist of a combination of 8 1x8 switches and 8 8x1 switches.",
    "TB088": "Design a 8×8 optical switching network using a crossbar architecture with mzi_2x2_pn_diode devices. Arrange 64 of these 2×2 switches in a grid such that each of the 8 input waveguides intersects with each of the 8 output waveguides exactly once.",
    "TB086": "Design an 8 × 8 reconfigurable unitary mesh using the Clements scheme. Arrange 28 MZI blocks in a rectangular topology of 8 columns and 4 rows. Use a built-in components called mzi_2x2_pn_diode for the MZI block.",
    "TB097": "Implement an 8×8 unitary mesh in the Reck triangular topology using 28 MZI blocks. Stack seven layers: layer 1 has seven MZIs, layer 2 six, layer 3 five, layer 4 four, layer 5 three, layer 6 two, and layer 7 one. Connect the waveguides so that each MZI’s lower output port feeds the appropriate input of the next layer diagonally, following Reck’s canonical diagram. Use the built-in component mzi_2x2_pn_diode for every MZI block",
    "TB087": "Design a 4×4 optical switching network using a crossbar architecture with mzi_2x2_pn_diode devices. Arrange 16 of these 2×2 switches in a grid such that each of the 4 input waveguides intersects with each of the 4 output waveguides exactly once.",
    "TB084": "Design a 8×8 optical switching network based on the Benes architecture, ensuring rearrangeably non-blocking switching between 8 input and 8 output ports. The network should use 20 2×2 optical switching units and consist of 5 stages, each with 4 2×2 optical switching units. Use the built-in component mzi_2x2_pn_diode to represent each 2×2 optical switching unit. Ensure that the final design places all 8 input ports on one side and all 8 output ports on the other, with correct inter-stage connectivity to realize a full Benes network.",
    "TB098": "Design an 4x4 optical switching network based on the Spanke architecture, ensuring strict-sense non-blocking switching between 4 input and 4 output ports. The network should use 56 built-in Multimode Interferometer (MMI) components and consist of a combination of 4 1x4 switches and 4 4x1 switches.",
    "TB037": "Design a 1×16 power splitter using 15 1×2 MMIs in a four-stage tree: the first stage has 1 MMI splitting into 2 outputs, the second stage has 2 MMIs splitting to 4 outputs, and so on, until 16 outputs. Each of the 16 outputs should feed a VOA (variable optical attenuator) and then a thermo-optic phase shifter, finally connecting to a grating coupler.",
    "TB083": "Design a 4x4 optical switching network based on the Benes architecture, ensuring rearrangeably nonblocking switching between 4 input and 4 output ports. The network should use six 2x2 optical switching units and consist of 3 stages, each with two 2x2 optical switching units. Use the built-in component mzi_2x2_pn_diode to represent each 2×2 optical switching unit.",
    "TB030": "Layout a 1×4 power splitter using three 1×2 MMI couplers in a two-stage cascade. Place 2 phase shifters on the waveguides connecting the first and second stage, and 4 phase shifters on the final output arms, for a total of 6. All phase shifters should be thermo-optic",
    "TB071": "Design an add-drop filter with three double bus ring resonators (10 µm, 15 µm, 20 µm radii) connected along a common bus waveguide. Provide separate grating couplers at the bus waveguide input/output and at each drop port to measure the drop response.",
    "NEW_OPA16": "Design a 1×16 optical phased array: split one input into 16 waveguides through a four-stage tree of 1×2 MMI splitters, place an independent thermo-optic phase shifter on each of the 16 arms, and terminate each arm in a grating-coupler emitter.",
    "NEW_MZITREE32": "Design a 1×32 Mach-Zehnder tree using five stages of 1×2 MZIs (31 MZI blocks) to split one input equally across 32 outputs.",
    "NEW_CLEMENTS16": "Design a 16×16 reconfigurable unitary mesh using the Clements rectangular scheme, arranging 120 MZI blocks in 16 columns; use mzi_2x2_pn_diode for each MZI block.",
}


def gold_tb037() -> dict:
    """TB037: 1x16 MMI splitter tree; each of 16 outputs -> VOA -> HEATER -> GC.

    The PDK HAS a dedicated VOA cell (pindiode_cband, a PIN-diode 1x1 amplitude
    modulator labelled 'Variable Optical Attenuator (VOA)'); the scorer canonicalizes
    it to _ELECTRICAL. Gold node is MOD_1x1, and the accept-set also admits _ELECTRICAL
    (correct VOA choice) and MZI_1x1 so a correctly-selected VOA is not penalized.
    """
    t = binary_tree_1xN(16, "MMI_1x2", prefix="S")
    nodes = dict(t["nodes"]); edges = list(t["edges"]); accept = {}
    for k, ref in enumerate(t["out_refs"], 1):
        v, h, g = f"V{k}", f"H{k}", f"G{k}"
        nodes[v] = "MOD_1x1"; nodes[h] = "HEATER"; nodes[g] = "GC"
        accept[v] = ["MOD_1x1", "MZI_1x1", "_ELECTRICAL"]   # VOA = pindiode_cband -> _ELECTRICAL
        edges += [[ref, f"{v}.o1"], [f"{v}.o2", f"{h}.o1"], [f"{h}.o2", f"{g}.o1"]]
    return {"nodes": nodes, "edges": edges, "external": {"o1": t["in_ref"]}, "accept": accept}


def gold_tb030() -> dict:
    """1x4 = three 1x2 MMIs, 2 inter-stage + 4 output thermo-optic phase shifters (6 total)."""
    nodes = {"S1": "MMI_1x2", "S2": "MMI_1x2", "S3": "MMI_1x2",
             "H1": "HEATER", "H2": "HEATER", "H3": "HEATER", "H4": "HEATER", "H5": "HEATER", "H6": "HEATER"}
    edges = [["S1.o2", "H1.o1"], ["H1.o2", "S2.o1"], ["S1.o3", "H2.o1"], ["H2.o2", "S3.o1"],
             ["S2.o2", "H3.o1"], ["S2.o3", "H4.o1"], ["S3.o2", "H5.o1"], ["S3.o3", "H6.o1"]]
    external = {"o1": "S1.o1", "o2": "H3.o2", "o3": "H4.o2", "o4": "H5.o2", "o5": "H6.o2"}
    return {"nodes": nodes, "edges": edges, "external": external}


def gold_tb071() -> dict:
    """Three double-bus rings serially on a common bus; GCs at bus in/out and each drop."""
    nodes = {"R1": "RING_2bus", "R2": "RING_2bus", "R3": "RING_2bus",
             "GIN": "GC", "GOUT": "GC", "GD1": "GC", "GD2": "GC", "GD3": "GC"}
    edges = [["GIN.o1", "R1.o1"], ["R1.o2", "R2.o1"], ["R2.o2", "R3.o1"], ["R3.o2", "GOUT.o1"],
             ["R1.o3", "GD1.o1"], ["R2.o3", "GD2.o1"], ["R3.o3", "GD3.o1"]]
    return {"nodes": nodes, "edges": edges, "external": {}}


def assemble() -> dict:
    """Build the 12-L3 + 12-L4 v2 gold set. Reuses the 9 existing paper-L3 entries from
    b3_gold.json verbatim; generates the L4 and 3 added L3 topologies."""
    base = json.loads((Path(__file__).parent / "b3_gold.json").read_text())
    old = {g["id"]: g for g in base["gold"]}
    meta = dict(base["_meta"])
    meta["title"] = "B3 gold v2 (12 L3 + 12 L4, paper Table-1 axis; L1/L2 dropped)"
    meta["status"] = "DRAFT (in-house, generator-authored) — auto-built by gold_gen.py"

    def reuse(old_id, new_id, level):
        e = dict(old[old_id]); e["id"] = new_id; e["level"] = level
        e["source"] = f"reused:{old_id}"; return e

    def wrap(new_id, level, src, body, certain, prompt, rationale, ambiguity=None):
        e = {"id": new_id, "level": level, "topology_certain": certain,
             "prompt": prompt, "nodes": body["nodes"], "edges": body["edges"],
             "external": body.get("external", {}), "source": src, "rationale": rationale}
        if "accept" in body:
            e["accept"] = body["accept"]
        if ambiguity:
            e["ambiguity"] = ambiguity
        return e

    # ---- 12 L3 (paper 3..15 components) ----
    l3 = [
        reuse("L2_2", "L3_01", 3), reuse("L2_6", "L3_02", 3),
        reuse("L3_4", "L3_03", 3), reuse("L3_5", "L3_04", 3),
        reuse("L4_4", "L3_05", 3), reuse("L4_5", "L3_06", 3),
        reuse("L4_2", "L3_07", 3), reuse("L4_3", "L3_08", 3),
        reuse("L4_1", "L3_09", 3),
        wrap("L3_10", 3, "TB030", gold_tb030(), True, TB["TB030"],
             "Two-stage 1x4 MMI tree; 2 inter-stage + 4 output thermo-optic phase shifters (9 comp)."),
        wrap("L3_11", 3, "TB071", gold_tb071(), True, TB["TB071"],
             "Three double-bus rings serially on a common bus; GC at bus in/out and each drop (8 comp)."),
        wrap("L3_12", 3, "TB083", gold_benes(4), False, TB["TB083"],
             "4x4 Benes = 6 2x2 switches in 3 stages of 2 (paper-L3).",
             "Benes inter-stage shuffle is canonical; exact port indexing is a convention. Node/typed-edge multiset invariant."),
    ]

    # ---- 12 L4 (paper 16..112 components) ----
    l4 = [
        wrap("L4_01", 4, "TB040", gold_tree_mzi_1x64(), True, TB["TB040"],
             "1x64 binary tree of 63 1x2 MZIs; GC at input + all 64 outputs; separate GC-BEND-GC loopback (131 comp)."),
        wrap("L4_02", 4, "TB099", gold_spanke(8), False, TB["TB099"],
             "8x8 Spanke: 8 input 1x8 + 8 output 8x1 switch-trees (7 elems each = 112), fully crossed.",
             "Switch element = 1 node (MMI_1x2 class); the prompt's '112 MMI' matches the element count. Crossover indexing is a convention."),
        wrap("L4_03", 4, "NEW_CLEMENTS16", gold_clements(16), False, TB["NEW_CLEMENTS16"],
             "16x16 Clements rectangular mesh = 120 MZI_2x2 blocks (120 comp).",
             "Rectangular nearest-neighbor mesh; exact port indexing per Clements diagram is a convention. Multiset invariant."),
        wrap("L4_04", 4, "TB088", gold_crossbar(8), False, TB["TB088"],
             "8x8 crossbar = 64 MZI_2x2 switches; each links its right (through) and down (cross) neighbor.",
             "Crossbar grid adjacency is determinate; the external input/output/through port convention varies."),
        wrap("L4_05", 4, "TB086", gold_clements(8), False, TB["TB086"],
             "8x8 Clements rectangular mesh = 28 MZI_2x2 blocks (28 comp).",
             "See L4_03 convention note."),
        wrap("L4_06", 4, "TB097", gold_reck(8), False, TB["TB097"],
             "8x8 Reck triangular mesh = 28 MZI_2x2 in 7 layers (7..1).",
             "Reck triangular layout; canonical port indexing per Reck's diagram is a convention. Multiset invariant."),
        wrap("L4_07", 4, "TB098", gold_spanke(4), False, TB["TB098"],
             "4x4 Spanke: 4 input 1x4 + 4 output 4x1 switch-trees (3 elems each = 24), fully crossed.",
             "Element-level count = 24; the prompt's '56 MMI' counts sub-MMIs inside each switch. Crossover indexing is a convention."),
        wrap("L4_08", 4, "TB084", gold_benes(8), False, TB["TB084"],
             "8x8 Benes = 20 2x2 switches in 5 stages of 4 (20 comp).",
             "Benes inter-stage shuffle is canonical; exact port indexing is a convention."),
        wrap("L4_09", 4, "TB087", gold_crossbar(4), False, TB["TB087"],
             "4x4 crossbar = 16 MZI_2x2 switches (16 comp).",
             "See L4_04 convention note."),
        wrap("L4_10", 4, "NEW_OPA16", gold_opa_1x16(), True, TB["NEW_OPA16"],
             "1x16 optical phased array: 15 1x2 MMI splitter tree -> HEATER per arm -> GC emitter (47 comp)."),
        wrap("L4_11", 4, "NEW_MZITREE32", gold_tree_mzi_1x32(), True, TB["NEW_MZITREE32"],
             "1x32 binary tree of 31 1x2 MZIs (31 comp)."),
        wrap("L4_12", 4, "TB037", gold_tb037(), False, TB["TB037"],
             "1x16 MMI tree (15) -> per output: VOA (MOD_1x1) -> thermo-optic phase (HEATER) -> GC (63 comp).",
             "VOA IS in the PDK (pindiode_cband, scorer token _ELECTRICAL); gold node MOD_1x1, accept-set {MOD_1x1, MZI_1x1, _ELECTRICAL}."),
    ]
    return {"_meta": meta, "gold": l3 + l4}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--build", action="store_true", help="write b3_gold_v2.json")
    args = ap.parse_args()

    if args.build:
        out = Path(__file__).parent / "b3_gold_v2.json"
        data = assemble()
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        from collections import Counter
        lv = Counter(g["level"] for g in data["gold"])
        print(f"wrote {out} : {len(data['gold'])} entries, by level {dict(sorted(lv.items()))}")
        raise SystemExit(0)

    builders = {
        "TB040 (1x64 MZI tree)": (gold_tree_mzi_1x64, None),
        "NEW 1x32 MZI tree": (gold_tree_mzi_1x32, 31),
        "NEW 1x16 OPA": (gold_opa_1x16, 15 + 16 + 16),
        "TB087 (4x4 crossbar)": (lambda: gold_crossbar(4), 16),
        "TB088 (8x8 crossbar)": (lambda: gold_crossbar(8), 64),
        "TB083 (4x4 Benes)": (lambda: gold_benes(4), 6),
        "TB084 (8x8 Benes)": (lambda: gold_benes(8), 20),
        "TB086 (8x8 Clements)": (lambda: gold_clements(8), 28),
        "NEW 16x16 Clements": (lambda: gold_clements(16), 120),
        "TB097 (8x8 Reck)": (lambda: gold_reck(8), 28),
        "TB098 (4x4 Spanke)": (lambda: gold_spanke(4), 24),
        "TB099 (8x8 Spanke)": (lambda: gold_spanke(8), 112),
    }
    for label, (fn, expect_nodes) in builders.items():
        g = fn()
        nn, ne = len(g["nodes"]), len(g["edges"])
        exp = f" (expect {expect_nodes})" if expect_nodes else ""
        flag = ""
        if expect_nodes and nn != expect_nodes:
            flag = "  <-- NODE COUNT MISMATCH"
        print(f"{label:28} nodes={nn:>4}{exp}  edges={ne:>4}{flag}")

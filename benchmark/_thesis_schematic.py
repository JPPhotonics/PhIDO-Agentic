"""Presentation-quality circuit schematic renderer — the SAME style as the Ch6 thesis figures
(thesis/figures/gen_e2_schematics.py), packaged for reuse by the blind-review HTML builder.

Why it reads better than the review tool's original record layout: edges are oriented
source-port -> sink-port so `rankdir=LR` ranks the pipeline stages (no backward-snaking edges),
node labels are HTML tables with input ports on the left face / outputs on the right (graphviz
2.43 can route flat same-rank edges only between HTML-plain nodes, not record nodes), and crossbar
grids (X<r>_<c> names) get explicit same-rank column groups.

`safe_to_dot()` wraps it so any malformed topology (a weak model emits plenty) falls back to the
caller's simpler renderer instead of crashing the whole build.
"""
from __future__ import annotations

import re

# Per abstract class: (n_ports, ports on the input face). Source: b3_gold_v2 _meta.port_conventions.
CLASS_PORTS = {
    "GC": (1, {"o1"}),
    "STRAIGHT": (2, {"o1"}), "BEND": (2, {"o1"}), "RING_1BUS": (2, {"o1"}),
    "HEATER": (2, {"o1"}), "MZI_1X1": (2, {"o1"}), "MOD_1X1": (2, {"o1"}),
    "PD": (2, {"o1"}), "_ELECTRICAL": (2, {"o1"}),
    "MMI_1X2": (3, {"o1"}), "MZI_1X2": (3, {"o1"}), "PSR": (3, {"o1"}),
    "DC_2X2": (4, {"o1", "o2"}), "MMI_2X2": (4, {"o1", "o2"}),
    "MZI_2X2": (4, {"o1", "o2"}), "RING_2BUS": (4, {"o1", "o4"}),
    "CROSSING": (4, {"o1", "o2"}), "COUPLER_RING": (4, {"o1", "o2"}),
    "WDM": (5, {"o1"}),
}
FLEX_CLASSES = {"WDM"}  # channel ports may act as inputs (mux) or outputs (demux)


def _class(typ: str):
    key = (typ or "").upper()
    if key in CLASS_PORTS:
        return CLASS_PORTS[key]
    m = re.search(r"_(\d+)X(\d+)$", key)          # infer from a trailing _NxM
    if m:
        ni, no = int(m.group(1)), int(m.group(2))
        return (ni + no, {f"o{i}" for i in range(1, ni + 1)})
    return (4, {"o1", "o2"})


def _split(ep):
    return ep.rsplit(".", 1) if isinstance(ep, str) and "." in ep else (ep, None)


def _pidx(p) -> int:
    m = re.search(r"(\d+)", p or "")
    return int(m.group(1)) if m else 1


def orient_edges(nodes: dict, edges: list):
    """(directed [(na,pa,nb,pb,dashed)], per-node input-port set). Skips edges to missing nodes."""
    sink_ports = {n: set() for n in nodes}
    directed = []
    for e in edges:
        if len(e) != 2:
            continue
        (na, pa), (nb, pb) = _split(e[0]), _split(e[1])
        if na not in nodes or nb not in nodes or pa is None or pb is None:
            directed.append((na, pa, nb, pb, True))
            continue
        _, ins_a = _class(nodes[na])
        _, ins_b = _class(nodes[nb])
        a_in, a_flex = pa in ins_a, nodes[na].upper() in FLEX_CLASSES
        b_in, b_flex = pb in ins_b, nodes[nb].upper() in FLEX_CLASSES
        if b_in and not a_in:
            src, dst = (na, pa), (nb, pb)
        elif a_in and not b_in:
            src, dst = (nb, pb), (na, pa)
        elif b_flex and not a_flex:
            src, dst = (na, pa), (nb, pb)
        elif a_flex and not b_flex:
            src, dst = (nb, pb), (na, pa)
        else:
            directed.append((na, pa, nb, pb, True))   # unresolvable -> dashed (flags miswiring)
            continue
        sink_ports[dst[0]].add(dst[1])
        directed.append((src[0], src[1], dst[0], dst[1], False))
    return directed, sink_ports


def _esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node_label(name, typ, extra_ins, max_seen, module=None) -> str:
    n_ports, class_ins = _class(typ)
    n_ports = max(n_ports, max_seen)
    if typ.upper() in FLEX_CLASSES and extra_ins:
        ins = sorted(extra_ins)
    else:
        ins = sorted(class_ins | (extra_ins - class_ins))
    outs = [f"o{i}" for i in range(1, n_ports + 1) if f"o{i}" not in ins]
    rows = max(1, len(ins), len(outs))

    def col(ports):
        if not ports:
            return {}
        spans, base, rem, row = {}, rows // len(ports), rows % len(ports), 0
        for i, p in enumerate(ports):
            span = base + (1 if i < rem else 0)
            spans[row] = (p, span)
            row += span
        return spans

    lcol, rcol = col(ins), col(outs)
    center = f" {name}: {typ}" + (f" ({module})" if module else "") + " "
    body = []
    for r in range(rows):
        cells = []
        if r in lcol:
            p, s = lcol[r]
            cells.append(f'<TD ROWSPAN="{s}" PORT="{_esc(p)}">{_esc(p)}</TD>')
        if r == 0:
            cells.append(f'<TD ROWSPAN="{rows}">{_esc(center)}</TD>')
        if r in rcol:
            p, s = rcol[r]
            cells.append(f'<TD ROWSPAN="{s}" PORT="{_esc(p)}">{_esc(p)}</TD>')
        body.append("<TR>" + "".join(cells) + "</TR>")
    return ('<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">'
            + "".join(body) + "</TABLE>>")


def _grid_groups(nodes: dict):
    pat = re.compile(r"^X(\d+)_(\d+)$")
    cols: dict = {}
    for n in nodes:
        m = pat.match(n)
        if not m:
            return None
        cols.setdefault(int(m.group(2)), []).append((int(m.group(1)), n))
    return [[n for _r, n in sorted(v)] for _c, v in sorted(cols.items())]


def _safe_id(n):
    """Graphviz-safe node id (Qwen may emit ids with punctuation)."""
    return '"' + str(n).replace('"', "'") + '"'


def to_dot(title: str, nodes: dict, edges: list, modules: dict | None = None) -> str:
    modules = modules or {}
    rank_groups = _grid_groups(nodes)
    directed, sink_ports = orient_edges(nodes, edges)
    max_seen = {n: 0 for n in nodes}
    for e in edges:
        if len(e) != 2:
            continue
        for end in e:
            n, p = _split(end)
            if n in max_seen:
                max_seen[n] = max(max_seen[n], _pidx(p))
    lines = [
        f'digraph "{_esc(title)}" {{',
        "  rankdir=LR; splines=true; nodesep=0.35; ranksep=0.7;",
        '  node [shape=plain, fontsize=11, fontname="Helvetica"];',
        "  edge [arrowhead=none];",
    ]
    for n, t in nodes.items():
        lines.append(f"  {_safe_id(n)} [label={node_label(n, t, sink_ports[n], max_seen[n], modules.get(n))}];")
    in_group = {}
    if rank_groups:
        for gi, grp in enumerate(rank_groups):
            lines.append("  { rank=same; " + "; ".join(_safe_id(n) for n in grp) + "; }")
            for n in grp:
                in_group[n] = gi
            for a, b in zip(grp, grp[1:]):
                lines.append(f"  {_safe_id(a)} -> {_safe_id(b)} [style=invis];")
    for na, pa, nb, pb, dashed in directed:
        attrs = []
        if dashed:
            attrs.append("style=dashed")
        if rank_groups and in_group.get(na) is not None and in_group.get(na) == in_group.get(nb):
            attrs.append("constraint=false")
        a = f" [{', '.join(attrs)}]" if attrs else ""
        sa = f"{_safe_id(na)}:{pa}" if pa else _safe_id(na)
        sb = f"{_safe_id(nb)}:{pb}" if pb else _safe_id(nb)
        lines.append(f"  {sa} -> {sb}{a};")
    lines.append("}")
    return "\n".join(lines)


def safe_to_dot(title, nodes, edges, modules=None):
    """Thesis-style dot, or None if the topology is too malformed to render this way."""
    try:
        if not nodes:
            return None
        return to_dot(title, nodes, edges, modules)
    except Exception:  # noqa: BLE001
        return None

"""Render the built knowledge graph itself as thesis figures (vector PDF + PNG preview).

F1 ch6_kg_artifact  — the whole SERVED CONTENT graph (465 nodes / 3323 edges): every node
                      of the seven content labels, coloured by label, sized by degree.
                      Excludes the ReviewItem curation queue, the RelationshipObservation
                      ledger, and the SchemaRelationType registry, which are construction
                      bookkeeping rather than knowledge the online pipeline reads (A6).
F2 ch5_kg_instance  — one PDK cell's neighbourhood (the TiN heater, module heater_tin_cband),
                      the schema of Table 5.1/5.2 populated with real extracted data. This is
                      the same cell used as the A5 worked example and the E1 retrieval example.

Layout is deterministic (seeded Fruchterman-Reingold), so a re-run of this script against the
same frozen build reproduces the same picture. Read-only: uses benchmark.kb_access.KB.
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import _thesis_fig_style as S
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from kb_access import KB

OUT = Path("/home/tony/PhIDOv1/PhIDO-Release/thesis/figures")
S.use_style()

CONTENT = ["Component", "Architecture", "Design_Function", "Physical_Principle",
           "Property", "PDK_Cell", "Document"]
# MATLAB default colour order, one slot per label (fills with black edges, so the
# sub-3:1 slots are admissible here as they are in the Ch6 stacked bars).
COLOR = {"Component": S.BLUE, "Architecture": S.ORANGE, "Design_Function": S.YELLOW,
         "Physical_Principle": S.PURPLE, "Property": S.GREEN, "PDK_Cell": S.LBLUE,
         "Document": "#A2142F"}
LEGEND_ORDER = ["Component", "Architecture", "Design_Function", "Physical_Principle",
                "Property", "PDK_Cell", "Document"]

kb = KB()
lab_list = "[" + ",".join(f"'{x}'" for x in CONTENT) + "]"

# ================= F1: the whole served content graph =================
nodes = kb.q(f"""MATCH (n) WHERE any(l IN labels(n) WHERE l IN {lab_list})
RETURN elementId(n) AS id, [l IN labels(n) WHERE l IN {lab_list}][0] AS lab,
       coalesce(n.name, n.title) AS name""")
edges = kb.q(f"""MATCH (a)-[r]->(b)
WHERE any(l IN labels(a) WHERE l IN {lab_list}) AND any(l IN labels(b) WHERE l IN {lab_list})
RETURN elementId(a) AS s, elementId(b) AS t, type(r) AS ty""")
print(f"F1 content subgraph: {len(nodes)} nodes, {len(edges)} edges")

G = nx.Graph()
for n in nodes:
    G.add_node(n["id"], lab=n["lab"], name=n["name"])
for e in edges:
    G.add_edge(e["s"], e["t"], ty=e["ty"])

pos = nx.spring_layout(G, k=0.62, iterations=400, seed=7)
deg = dict(G.degree())

fig, ax = plt.subplots(figsize=(7.8, 7.5))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9a9a9a", width=0.30, alpha=0.85)
# IMPLEMENTS is the hinge between the two layers: draw it on top, in ink.
imp = [(e["s"], e["t"]) for e in edges if e["ty"] == "IMPLEMENTS"]
nx.draw_networkx_edges(G, pos, edgelist=imp, ax=ax, edge_color=S.INK, width=0.9)
for lab in LEGEND_ORDER:
    ids = [n for n, d in G.nodes(data=True) if d["lab"] == lab]
    nx.draw_networkx_nodes(
        G, pos, nodelist=ids, ax=ax, node_color=COLOR[lab],
        node_size=[(6 + 1.5 * deg[n]) * 1.25 for n in ids],
        edgecolors=S.INK, linewidths=0.25)
# Name every node. The plate is printed landscape at full page, where the outer two
# thirds of the labels resolve; the core stays dense and is read as texture.
for n in G.nodes:
    nm = str(G.nodes[n]["name"] or "")
    if not nm:
        continue
    if G.nodes[n]["lab"] == "Document":
        nm = nm.split("_", 1)[-1].replace("_", " ")
    nm = re.sub(r"\s*\([^)]*\)\s*$", "", nm)
    nm = nm if len(nm) <= 22 else nm[:21] + "\u2026"
    x, y = pos[n]
    ax.text(x, y + 0.010 + 0.0004 * deg[n], nm, fontsize=3.2, color="#111111",
            ha="center", va="bottom", zorder=6)
ax.set_aspect("equal")
ax.set_axis_off()
handles = [mpatches.Patch(facecolor=COLOR[m], edgecolor=S.INK, linewidth=0.5,
                          label=m.replace("_", "\\_") if False else m.replace("_", " "))
           for m in LEGEND_ORDER]
handles.append(plt.Line2D([], [], color=S.INK, lw=1.1, label="IMPLEMENTS"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
          ncol=4, fontsize=7, handletextpad=0.5, columnspacing=1.1)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch6_kg_artifact.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote ch6_kg_artifact")

# ================= F2: one cell's neighbourhood, grouped by relation =================
MOD = "heater_tin_cband"
# outgoing only: what the graph asserts ABOUT this cell (an incoming COMPOSED_OF is
# another cell's claim, and would read backwards once the sector layout drops direction)
ego = kb.q(f"""MATCH (c:PDK_Cell {{module_name:'{MOD}'}})-[r]->(m)
WHERE any(l IN labels(m) WHERE l IN {lab_list})
RETURN c.name AS cname, type(r) AS ty, labels(m)[0] AS lab,
       coalesce(m.name, m.title) AS name,
       true AS outgoing
ORDER BY ty, name""")
root = ego[0]["cname"]
print(f"F2 ego: {len(ego)} edges on {MOD} ({root})")

# angular sectors, one per relation type, sized by how many edges it carries
ORDER = ["IMPLEMENTS", "PERFORMS_FUNCTION", "EXHIBITS", "FABRICATED_WITH", "COMPOSED_OF"]
groups = {t: [e for e in ego if e["ty"] == t] for t in ORDER}
groups = {t: v for t, v in groups.items() if v}
for t in {e["ty"] for e in ego} - set(groups):        # anything unforeseen still drawn
    groups[t] = [e for e in ego if e["ty"] == t]
n_tot = sum(len(v) for v in groups.values())

fig, ax = plt.subplots(figsize=(6.6, 5.0))
GAP = 0.16                                            # radians of blank between sectors
span_free = 2 * np.pi - GAP * len(groups)
theta = np.pi / 2                                     # start at the top, go clockwise
placed = []
for t, items in groups.items():
    span = span_free * len(items) / n_tot
    for i, e in enumerate(items):
        frac = (i + 0.5) / len(items)
        a = theta - span * frac
        # three-level stagger in crowded sectors so adjacent labels never share a band
        r = 1.0 if len(items) < 6 else (1.0, 1.36, 1.72)[i % 3]
        placed.append((e, a, r))
    mid = theta - span / 2
    rmax = max(1.0 if len(items) < 6 else (1.0, 1.36, 1.72)[i % 3]
               for i in range(len(items)))
    # node labels run horizontally, so a sector facing left or right needs more
    # clearance than one facing up or down
    rlab = rmax + (0.45 if len(items) <= 3 else 0.34) + 0.78 * abs(np.cos(mid))
    ax.text(rlab * np.cos(mid), rlab * np.sin(mid),
            t.replace("_", " ").lower(), fontsize=7.4, fontweight="bold",
            ha="center", va="center", color=S.INK)
    theta -= span + GAP

ax.scatter([0], [0], s=620, color=COLOR["PDK_Cell"], edgecolor=S.INK, lw=0.9, zorder=4)
ax.text(0, -0.19, root, fontsize=7.0, fontweight="bold", ha="center", va="top", zorder=5)
for e, a, r in placed:
    x, y = r * np.cos(a), r * np.sin(a)
    ax.plot([0, x], [0, y], color="#bbbbbb", lw=0.7, zorder=1)
    ax.scatter([x], [y], s=210, color=COLOR[e["lab"]], edgecolor=S.INK, lw=0.6, zorder=3)
    nm = str(e["name"])
    nm = nm if len(nm) <= 30 else nm[:29] + "\u2026"
    # push the label straight outward along its own spoke, so neighbours diverge
    lx, ly = (r + 0.145) * np.cos(a), (r + 0.145) * np.sin(a)
    ha = "left" if np.cos(a) > 0.12 else ("right" if np.cos(a) < -0.12 else "center")
    va = "center" if ha != "center" else ("bottom" if np.sin(a) > 0 else "top")
    ax.text(lx, ly, nm, fontsize=6.2, ha=ha, va=va, zorder=6,
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none",
                  "alpha": 0.88})
ax.set_xlim(-2.95, 2.95)
ax.set_ylim(-2.30, 2.30)
ax.set_aspect("equal")
ax.set_axis_off()
handles2 = [mpatches.Patch(facecolor=COLOR[m], edgecolor=S.INK, linewidth=0.5,
                           label=m.replace("_", " "))
            for m in LEGEND_ORDER
            if any(e["lab"] == m for e in ego) or m == "PDK_Cell"]
ax.legend(handles=handles2, loc="lower center", bbox_to_anchor=(0.5, -0.06),
          ncol=5, fontsize=7, handletextpad=0.5, columnspacing=1.1)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ch5_kg_instance.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote ch5_kg_instance")

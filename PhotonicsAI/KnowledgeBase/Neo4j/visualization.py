"""Neo4j Graph Visualization using PyVis.

Performance note: the browser bottleneck for large graphs is vis.js *continuous*
physics. We instead compute a layout ONCE server-side (networkx spring layout) and
ship the graph with physics disabled, so the browser only draws static positions —
this renders smoothly even for thousands of nodes. Bookkeeping labels (ReviewItem,
RelationshipObservation, SchemaRelationType) are excluded by default because they
dominate the node count (~2100 ReviewItems) without adding semantic structure.
"""

from __future__ import annotations

import networkx as nx
from pyvis.network import Network

from .client import Neo4jClient

# Internal bookkeeping labels excluded by default (not semantic KB content).
INTERNAL_LABELS = ("ReviewItem", "RelationshipObservation", "SchemaRelationType", "PDK_Cell_History")

# Label -> colour (content labels; bookkeeping labels fall back to grey).
COLORS = {
    "Component": "#ff9900",
    "Architecture": "#00ccff",
    "Property": "#cc00ff",
    "Design_Function": "#00ff99",
    "Physical_Principle": "#ff0066",
    "PDK_Cell": "#ffe14d",
    "Document": "#aaaaaa",
    "ReviewItem": "#666666",
    "RelationshipObservation": "#888888",
    "SchemaRelationType": "#5555aa",
}


class Neo4jVisualizer:
    """Visualizes the Neo4j Knowledge Graph using PyVis (static precomputed layout)."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    def visualize_graph(
        self,
        output_file: str = "neo4j_graph.html",
        include_internal: bool = False,
        limit: int | None = None,
        layout_iterations: int = 60,
        seed: int = 42,
        scale: float = 1800.0,
    ) -> dict:
        """Fetch the graph from Neo4j and render it to a self-contained HTML file.

        Args:
            output_file: Path to save the HTML file.
            include_internal: Include bookkeeping nodes (ReviewItem, RelationshipObservation,
                SchemaRelationType, PDK_Cell_History). Default False — these dwarf the content
                graph and slow rendering.
            limit: Optional cap on the number of nodes (highest-degree kept first).
            layout_iterations: networkx spring-layout iterations (more = tidier, slower).
            seed: layout RNG seed (deterministic positions across runs).
            scale: pixel scale applied to the unit-square layout coordinates.

        Returns:
            Summary dict {nodes, edges, excluded_labels}.
        """
        if not self.client._connected:
            self.client.connect()

        excluded = [] if include_internal else list(INTERNAL_LABELS)

        # Fetch nodes (optionally excluding bookkeeping labels). element_id is the stable key;
        # display name falls back to title/module_name/element_id.
        where = "" if include_internal else "WHERE NONE(l IN labels(n) WHERE l IN $excluded)"
        node_query = f"MATCH (n) {where} RETURN n"
        # Edges only between kept nodes.
        edge_where = (
            ""
            if include_internal
            else "WHERE NONE(l IN labels(n) WHERE l IN $excluded) "
            "AND NONE(l IN labels(m) WHERE l IN $excluded)"
        )
        rel_query = f"MATCH (n)-[r]->(m) {edge_where} RETURN n, r, m"

        nodes: dict[str, dict] = {}
        edges: list[tuple[str, str, str, str]] = []

        with self.client.driver.session() as session:
            for record in session.run(node_query, excluded=excluded):
                n = record["n"]
                nid = n.element_id
                if nid in nodes:
                    continue
                label = list(n.labels)[0] if n.labels else "Unknown"
                name = n.get("name") or n.get("title") or n.get("module_name") or label
                title = f"{name}\n({label})"
                if n.get("description"):
                    title += f"\n\n{str(n.get('description'))[:200]}"
                nodes[nid] = {"name": str(name), "label": label, "title": title}

            for record in session.run(rel_query, excluded=excluded):
                a, b, r = record["n"], record["m"], record["r"]
                aid, bid = a.element_id, b.element_id
                # Endpoints must be among the kept nodes (the WHERE guarantees this).
                if aid not in nodes or bid not in nodes:
                    continue
                etitle = r.type + (f"\n{r.get('description')}" if r.get("description") else "")
                edges.append((aid, bid, r.type, etitle))

        # Build a networkx graph for layout + degree-based sizing.
        g = nx.Graph()
        g.add_nodes_from(nodes.keys())
        g.add_edges_from((a, b) for a, b, _, _ in edges)

        # Optional node cap: keep the highest-degree nodes (most structurally central).
        if limit and len(nodes) > limit:
            keep = {n for n, _ in sorted(g.degree, key=lambda kv: kv[1], reverse=True)[:limit]}
            nodes = {nid: v for nid, v in nodes.items() if nid in keep}
            edges = [e for e in edges if e[0] in keep and e[1] in keep]
            g = nx.Graph()
            g.add_nodes_from(nodes.keys())
            g.add_edges_from((a, b) for a, b, _, _ in edges)

        # Precompute positions ONCE (sparse FR layout); scale to pixels.
        pos = nx.spring_layout(g, k=None, iterations=layout_iterations, seed=seed)
        degrees = dict(g.degree)

        net = Network(
            height="100vh",
            width="100%",
            bgcolor="#1a1a1a",
            font_color="white",
            select_menu=True,
            filter_menu=True,
            directed=True,
            cdn_resources="in_line",  # inline vis-network JS/CSS -> single portable file
        )

        for nid, v in nodes.items():
            x, y = pos[nid]
            deg = degrees.get(nid, 0)
            net.add_node(
                nid,
                label=v["name"],
                title=v["title"],
                color=COLORS.get(v["label"], "#999999"),
                group=v["label"],
                shape="dot",
                size=8 + min(deg, 40) * 1.2,  # degree-scaled, capped
                x=float(x) * scale,
                y=float(y) * scale,
                physics=False,  # use the precomputed position; do not simulate
            )

        for aid, bid, etype, etitle in edges:
            net.add_edge(aid, bid, title=etitle, label=etype, arrows="to")

        # Static render: physics OFF (positions are precomputed), straight edges, light arrows.
        net.set_options(
            """
            var options = {
              "physics": { "enabled": false },
              "interaction": {
                "hover": true, "tooltipDelay": 120,
                "navigationButtons": true, "keyboard": true,
                "hideEdgesOnDrag": true, "hideEdgesOnZoom": true
              },
              "edges": {
                "smooth": false,
                "color": { "color": "#5a5a5a", "highlight": "#ffffff", "opacity": 0.5 },
                "arrows": { "to": { "enabled": true, "scaleFactor": 0.4 } },
                "font": { "size": 0 }
              },
              "nodes": {
                "font": { "size": 14, "color": "#ffffff" },
                "borderWidth": 1
              }
            }
            """
        )

        net.save_graph(output_file)
        self._make_self_contained(output_file)
        self._inject_legend(output_file, nodes)
        print(f"Visualized {len(nodes)} nodes / {len(edges)} relationships -> {output_file}")
        return {"nodes": len(nodes), "edges": len(edges), "excluded_labels": excluded}

    @staticmethod
    def _make_self_contained(output_file: str) -> None:
        """Inline any LOCAL lib references PyVis emitted as relative paths.

        `cdn_resources="in_line"` inlines vis-network, but the select/filter-menu glue
        (lib/bindings/utils.js, lib/tom-select/*) and a stray ../node_modules ref are still
        written as relative paths that break when the .html is moved off the host. We read
        those files from beside the output (or the pyvis package) and inline them, and drop
        the unresolvable ../node_modules duplicate (vis is already inlined). Remaining http(s)
        CDN refs (bootstrap, tom-select css) are left as-is — they resolve with internet; the
        graph itself renders offline because vis-network is inlined.
        """
        import os
        import re

        out_dir = os.path.dirname(os.path.abspath(output_file))
        pkg_dir = os.path.dirname(os.path.abspath(__import__("pyvis").__file__))

        def _read_local(rel: str) -> str | None:
            for base in (out_dir, os.path.join(pkg_dir, "templates")):
                p = os.path.normpath(os.path.join(base, rel))
                if os.path.isfile(p):
                    try:
                        return open(p, encoding="utf-8").read()
                    except OSError:
                        return None
            return None

        try:
            html = open(output_file, encoding="utf-8").read()
        except OSError:
            return

        # Drop the unresolvable ../node_modules vis duplicate (vis-network is already inlined).
        html = re.sub(r'\s*<script[^>]*src="\.\./node_modules/[^"]*"[^>]*></script>', "", html)
        html = re.sub(r'\s*<link[^>]*href="\.\./node_modules/[^"]*"[^>]*/?>', "", html)

        # Inline local <script src="lib/..."> by replacing with the file contents.
        def _sub_script(m: re.Match) -> str:
            body = _read_local(m.group(1))
            return f"<script>\n{body}\n</script>" if body is not None else m.group(0)

        html = re.sub(
            r'<script[^>]*src="(lib/[^"]*)"[^>]*></script>', _sub_script, html
        )

        # Inline local <link href="lib/...css"> as a <style> block.
        def _sub_link(m: re.Match) -> str:
            body = _read_local(m.group(1))
            return f"<style>\n{body}\n</style>" if body is not None else m.group(0)

        html = re.sub(
            r'<link[^>]*href="(lib/[^"]*\.css)"[^>]*/?>', _sub_link, html
        )

        try:
            open(output_file, "w", encoding="utf-8").write(html)
        except OSError:
            pass

    @staticmethod
    def _inject_legend(output_file: str, nodes: dict) -> None:
        """Inject a fixed-position colour legend (only labels actually present)."""
        present = sorted({v["label"] for v in nodes.values()})
        items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{COLORS.get(l, "#999999")};'
            f'display:inline-block;"></span><span>{l}</span></div>'
            for l in present
        )
        legend = (
            '<div style="position:fixed;top:12px;left:12px;z-index:999;background:rgba(0,0,0,0.65);'
            'color:#fff;padding:10px 12px;border-radius:8px;font:13px sans-serif;">'
            f'<div style="font-weight:600;margin-bottom:4px;">Node types ({len(nodes)} nodes)</div>'
            f"{items}</div>"
        )
        try:
            html = open(output_file, encoding="utf-8").read()
            html = html.replace("</body>", legend + "</body>", 1)
            open(output_file, "w", encoding="utf-8").write(html)
        except OSError:
            pass

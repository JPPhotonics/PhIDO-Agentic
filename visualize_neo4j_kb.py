"""Generate a PyVis visualization of the Neo4j Knowledge Base.

Fast by default: bookkeeping nodes (ReviewItem/RelationshipObservation/SchemaRelationType)
are excluded and the layout is precomputed server-side so the browser renders static
positions (no continuous physics) — smooth even at thousands of nodes.

Examples:
    python visualize_neo4j_kb.py                       # content graph (~465 nodes), fast
    python visualize_neo4j_kb.py --include-internal     # everything, incl. 2100+ ReviewItems
    python visualize_neo4j_kb.py --limit 300 -o kb.html # cap to 300 highest-degree nodes
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.KnowledgeBase.Neo4j.visualization import Neo4jVisualizer


def main():
    ap = argparse.ArgumentParser(description="Visualize the Neo4j KB as an interactive HTML graph.")
    ap.add_argument("-o", "--output", default="neo4j_kb_graph.html", help="Output HTML path.")
    ap.add_argument("--include-internal", action="store_true",
                    help="Include bookkeeping nodes (ReviewItem/RelationshipObservation/SchemaRelationType).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap to the N highest-degree nodes (default: no cap).")
    ap.add_argument("--iterations", type=int, default=60, help="Layout iterations (tidier=more).")
    args = ap.parse_args()

    print("Initializing Neo4j Client...")
    if not os.getenv("NEO4J_PASSWORD"):
        print("Note: Using default NEO4J_PASSWORD='password'")
        print("  If your Neo4j instance uses a different password, set NEO4J_PASSWORD.\n")

    try:
        client = Neo4jClient(config=Neo4jConfig())
        client.connect()
        print("✓ Connected to Neo4j")

        print(f"Generating visualization to {args.output}...")
        viz = Neo4jVisualizer(client)
        summary = viz.visualize_graph(
            output_file=args.output,
            include_internal=args.include_internal,
            limit=args.limit,
            layout_iterations=args.iterations,
        )
        print(f"✓ Saved {summary['nodes']} nodes / {summary['edges']} edges to "
              f"{os.path.abspath(args.output)}")
        if summary["excluded_labels"]:
            print(f"  (excluded bookkeeping labels: {', '.join(summary['excluded_labels'])} "
                  f"— pass --include-internal to show them)")

    except Exception as e:
        print(f"X Error: {e}")
        if "Connection refused" in str(e) or "Can't connect" in str(e):
            print("\nMake sure the Neo4j container is running:")
            print("  sudo docker start neo4j-phido")


if __name__ == "__main__":
    main()

"""Run the PDK Ingestion Agent over the DesignLibrary (DemoPDK).

Creates PDK_Cell nodes + IMPLEMENTS/PERFORMS_FUNCTION/FABRICATED_WITH/EXHIBITS edges,
then runs Phase-6 bidirectional enrichment (Component<->PDK) internally. This is the
PDK side of the KG; run AFTER process_papers.py so Direction-A enrichment has paper-
derived Component edges to propagate down to the PDK cells.

Run with CPU embeddings:  CUDA_VISIBLE_DEVICES="" .venv/bin/python run_pdk_ingestion.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.KnowledgeBase.agents.pdk_ingestion_agent.pdk_ingestion_agent import (
    PDKIngestionAgent,
)


def main() -> None:
    print("Initializing Neo4j Client...")
    client = Neo4jClient(config=Neo4jConfig())
    client.connect()
    print("✓ Connected to Neo4j")

    agent = PDKIngestionAgent(kb_client=client, pdk_name="DemoPDK")
    report = agent.ingest()

    print("\n" + "=" * 50)
    print("PDK Ingestion Report")
    print("=" * 50)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

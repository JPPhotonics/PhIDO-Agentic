# PhIDO-Agentic

Knowledge-grounded, formally gated, agentic redesign of [PhIDO](https://github.com/JPPhotonics/PhIDO-Release), the natural-language-to-GDSII pipeline for photonic integrated circuit (PIC) design.

This repository is the code release accompanying the MASc thesis *Language-Model Agents for Photonic Integrated Circuit Design* (Y. Fu, University of Toronto, 2026). The baseline PhIDO system and its benchmark are described in [Sharma, Fu, Ansari, et al., APL Machine Learning 3, 046113 (2025)](https://doi.org/10.1063/5.0300741) and released at [JPPhotonics/PhIDO-Release](https://github.com/JPPhotonics/PhIDO-Release).

## What this adds over baseline PhIDO

- **Knowledge graph** (Neo4j): a provenance-tracked, two-layer (conceptual + fabrication) knowledge base with an OWL ontology aligned to SAREF/SOSA/BFO/QUDT/PROV (`ontology/`, `PhotonicsAI/KnowledgeBase/`).
- **Offline ingestion agents**: paper-to-graph pipeline (PPC, VSA, DIA, SEA) plus a PDK ingestion agent (`PhotonicsAI/KnowledgeBase/agents/`).
- **Online agentic pipeline**: a tool-using interpreter with typed stage schemas, bounded loops, and an optional design-intent critic, served by MCP tool servers (`mcp_servers/`).
- **Formal verification layer**: an answer-set-programming topology gate (Clingo) and an automated-reasoning parameter gate.
- **Benchmark harness**: the E1 retrieval and E2 end-to-end benchmarks, the blind three-axis human-review tooling, trace instrumentation, and scoring/significance scripts (`benchmark/`).

## Data

The evaluation datasets (benchmark definitions and golden references, the full run corpora and blind-review labels for gpt-5.4, Qwen3.6-27B, and Nemotron-3-Ultra-550B, agent traces, review instruments, and an export of the knowledge graph) are archived at the University of Toronto Dataverse (Borealis); see the thesis's Statement of Contributions for the DOI. `tools/import_kg.py` rebuilds the archived knowledge-graph export into an empty Neo4j 5.x database (round-trip verified).

## Setup

See [KG_PIPELINE_SETUP.md](KG_PIPELINE_SETUP.md) for the knowledge-graph pipeline and [GETTING_STARTED.md](GETTING_STARTED.md) for the baseline workflows ([BASELINE_README.md](BASELINE_README.md) holds the original PhIDO documentation). Python 3.12; see `pyproject.toml`. A `.env` with API keys (`OPENAI_API_KEY`, and per-provider keys as needed) and Neo4j credentials is required; no credentials ship with this repository.

## License

Same license as PhIDO-Release; see [LICENSE](LICENSE).

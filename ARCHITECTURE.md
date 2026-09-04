# PhIDO — Cross-Branch Architecture (Thesis Consolidation)

This document consolidates two bodies of work into a single architectural reference for the
master's thesis:

1. **The published baseline** — `main` branch (arXiv:2508.14123): a linear, LLM-driven
   specification-to-layout pipeline.
2. **The GraphRAG extension** — `origin/graphRAG-implementation` branch (~29k LOC, 12 commits;
   a clean, fast-forwardable descendant of `main`): a knowledge-graph-grounded, multi-agent
   redesign with formal-verification gates and an automated literature-ingestion pipeline.

> **Provenance.** Claims below were extracted from the source on both branches and
> cross-checked against the code (file:line references), not only the prose docs — several
> branch markdown files lag the implementation and are flagged where relevant. Acronym
> expansions marked *(inferred)* were derived from code/tests and should be confirmed against
> the agents' own docstrings before quoting in the thesis.

---

## 1. The two systems at a glance

| Dimension | Baseline (`main`) | GraphRAG extension (`graphRAG-implementation`) |
|---|---|---|
| Control flow | Linear forward pass, 4 stages `p100→p400` | Deterministic state machine, ~8 phases, with validation gates and automatic retry loops |
| State carrier | One mutable `session` dict, stage-namespaced keys | Structured Pydantic data contracts (`DesignIntent`, `ComponentIntent`, …) |
| Domain knowledge | PDK catalog only (component docstrings) | **Hybrid**: PDK catalog (fabrication ground truth) **+** Neo4j knowledge graph (concept ground truth) |
| Agent design | Single-pass interpreter | Interpreter → Critic → Selector, each with tool access and feedback loops |
| Disambiguation | Implicit | Explicit clarification-question phase; user answers before structuring |
| Verification | Programmatic checks (port counts, module existence) | Programmatic **+** Answer-Set Programming (Clingo) topology gate **+** AWS Bedrock Automated Reasoning parameter/architecture gate **+** LLM functional-compliance check |
| Tooling substrate | In-process function calls | Model Context Protocol (MCP) servers exposing typed tools |
| Knowledge source | Static, hand-curated library | Self-growing KG built from research papers, with schema evolution |

The conceptual leap: the baseline *generates* a circuit and checks it superficially; the
extension *grounds, decomposes, verifies, and repairs* a circuit against an evolving body of
domain knowledge and formal rules.

---

## 2. Part A — Baseline pipeline (`main`)

A Streamlit app (`PhotonicsAI/Photon/webapp.py`) turns a natural-language prompt into a
simulated GDS layout that passes DRC. State flows through a single mutable `session` dict whose
keys are namespaced by stage (`p100_*` → `p400_*`).

```
prompt
  │  p100  Entity Extraction      intent_classification → verify_input_clarity
  │                               → entity_extraction → preschematic
  │  p200  Component Spec          llm_search (BM25 + embeddings) → parse_user_specs → apply_settings
  │  p300  Schematic Generation    dot_add_edges → circuit_to_dot → edges_dot_to_yaml → placements
  │  p400  Layout & Simulation     yaml_netlist_to_gds (gdsfactory) → SAX circuit sim → circuit_optimizer
  ▼
KLayout DRC (drc/drc.py)
```

- **Representation pivot**: the circuit is carried as DSL ↔ DOT (Graphviz) ↔ gdsfactory YAML
  netlist, converted back and forth in `Photon/utils.py`.
- **LLM access**: all providers sit behind `llm_api.call_llm(prompt, sys_prompt, llm_api_selection)`;
  the per-stage model is hard-coded at the top of `webapp.py` (default `o1`).
  `OPENAI_API_KEY` is always required (Pydantic-structured formatting of extraction results).
- **Component library** (`KnowledgeBase/DesignLibrary/`): one module per component; the
  **module-level docstring is structured, searchable metadata** (Name, ports, NodeLabels,
  Bandwidth, Args). Each module exposes a `@gf.cell` builder and a `get_model()` returning SAX
  models. `_`-prefixed modules are internal building blocks.

The repo `CLAUDE.md` documents this layer in operational detail.

---

## 3. Part B — GraphRAG extension (`graphRAG-implementation`)

The extension has three loosely coupled subsystems plus a formal-verification layer:

```
  (B2) Knowledge acquisition        (B1) Knowledge representation        (B3) Design-time pipeline
  papers ──► PPC ─► VSA ─► DIA ─►    Neo4j KG  ◄── OWL ontology +     ◄── interpreter ─► critic ─► selector
                       ▲   │          (concepts, PDK cells,            ─► circuit DSL ─► layout ─► gf netlist
                       │   ▼          relationships, embeddings)               ▲
                      SEA (schema evolution)                          (B4) Clingo + Bedrock AR gates
```

### 3.1 (B1) Knowledge representation — Neo4j knowledge graph

**Backend.** Neo4j 5.15-enterprise (APOC + Graph Data Science plugins), `bolt://localhost:7687`,
defined in `docker-compose.yml`. **Neo4j is the production backend** — confirmed by
`process_papers.py:25`, `agents/ppc_agent/ppc_agent.py:7`, `kb_grounding_tool.py:8`, and
`mcp_servers/kg_server.py`. **ArangoDB** (`KnowledgeBase/ArangoDB/`) is the earlier backend,
now **legacy** (the switch is commit `15b26e0`; the `PPC_AGENT_ARANGODB_PIPELINE.md` doc
describes the superseded Arango path).

**Ontology / type system.** An OWL file `KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl`
defines the class hierarchy; seed instance knowledge lives as per-primitive YAML under
`GenerativeOntology/Primitives/` (e.g. `_mzi.yaml`, `_microring_modulator.yaml`,
`design_functions.yaml`, `properties.yaml`). Loaded into Neo4j via `reinitialize_neo4j_kb.py`.

**Node types**: `Component`, `Architecture`, `Property`, `Design_Function`,
`Physical_Principle`, `Document`, `PDK_Cell` (+ `PDK_Cell_History`).
**Relationship types**: `PERFORMS_FUNCTION`, `BASED_ON_PRINCIPLE`, `HAS_PROPERTY`,
`USES_COMPONENT`, `RELATED_TO` (generic, bidirectional), `EXTRACTED_FROM` (provenance), plus
PDK-linkage edges (`IMPLEMENTS`, `COMPOSED_OF`, `FABRICATED_WITH`, `SUPERSEDES`).
**Embeddings**: 1024-dim from `Qwen/Qwen3-Embedding-0.6B`, text = `name + description + equations`,
served via Neo4j native vector indexes (cosine).

**Retrieval modes**: vector semantic search, exact/fuzzy name lookup, neighbor/multi-hop
traversal, and hybrid (vector + relationship filter). Neo4j **GDS graph algorithms** are used
to surface *candidate implicit relationships* for the DIA agent (commit `dd352db`).

**`models/architecture_template.py`**: a structured circuit-topology blueprint
(`ComponentRole`, `AbstractConnection`, `ArchitectureTemplate` with `topology_class`,
`scaling_rules`, `variants`, `confidence`). Used for architecture *decomposition*, distinct
from KG retrieval.

### 3.2 (B2) Knowledge acquisition — paper → KG pipeline

Orchestrated by `process_papers.py`; four agents run per PDF, with a fifth running per batch.
*Acronym expansions below are inferred from code/tests — verify before quoting.*

1. **PPC — Paper Processing & Concept agent** *(inferred)* (`agents/ppc_agent/`).
   PDF → markdown → section filtering → ontology-guided entity extraction → rich descriptions
   (evidence quotes, key metrics) → **normalization** against the KG by vector search →
   conflict detection (known vs. new vs. ambiguous, thresholds 0.8 / 0.4 with an LLM judge in
   between, and an exact-name override). Acronyms resolved via `acronym_agent` against the
   persistent `acronym_mappings.json` (81 photonics acronyms). Output: known entities + new concepts.
2. **VSA — Validation & Synthesis agent** *(inferred)* (`agents/vsa_agent/`).
   Architecture-integrity gate (rejects vague architectures) → explicit `USES_COMPONENT` edges
   → **deep inference** of implicit edges (novel types logged as `RelationshipObservation` for
   SEA) → knowledge merge into existing entities → a **Transaction Manifest** (`VSAUpdatePayload`)
   with `EXTRACTED_FROM` provenance edges.
3. **DIA — Database Integration agent** *(inferred)* (`agents/dia_agent/`).
   Executes the manifest into Neo4j (insert/update/merge) and runs **global inference** —
   GDS-assisted discovery of implicit relationships between new entities and the existing KB,
   LLM-verified in batches; low-confidence results queued as `ReviewItem`s.
4. **SEA — Schema Evolution agent** (`agents/sea_agent/`).
   Per-batch: clusters accumulated `RelationshipObservation`s (agglomerative, cosine 0.85),
   applies statistical promotion gates (≥3 distinct documents, mean confidence ≥0.70,
   directional consistency ≥0.6), LLM-validates each candidate as genuinely novel, then
   **promotes** it to a first-class schema type and **recategorizes** matching `RELATED_TO`
   edges (auto-commit ≥0.85, else human review). This is the mechanism by which the graph
   schema *grows from evidence* across the corpus.

**Human-in-the-loop**: `review_queue_app.py` (Streamlit) surfaces queued nodes/edges for
approve/reject/re-commit.

### 3.3 (B3) Design-time agentic pipeline + MCP servers

Three MCP servers expose typed tools; `mcp_servers/pipeline_orchestrator.py` is a deterministic
state machine (not itself a server) that chains them. `mcp_servers/streamlit_app.py` is the UI.

- **`kg_server.py`** — ontology + Neo4j retrieval: `search_concepts`, `resolve_function`,
  `get_concept_neighborhood`, `get_component_properties`, `get_pdk_implementations`,
  `get_pdk_cell_details`, `search_pdk_by_function`, `kg_stats`.
- **`pdk_catalog_server.py`** — fabrication ground truth via gdsfactory: `list_all_components`,
  `search_components`, `get_component_details`, `validate_port_config`, `get_module_params`,
  `get_component_footprint`, `get_port_names`, `validate_selection`.
- **`schematic_builder_server.py`** — circuit assembly: `circuit_dsl_to_dot`, `check_planarity`,
  `compute_layout`, `find_open_ports`, `export_gf_netlist`.

**Orchestrated phases** (with gates marked ◆):

```
0–1.75  Interpreter exploration   (interpreter_agent.explore_and_ask)
        LLM tool-calling over PDK + KG; KG-grounding gate forces verification
        instead of relying on training data; emits clarification questions to the user.
2       Structured extraction      → DesignIntent (only PDK/KG-verified claims; assumptions flagged)
◆2.5    Topology gate (A)          validate_topology()  — Clingo ASP rules            [pipeline_orchestrator.py:945]
3       Critic review              independent LLM verifies completeness/faithfulness; up to N rounds
4       Component selection        PDK search + LLM mapping intent → concrete module
◆4.75   Selection validation       programmatic + PDK instantiation check
4.8     LLM compliance check       device-type faithfulness (ring-mod ≠ MZI-mod)
5       Circuit DSL construction   real gdsfactory defaults + user overrides → DSL → DOT
◆5.25   Parameter gate (B)         validate_parameters() — Bedrock AR                 [pipeline_orchestrator.py:1401]
5b      LLM edge routing           port-level edges + planarity retry loop
6       Layout computation         Graphviz placements (µm)
6.5     Schematic validation       planarity / connectivity / orphans
7       Export                     gdsfactory YAML netlist → GDS
```

A failed *fundamental* gate (topology or selection) re-invokes the interpreter with the error
as feedback (up to `_MAX_VALIDATION_RETRIES`), rather than failing forward.

### 3.4 (B4) Formal-verification gates — **live in the pipeline**

Both validators are imported and called by the orchestrator (verified at
`pipeline_orchestrator.py:65–66, 945, 1244, 1401`) — the "prepared but unused" note in older
branch docs is **stale**.

- **`clingo_validator.py` — topology gate (Checkpoint A).** Serializes `DesignIntent` into
  Clingo/ASP facts (`component/3`, `connection/2`, `architecture/1`, `n_value/1`) and solves
  against `architecture_rules.lp` encoding invariants for MZI, splitter/Benes/Clements/Reck
  trees, QPSK, and WDM (e.g. *MZI = exactly one splitter + one combiner on a connected
  interferometric path*). Returns structural error atoms early, before any expensive LLM
  critic pass. Degrades gracefully (returns empty) if Clingo is absent.
- **`ar_validator.py` — architecture + parameter gates (Checkpoint B).** Serializes design
  facts into natural-language premises/claims and calls **AWS Bedrock `ApplyGuardrail`
  Automated Reasoning** against a physical-bound policy — per-component for parameters (width,
  radius, gap, coupling, …) to keep each query small. Non-blocking (severity *major*); gated by
  an `AR_ENABLED` env flag and degrades gracefully without credentials.

New dependencies for this layer: `clingo>=5.7.0` and `boto3>=1.34.0` (added to
`requirements.txt`/`pyproject.toml` on the branch).

---

## 4. How the two map onto each other (consolidation view)

The agentic pipeline is best read as the baseline's four stages *re-decomposed and hardened*:

| Baseline stage | GraphRAG counterpart | What changed |
|---|---|---|
| `p100` Entity Extraction | Phases 0–3 (interpreter exploration, structured `DesignIntent`, critic) | KG grounding, explicit clarification questions, independent critic, structured contracts |
| `p200` Component Spec | Phase 4 + 4.75 + 4.8 | Selection now PDK-instantiation-validated and LLM-compliance-checked for device-type faithfulness |
| `p300` Schematic Generation | Phases 5, 5b, 6, 6.5 | Same DOT/planarity machinery (shared lineage with `utils.py`), now behind an MCP server with retry loops |
| `p400` Layout & Simulation | Phase 7 export → gdsfactory | Largely shared; gdsfactory YAML netlist is the common interface |
| (none) | Topology gate (Clingo) + Parameter gate (AR) | New formal-methods layer with no baseline equivalent |
| (static library) | Neo4j KG + paper-ingestion pipeline | New knowledge substrate that *grows* |

This table is a natural backbone for a thesis "system evolution" chapter: each row is a claim
about *what grounding/verification was added and why*.

---

## 5. Repository & infrastructure inventory (branch additions)

- **KG backends**: `KnowledgeBase/Neo4j/` (primary), `KnowledgeBase/ArangoDB/` (legacy).
- **Ontology**: `KnowledgeBase/GenerativeOntology/` (`pic_ontology.ttl` + `Primitives/*.yaml`).
- **Agents**: `KnowledgeBase/agents/{ppc_agent, vsa_agent, dia_agent, sea_agent, pdk_ingestion_agent}/`.
- **MCP layer**: `mcp_servers/` (3 servers + orchestrator + interpreter + 2 validators + Streamlit UI).
- **Ingestion / ops**: `process_papers.py`, `reinitialize_neo4j_kb.py`, `neo4j_backup.sh`,
  `start_neo4j.sh`, `docker-compose.yml`, `review_queue_app.py`, `visualize_neo4j_kb.py`.
- **Docs (branch)**: `KG_PIPELINE_SETUP.md`, `NEO4J_README.md`, `CONNECT_TO_DOCKER.md`,
  `PPC_AGENT_ARANGODB_PIPELINE.md` (legacy), `Supplemental_Experiment_Description.md`.
- **Runbook (per `KG_PIPELINE_SETUP.md`)**: start Neo4j → set `.env`
  (`NEO4J_*`, `EMBEDDING_MODEL`, `GOOGLEGENAI_API_KEY`, optional `AR_ENABLED`/AWS) →
  `reinitialize_neo4j_kb.py` (seed ontology) → drop PDFs in `papers/` →
  `process_papers.py` → optional `review_queue_app.py` → `neo4j_backup.sh backup`.

### Thesis-relevant experiment
`Supplemental_Experiment_Description.md` defines a **PDK-scaling retrieval study**: 262
gdsfactory components, three library sizes (50 sampled from first 50 / first 131 / all 262),
LLM-generated *functional* queries (not name matches), measuring pass@1, pass@3, mean rank,
and token cost; Gemini 2.5 Pro vs. OpenAI o1 at temperature 0.1. This is a self-contained
evaluation chapter on how retrieval accuracy and cost scale with library size — a likely core
quantitative result.

---

## 6. Maturity, transitions, and open issues (honest assessment)

For thesis accuracy, the following should be stated rather than smoothed over:

1. **ArangoDB → Neo4j migration is incomplete in the docs, complete in the code.** Production
   paths use Neo4j; Arango code and the `PPC_AGENT_ARANGODB_PIPELINE.md` doc remain and will
   confuse a reader unless labeled legacy.
2. **Acronym expansions (PPC/VSA/DIA) are inferred** from code and tests — confirm against the
   agents' own docstrings before printing them in the thesis.
3. **Repository hygiene**: the branch commits many `__pycache__/*.pyc` files; a `.gitignore`
   was added on the branch but the tracked artifacts should be removed before any merge.
4. **External-service dependence**: the AR gate requires AWS Bedrock credentials and a policy;
   without `AR_ENABLED`/credentials it silently no-ops. Reproducibility claims must note this.
5. **Unmerged sibling work**: `origin/tidy3d_integration` exists and was *not* examined here —
   decide whether it is in or out of thesis scope.
6. **Verification depth**: this document's pipeline-phase and KG-schema claims are code-checked;
   the DIA/PPC internal behavior is partly inferred from tests (`test_dia_agent.py`,
   `test_ppc_agent.py`) rather than a full read of every agent module.

---

## 7. Suggested next steps

- **Confirm the acronyms and DIA/PPC internals** with a focused read of each agent's module
  docstring, then lock the terminology used throughout the thesis.
- **Decide the consolidation target**: a single merged branch (with `.pyc` cleanup and an
  ArangoDB-deprecation note) vs. keeping the baseline pristine and treating GraphRAG as an
  overlay. Section 4's mapping table works for either.
- **Reproduce the PDK-scaling experiment** to regenerate figures with current models.

---

*Generated as a consolidation reference; figures and exact metrics still need to be produced
from the experiments themselves.*

# PhIDO Thesis — Evaluation Overview
---

## 1. The thesis claim, in one paragraph

The baseline (`main`, arXiv:2508.14123) turns a natural-language prompt into a DRC-clean GDS
layout through a linear LLM pipeline that retrieves components by text similarity (BM25 +
embeddings). The GraphRAG extension (`origin/graphRAG-implementation`) replaces that with an
agentic pipeline **grounded in a knowledge graph (KG) that is built automatically from research
papers**, plus formal-verification gates. **The claim we must defend: grounding the design task
in the KG measurably produces better photonic designs than the baseline — and the advantage grows
as the component search space grows.** Everything we test is in service of supporting, qualifying,
or honestly bounding that single claim.

## 2. How we test — three principles

1. **Extrinsic-primary.** The KG is *instrumental*, not the product. The headline result is a
   design-task improvement over the baseline. Everything about the KG's internal quality is a
   **diagnostic** that explains *why* the design number moved — a beautiful KG that doesn't improve
   design is a failure.
2. **KG vs. baseline is the spine.** Every comparison holds the LLM, prompts, and component
   descriptions fixed across arms so the **KG is the only variable**. We measure the *treatment*,
   not a model upgrade.
3. **Honest funnels and bounds.** End-to-end success is reported as a staged funnel (where designs
   die), not a single pass/fail. We state explicitly what each test does **not** prove (e.g.
   "manufacturable ≠ functionally correct").

## 3. The two headline experiments (the whole claim rests here)

| | **E1 — Retrieval A/B** | **E2 — End-to-end A/B** |
|---|---|---|
| Question | Does KG-grounded selection pick the right component(s)? | Does the whole KG pipeline produce better designs? |
| Input | Auto-generated functional queries, 3 PDK sizes (50/131/262) | ~100 NL design prompts (`Testbench.xlsx`), tiered Level 1–4 |
| Arms | BM25+embedding / LLM-over-JSON / **KG-grounded** | Baseline pipeline vs **GraphRAG pipeline** |
| Output | pass@1/3, MRR, set P/R, tokens | staged funnel: instantiate → routing_ok → models → sim-success → DRC-clean |
| Ground truth | source component per query (free) | self-labeling via DRC (free) |
| Distinctive result | **KG advantage grows with PDK size** | **per-stage** comparison shows *where* GraphRAG helps |

Everything in §4 either **feeds** these two experiments or **diagnoses** their results.

## 4. Feature inventory — what we test and how

Two groups: the **offline** machinery that builds the KG, and the **online** pipeline that uses it
to design. For each feature: what it does, what we test about it, and how that connects to the
headline. (Metric-level detail and ground-truth sources live in `BENCHMARK_ARCHITECTURE.md` §5.)

### 4a. Offline — building the knowledge graph (paper → KG)

These are *diagnostics*: they explain the quality of the knowledge that the online pipeline draws on.

| Feature | What it does | What we test | How (in plain terms) |
|---|---|---|---|
| **A1 — PPC** (Pre-processing & Context) | Reads a paper, extracts photonic concepts, normalizes them against the KG (known vs. new) | Are extracted entities correct, and is the known/new decision right? | Extraction F1 against a small gold-annotated paper set; precision/recall on the "is this already in the KB" call; duplication rate |
| **A2 — VSA** (Validation & Synthesis) | Infers relationships (edges) between concepts, with a confidence score that drives auto-commit | Are the edges *supported by the source text*, and is the confidence trustworthy? | **Evidence-grounded faithfulness** — sample edges, check the stored quote actually supports them; edge-type confusion matrix; **confidence calibration** (does 0.8 really mean 80%?) to justify the auto-commit thresholds |
| **A3 — DIA** (Database Integration) | Writes the new knowledge into Neo4j and discovers implicit links to existing knowledge | Does it find the right cross-document links, and write them safely? | Two-stage: candidate **recall@K** (can it find the link at all?) + a verification gate + final edge precision; transaction-integrity check on the database writes |
| **A4 — SEA** (Schema Evolution) | Grows the graph schema itself — promotes recurring patterns into first-class relationship types | Does the schema **converge** as papers accumulate, or drift and add noise? | **Systems-dynamics first, no labels:** type-growth curve, churn, order-invariance (does paper order change the schema?); threshold-sensitivity sweep; recategorization accuracy |
| **A5 — PDK ingestion** | Imports the gdsfactory component library into the KG and links concepts to fabricable cells | Are component nodes, ports, params, and concept↔cell edges correct? | Node and per-edge-type P/R against the library itself (cheap — the PDK is enumerable). **This bounds the ceiling of E1's KG arm.** |
| **A6 — Neo4j backend** | Stores and serves the KG | Does it stay consistent, and does it scale? | Graph-integrity constraints (health); query/GDS-build latency vs. graph size (scalability). No standalone quality metric — validated through the features that use it |

### 4b. Online — using the KG to design (prompt → GDS)

These run on **both pipelines** wherever a baseline equivalent exists, so each is an A/B contrast.

| Feature | What it does | What we test | How (in plain terms) |
|---|---|---|---|
| **B1 — Interpreter / clarification** | Reads the prompt, asks clarifying questions when ambiguous, emits a structured intent | Does it ask when it should (and stay quiet when it shouldn't)? Is the intent correct? | **should-clarify precision/recall** on a curated clear/ambiguous set; intent correctness vs. gold; on/off **ablation** in E2 (needs a user simulator to answer questions) |
| **B2 — KG-grounded selection** | Picks concrete PDK components for the intent | **This *is* E1** — the core retrieval claim | pass@1/3, MRR, **set P/R** for multi-component queries; **scaling** across 3 PDK sizes; 3 arms |
| **B3 — Schematic builder** | Wires selected components into a circuit topology | Is the topology *correct*, not just valid? | edge-F1 / graph-edit-distance vs. gold topologies on a curated subset; netlist validity + port-compat on the full suite; both arms (validity ≠ correctness) |
| **B4 — Clingo topology gate** | Formal ASP rules reject structurally invalid circuits | Which topology errors does it actually catch? | **Formal-gate framing:** per-error-class catch rate over a defined error taxonomy + coverage-gap analysis (which real errors are *not* encoded) + false-reject on valid designs + on/off ablation on the full success funnel |
| **B5 — AR parameter gate** | AWS Bedrock Automated Reasoning checks parameters against physical bounds | Which parameter errors does it catch, and what does the external service cost? | Same formal-gate framing over a *parameter*-error taxonomy; plus AR cost/latency/availability as an explicit operational finding (external, paid → cached, run on a subset) |
| **B6 — Layout & simulation (+ DRC)** | Builds the GDS, runs SAX sim, runs KLayout DRC | Does the design actually instantiate, simulate, and pass DRC? | **This is the E2 headline** — the staged success funnel, complexity-stratified, self-labeling; compare baseline vs GraphRAG at *each* stage |
| **C — Infrastructure** (orchestration/queue/UI) | Runs the agentic pipeline | Is the agentic overhead worth it? | **Cost A/B** (tokens / latency / retry rounds vs. the linear baseline); human review-queue burden; critic on/off ablation on E2 |

## 5. How the pieces roll up into the claim

```
   OFFLINE diagnostics              ONLINE A/B tests                 HEADLINE
   (why is the KG good/bad?)        (does it help?)                  (the thesis claim)

   A1 PPC   ─┐
   A2 VSA   ─┤ quality of the        B2 selection ───────────────►  E1  KG retrieval
   A3 DIA   ─┼─ knowledge feeding ─► (= E1)                          beats baseline,
   A4 SEA   ─┤  retrieval & wiring                                   advantage grows
   A5 PDK   ─┘  (A5 bounds B2)        B1 interpret ─┐                 with PDK size
   A6 Neo4j ──  serves it all         B3 schematic ─┤
                                      B4 Clingo    ─┼─ each gate's ► E2  GraphRAG funnel
                                      B5 AR        ─┤  funnel Δ        beats baseline
                                      B6 layout    ─┘                 stage-by-stage
                                      C  infra ──── cost that the
                                                    lift must justify
```

Read top-to-bottom: offline features explain the *quality* of the knowledge; online features turn
that knowledge into a *design advantage*; E1 and E2 are where the advantage becomes a defensible
number. **Ablations** (turn each piece off and re-measure E2) convert "the KG helps" into "*which
part* helps, and by how much."

## 6. What we deliberately do NOT test (scope boundaries — state these up front)

- **Functional-spec correctness.** The E2 funnel proves a design is *manufacturable and
  simulatable*, **not** that it meets the requested spec (FSR, splitting ratio, etc.). True
  functional verification is tied to the deferred `circuit_optimizer` and needs per-prompt target
  responses → **future work**.
- **Model sensitivity.** We hold a single canonical model (o1) fixed across all arms so the KG is
  the only variable. Comparing models (e.g. Gemini 2.5 Pro vs o1) is **future work**, not run.
- **The UI**, and any backend bake-off (no Neo4j-vs-ArangoDB — ArangoDB is deprecated).
- **`origin/tidy3d_integration`** — unexamined, out of scope unless explicitly pulled in.

## 7. Cross-cutting test discipline (applies to everything above)

- **Fairness:** same LLM, same prompts, same component descriptions across arms; the KG is the only
  added information.
- **Statistics:** paired design (same prompts both arms) → McNemar's test for pass-rate funnels,
  bootstrap CIs for pass@k; fixed temperature + seed; always report N and a per-tier (Level 1–4)
  breakdown.
- **Isolation:** every stage tested twice — once with clean gold input (clean attribution) and once
  in the full end-to-end run (compounding, realistic).
- **Ground-truth economy:** annotation effort concentrates on A1–A4 (gold sub-graph) and B4/B5
  (error taxonomies); A5, B2, and B6 are near-free (PDK is enumerable, DRC self-labels).
- **No silent caps:** any time coverage is bounded (top-N, sampling, no-retry), it is logged, not
  smoothed over.

## 8. Build order (what gets built first to produce these results)

Cleanest-first, dependency-aware (full backlog in `BENCHMARK_ARCHITECTURE.md` §8):

1. **E1 retrieval harness** — cleanest central result, fully automatable, no manual annotation.
2. **Headless baseline runner** — `main` is Streamlit-driven; GraphRAG already has `run_pipeline()`.
3. **E2 funnel harness** — enabled by the re-integrated DRC (commit `8e819f2`).
4. **Gold sets & taxonomies** — annotated paper set + fixed KB snapshot (A1–A4); B4/B5 error
   taxonomies; B3 gold topologies; B1 clear/ambiguous set.
5. **User simulator** — shared component for B1 clarification ablation and interactive E2 runs.
6. **AR caching layer** — keeps the external Bedrock calls cheap and reproducible for B5.
```

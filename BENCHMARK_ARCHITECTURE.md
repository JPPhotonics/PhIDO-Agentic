# PhIDO-GraphRAG Benchmark Architecture

> Status: **DRAFT — under active design.** This document is the agreed backbone for
> benchmarking the GraphRAG extension against the published PhIDO baseline, for the
> master's thesis. It is a *design artifact*, not an implementation.
> **DESIGN COMPLETE (2026-06-08):** all per-feature metrics (A1→C) DECIDED **and** all cross-cutting
> methodology (§6, items 1–6) LOCKED. No implementation has begun — build backlog in §8.
>
> Companion docs: `ARCHITECTURE.md` (cross-branch system writeup), `CLAUDE.md` (baseline pipeline),
> `KG_EVALUATION_LITERATURE.md` (fact-checked KG-evaluation lit review; §6 maps its findings onto the L0–L3 layers below, §4a sizes annotation IAA).
> Last updated: 2026-06-08.

---

## 1. Purpose & scope

Define a single, coherent benchmark that judges the GraphRAG knowledge-graph (KG) pipeline
on `origin/graphRAG-implementation` against the published baseline on `main`
(arxiv 2508.14123). The benchmark must produce thesis-grade, reproducible claims.

Two systems under test:
- **Baseline** (`main`): linear LLM pipeline; component retrieval via BM25 + embedding (`llm_search`).
- **GraphRAG** (`origin/graphRAG-implementation`): agentic, KG-backed; offline paper→KG
  ingestion (PPC→VSA→DIA→SEA + PDK ingestion) + online query→GDS pipeline with KG-grounded
  selection and formal-verification gates.

## 2. Guiding principle — the KG is *instrumental*

The KG is not the product; it exists to make **downstream photonic design better**. Therefore:

- **Headline claim is EXTRINSIC**: the KG measurably improves the design task vs. the baseline.
- **Intrinsic metrics are DIAGNOSTICS**: they explain *why* the extrinsic number moved. A KG with
  excellent entity-F1 that does not improve design is a failure.
- **KG vs. baseline (BM25+embedding) is a central, explicit comparison** (locked decision).

## 3. Evaluation layers

| Layer | Question | Role | Cost |
|---|---|---|---|
| **L3 Extrinsic** | Does the KG make design better than baseline? | **Headline** | medium (harness) |
| **L2 Intrinsic, reference-based** | Is each stage's output correct vs gold? | Supporting | high (gold annotation) |
| **L1 Intrinsic, reference-free** | Is the graph faithful & healthy at scale? | Supporting | low (uses stored evidence) |
| **L0 Operational** | Cost, latency, determinism, human burden | Practicality | low |

**L1 reference-free signals** (scalable, no gold): *faithfulness/groundedness* — every node/edge
stores `evidence_quotes`; sample and judge whether the quote actually supports the triple
(precision-like, no recall ground truth needed); *graph health* — duplication rate, contradiction
rate, orphan nodes, schema conformance.

### Three hard problems specific to this pipeline
1. **No canonical ground truth** for an open-ended KG → lean on L1 for scale, L2 only as a small anchor.
2. **"Novel vs. wrong"** — an inferred edge absent from the reference may be a genuine discovery, not an
   error. Evidence-grounded **precision** is robust to this; **recall** is not (report recall only on the
   gold anchor, or with pooled judgments).
3. **Moving target** — SEA evolves the schema across papers (non-stationary). Evaluate both **per-paper
   increments** and **cumulative convergence/stability**.

## 4. The two extrinsic experiments (L3 — the claim)

### E1 — Retrieval A/B (component selection)
Rebuild the methodology in `Supplemental_Experiment_Description.md` (code is lost; only prose survives).
Three arms on the **same** auto-generated functional queries across **3 PDK sizes** (50 small / 50 medium
/ 50 large, sampled from the 262-component GDSFactory library):
- baseline-A: BM25 + embedding (`main` `llm_search`)
- baseline-B: LLM-over-JSON-context (the Supplemental's own method)
- **GraphRAG: KG-grounded selection** (`pdk_catalog_server` + KG)

Ground truth = the source component per query. Metrics: **pass@1, pass@3, MRR/average rank, error rate,
tokens/query**. Fully automatable, no manual annotation. *Cleanest central result; build first.*

**Query distribution (LOCKED):** paraphrase queries (Supplemental-style) **plus functional/indirect queries**
("a component that does X") — the latter exercise the KG's `PERFORMS_FUNCTION`/`BASED_ON_PRINCIPLE` edges
and are *required* for the KG advantage to be observable (on paraphrase-only queries embeddings already win).

**Scaling hypothesis (LOCKED):** run all arms across the 3 PDK sizes and test whether the **KG advantage
grows with search-space size** (embedding/BM25 precision degrades with more near-duplicates; KG structure
stays discriminative). "KG helps more at scale" is the distinctive claim.

**Compositional regime (LOCKED):** beyond single-component pass@k, measure correct-**set** retrieval
(set P/R) for queries needing multiple components — closer to real design, where KG-grounding should help most.

**Fairness controls:** same LLM, same component descriptions, same queries across arms; the KG is the *only*
added information (the treatment). Grounding precision (are KG-returned components valid for the intent?) as
a secondary check.

### E2 — End-to-end A/B (prompt → GDS)
~100 NL design prompts from `Testbench.xlsx` through **both whole pipelines**, reported as a **staged
success funnel** (not a single number):

```
GDS instantiates → routing_ok → SAX models present → sim-success → DRC-clean
```

Stratified by the Level 1–4 complexity tiers; compare baseline vs GraphRAG at *each* stage. Enabled by the
DRC re-integration (commit 8e819f2). **Self-labeling** — no manual ground truth.

**Caveat (state explicitly):** passing the funnel = **manufacturable & simulatable, NOT functionally
correct.** A design can be DRC-clean yet be the wrong circuit (B3 topology error) or miss its spec. True
functional-spec verification (sim response vs requested FSR/splitting-ratio/etc.) is a **known gap**, tied to
the deferred `circuit_optimizer`, and requires per-prompt target responses → future work.

## 5. Per-feature diagnostic map

### OFFLINE — KG construction (ingestion)

| # | Feature | Diagnostic metric | Layer | Ground truth | Explains in L3 | Status |
|---|---|---|---|---|---|---|
| A1 | PPC | extraction F1 (P/R + 5-type confusion); known-vs-new P/R (NIL); dup-rate | L2+L1 | gold paper set + KB snapshot | quality of KG nodes feeding retrieval | **DECIDED** |
| A2 | VSA | **evidence-grounded edge faithfulness** (L1, primary precision); **edge-type confusion matrix**; **confidence calibration** (reliability/ECE/AUROC — justifies the 0.8/0.5 thresholds); gold-recall on anchor only; arch-gate via gate template (§6.6) → E2 ablation | L2+L1 | gold sub-graph + evidence quotes | richness/correctness of relations grounding traverses; validity of confidence-driven auto-commit | **DECIDED** |
| A3 | DIA | **Global Inference (two-stage):** candidate-retrieval **recall@K** (GDS) + semantic-verification **gate** (§6.6) + end-to-end edge **precision** (judge), recall on anchor only. **Manifest exec:** transaction integrity (pass/fail) + L0; schema conformance → L1 | L2+L1+L0 | gold cross-doc links + fixed KB snapshot | whether KG usefully links new concepts to existing knowledge | **DECIDED** |
| A4 | SEA | **Primary:** schema **convergence/stability** (type-growth curve, churn, **order-invariance**) — no labels. **+ threshold sensitivity sweep** (3 docs / 0.85 cosine / 0.70 conf; 0.70 ties to A2 calibration). **Recategorization accuracy** (L2). Promotion precision = sampled expert judgment; review-queue split (L0) | L2+L1+L0 | human type-judgments (sample) + labeled RELATED_TO→type | does evolving schema converge & help, or drift/add noise | **DECIDED** |
| A5 | PDK ingestion | **node P/R** (+ ports/params) + **per-edge-type P/R** (IMPLEMENTS/PERFORMS_FUNCTION/FABRICATED_WITH/EXHIBITS) + **topology-completion acc**; embeddings→B2; versioning=correctness check | L2 (cheap) + L0 | the DesignLibrary itself (enumerable) | **bounds B2's ceiling** — KG↔PDK bridge; ingestion anchor | **DECIDED** |
| A6 | Neo4j backend | graph-integrity constraints (→L1) + **KG-query/GDS latency vs graph size** (scalability, L0); no standalone quality metric | L1+L0 | n/a | validated transitively via B2/A3; scalability of the approach | **DECIDED** |

### ONLINE — query → design

| # | Feature | Diagnostic metric | Layer | Ground truth | Explains in L3 | Status |
|---|---|---|---|---|---|---|
| B1 | Interpreter/clarification | **should-clarify P/R** (2×2 on curated clear/ambiguous set, primary) + intent-extraction correctness vs gold DesignIntent + question usefulness/efficiency + **E2 clarification on/off ablation** (needs user simulator) | L2+L3+L0 | curated clear/ambiguous queries + gold DesignIntent | upstream intent quality; does clarification improve end-to-end | **DECIDED** |
| B2 | KG-grounded selection | **= E1.** pass@1/3, MRR/rank, tokens + **set P/R (compositional)** + grounding precision. Queries: **paraphrase + functional/indirect**. Axis: **scaling vs 3 PDK sizes**. Arms: BM25+emb / LLM-context / KG | **L3** | source component per query; gold component sets | **the core retrieval claim** | **DECIDED** |
| B3 | Schematic builder | **topology correctness** (edge-F1/GED vs gold) on curated subset + netlist validity + port-compat on full suite; measured on both arms | L2 | curated gold topologies (Level 1–4 + representative Testbench) | validity≠correctness; did better selection → better wiring | **DECIDED** |
| B4 | Clingo topology gate | **rule-adequacy:** solver-correctness sanity + **per-error-class catch rate** (error taxonomy) + coverage-gap analysis + false-reject; **funnel ablation** (routing/sim/DRC) | L2+L3+L0 | taxonomy-spanning labeled invalid+valid topologies | which topology errors the rules catch; contribution to full success funnel | **DECIDED** |
| B5 | AR parameter gate | **formal-gate framing** (§6.6) over **parameter-error taxonomy** + false-reject + funnel ablation; **AR cost/latency/availability (L0)**; cached results on representative subset | L2+L3+L0 | taxonomy-spanning labeled params | which param errors caught; gate contribution; cost of external AR | **DECIDED** |
| B6 | Layout & sim (+DRC) | **staged success funnel** (instantiate→routing_ok→models→sim-success→DRC-clean), complexity-stratified; self-labeling. Functional-spec verification = deferred gap (optimizer) | **L3** | automatic (self-labeling) | **the E2 headline**; where designs die | **DECIDED** |
| C | Infra (queue/UI/orch) | **cost A/B** (agentic vs linear baseline: tokens/latency/rounds/cap-hits) + **review burden** (auto/queue/reject split; ties to calibration) + **critic on/off ablation** (E2 Δ) | L0+L3 | n/a | cost-vs-benefit of the agentic approach; practicality | **DECIDED** |

## 6. Cross-cutting methodology (to be locked during walkthrough)

1. **Isolation convention** — *LOCKED.* Per-stage **gold input** (clean attribution) **+** one **end-to-end**
   run (compounding headline). Both.
2. **Ground-truth strategy by region** — *LOCKED (derived).* Annotation concentrates on **A1–A4 gold
   sub-graph + B4/B5 taxonomy sets**; **A5 / B2 / B6 near-free** (PDK enumerable; DRC self-labels).
3. **Statistical rigor** — *LOCKED.* Paired design (same prompts both arms): **McNemar's test** for funnel/
   DRC pass rates, **bootstrap CIs** for pass@k; fixed temp + seed for determinism; report N **+ per
   complexity-tier (Level 1–4) breakdown**.
4. **Ablation set** — *LOCKED (derived union).* Toggles: KG-grounding on/off, deep-inference on/off,
   schema-evolution on/off, each gate (VSA-arch / Clingo / AR) on/off, critic on/off, clarification on/off.
   Converts "KG helps" into "*which part* helps."
5. **Model control** — *LOCKED.* **Single canonical model held fixed across arms** (the KG is the only
   variable) — default o1. Model-sensitivity (e.g. Gemini 2.5 Pro vs o1) noted as **future work**, not run.
6. **Unified gate-eval template** — *LOCKED (2026-06-08).* All "gate" components are measured by one
   reusable pattern: **(a) catch rate** (true-positive on a labeled *invalid* set), **(b) false-reject
   rate** (false-positive on a labeled *valid* set), **(c) ablation Δ on E2** (gate on/off → change in
   DRC-clean rate). Applies to: **VSA architecture-integrity gate (A2)**, **DIA semantic verification
   (A3)**, **Clingo topology gate (B4)**, **AR parameter gate (B5)**. Each needs a labeled valid+invalid
   test-set (annotation cost concentrated here). Soundness note: for the formal gates (B4/B5) also report
   whether any *truly* invalid case ever passes (soundness violation), which is stronger than catch rate.
   **Formal-gate extension (B4/B5):** a correct ASP/AR gate is sound+complete *w.r.t. its encoded rules*,
   so its failure mode is **rule incompleteness**, not stochastic error. Evaluate **rule adequacy**: (i)
   solver-correctness sanity (~100% expected), (ii) **per-error-class catch rate** over a defined error
   taxonomy + coverage-gap analysis (which real error classes are *not* encoded), (iii) false-reject on
   valid controls, (iv) ablation on the **full success funnel** (routing_ok → sim-success → DRC-clean), since
   topology/parameter errors manifest upstream of DRC.

## 7. Existing assets

| Asset (both branches) | What it is | Benchmark role |
|---|---|---|
| `Testbench.xlsx` | ~100 NL end-to-end design prompts | E2 prompt suite |
| `Supplemental_Experiment_Description.md` | retrieval-scaling methodology (pass@1/3, 3 PDK sizes) | E1 spec (code must be rebuilt) |
| `GETTING_STARTED_EXAMPLE_OUTPUTS/Level 1–4` | 4 worked prompts w/ per-stage reference outputs | per-stage gold anchors + complexity tiers |
| `PhotonicsAI/Photon/drc/` | KLayout DRC script + runner | E2 / B6 self-labeling success signal |

## 8. Dependencies / build backlog (design only — not yet started)
- Gold-annotated paper set + fixed Neo4j KB snapshot (gates A1–A4).
- Rebuild E1 retrieval harness (query generation + scoring).
- Headless baseline runner (`main` pipeline is Streamlit-driven; GraphRAG has `run_pipeline()`).
- **Taxonomy-spanning** labeled valid/invalid topology (B4) & parameter (B5) sets — per-class error coverage.
- AR result caching layer + representative E2 subset for B5 (keeps external Bedrock calls cheap/reproducible).
- **User simulator** (LLM answers clarifications from a hidden gold spec) — shared component for the B1
  clarification ablation and any interactive end-to-end (E2) runs.
- Curated clear/ambiguous query set for B1 (ambiguous variants of Testbench prompts + labels).
- Curated gold-topology set for B3 (Level 1–4 + representative Testbench prompts, hand-drawn circuit graphs).

## 9. Decision log
- **2026-06-08:** Anchor = extrinsic-primary. KG-vs-baseline = central comparison. Layer structure approved.
  Walkthrough order A1→C; go deep on harder cells (A4, B4/B5, ground-truth strategy). A1 metric DECIDED
  (see table). DRC re-integrated into agentic pipeline (commit 8e819f2) — enables E2.
- **2026-06-08 (A2 DECIDED):** VSA = evidence-grounded edge faithfulness (primary, robust to novel-vs-wrong)
  + edge-type confusion matrix + confidence calibration (co-headline; validates 0.8/0.5 auto-commit). Recall
  on anchor only; arch-gate folded into E2 ablation. **Unified gate-eval template LOCKED (§6.6)** for all gates.
- **2026-06-08 (A3 DECIDED):** DIA Global Inference measured two-stage — candidate-retrieval recall@K (caps
  downstream) + semantic-verification gate (template) + end-to-end edge precision (judge); recall on anchor
  only. Manifest execution = correctness check (transaction integrity, idempotent) + L0; schema conformance
  → L1 graph-health. Decomposition isolates "failed to find" vs "failed to accept" the link.
- **2026-06-08 (A4 DECIDED):** SEA leads with **systems-dynamics** — schema convergence/stability (growth
  curve, churn, order-invariance), fully automatable/no-labels — inverting the usual accuracy-first priority.
  + threshold sensitivity sweep of {3, 0.85, 0.70} (the 0.70 confidence floor is principled only if VSA
  confidence is calibrated → A2 cross-link). + recategorization accuracy (L2) + sampled promotion-precision
  judgment + review-queue burden (L0). Convergence eval requires fixed paper set & ingestion order.
- **2026-06-08 (A5 DECIDED):** PDK ingestion = node P/R (+ports/params) + per-edge-type P/R (4 types) +
  topology-completion accuracy, vs hand-verified gold (cheap — PDK is enumerable). Embeddings validated via
  B2; versioning = correctness check. A5 correctness **bounds B2's ceiling**; serves as the ingestion anchor.
- **2026-06-08 (A6 DECIDED):** Neo4j backend = no standalone quality metric (correctness validated
  transitively via B2/A3). Integrity constraints → L1 graph-health; KG-query/GDS latency vs graph size → L0
  scalability result. No Neo4j-vs-ArangoDB comparison (ArangoDB deprecated). **Offline section A1–A6 complete.**
- **2026-06-08 (B1 DECIDED):** Interpreter = should-clarify P/R (2×2 on curated clear/ambiguous set, primary)
  + intent-extraction correctness vs gold DesignIntent + question usefulness/efficiency + E2 clarification
  on/off ablation. **User simulator** approved as a shared harness component (enables interactive E2).
- **2026-06-08 (B2 DECIDED = E1):** retrieval A/B with **functional/indirect queries** added to paraphrase
  queries (required for KG advantage to be observable), the **scaling hypothesis** (KG advantage vs PDK size)
  as the distinctive result, and a **compositional** regime (query→component set, set P/R) alongside
  single-component pass@k. Three arms (BM25+emb / LLM-context / KG), strict fairness controls.
- **2026-06-08 (B3 DECIDED):** Schematic builder — **validity ≠ correctness**: topology correctness
  (edge-F1/GED vs gold) on a curated subset (Level 1–4 + representative Testbench) + netlist validity +
  port-compat on the full suite; measured on both pipeline arms (baseline p300 vs schematic_builder_server).
- **2026-06-08 (B4 DECIDED):** Clingo topology gate evaluated as a **formal method** — rule adequacy over an
  error taxonomy (per-class catch rate + coverage-gap analysis), solver-correctness sanity, false-reject on
  valid controls, and ablation on the **full success funnel** (routing/sim/DRC). §6.6 extended for formal gates.
- **2026-06-08 (B5 DECIDED):** AR parameter gate reuses B4's formal-gate framing over a **parameter**-error
  taxonomy. **Operational caveat unique to B5:** AWS Bedrock AR is external/paid/networked → cache AR results,
  run on a representative subset, and report AR cost/latency/availability as an explicit L0 finding.
- **2026-06-08 (B6 DECIDED):** Layout & sim = **staged success funnel** (instantiate→routing_ok→models→
  sim-success→DRC-clean), self-labeling, complexity-stratified — the E2 headline; compare arms per stage.
  Explicit caveat: funnel = manufacturable & simulatable, **not functionally correct**; functional-spec
  verification deferred (tied to the deferred optimizer; needs per-prompt target responses).
- **2026-06-08 (C DECIDED):** Infra = **cost A/B** (agentic GraphRAG vs linear baseline: tokens/latency/
  rounds/cap-hits) — the extrinsic lift must justify this overhead — + review-queue burden (auto/queue/reject
  split, ties to calibration) + critic on/off ablation on E2. UI not benchmarked.
- **2026-06-08: PER-FEATURE WALKTHROUGH COMPLETE (A1→C, all DECIDED).** Remaining to lock: cross-cutting
  methodology §6 items 1–5 (isolation convention, ground-truth strategy, statistical rigor, ablation set,
  model control); §6.6 gate template already LOCKED.
- **2026-06-08: CROSS-CUTTING §6 LOCKED — BENCHMARK DESIGN COMPLETE.** Isolation = gold-input per stage +
  one end-to-end. Ground-truth = annotation on A1–A4 + B4/B5 taxonomies; A5/B2/B6 near-free. Stats = McNemar
  (pass rates) + bootstrap CIs (pass@k) + fixed temp/seed + per-tier breakdown. Ablations = KG-grounding,
  deep-inference, schema-evolution, each gate, critic, clarification. Model = single canonical (o1) fixed
  across arms; model-sensitivity = future work. Next phase = implementation (build backlog §8).

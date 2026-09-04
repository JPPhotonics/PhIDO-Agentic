# PhIDO GraphRAG Benchmark — Digest

> A one-look summary of every evaluation comparing the agentic, knowledge-graph-backed
> **GraphRAG** pipeline against the **baseline** PhIDO pipeline — what each test asks,
> what it found, and how settled the number is.

**Prepared:** 2026-06-30 · **Branch:** `graphRAG-implementation` · **Experiments:** 15
**Distilled from:** [`BENCHMARK_STATUS_REPORT.md`](BENCHMARK_STATUS_REPORT.md)

---

## The four things to know

| | Headline | What it means |
|---|---|---|
| ✅ | **Hybrid retrieval pass@3 = 0.80** | Lexical + embedding fusion beats every single method. Strongest positive in the suite. |
| ❌ | **End-to-end DRC-clean 0.25 vs 0.71** | GraphRAG loses to baseline at every stage, at 2.6× latency. The central negative result. |
| ✅ | **Topology-gate catch rate = 100%** | Catches all 15 seeded error classes with 0.00 false rejects. The formal layer is sound. |
| ◑ | **PDK node ingestion F1 = 1.00** | KG integrity clean (0 orphans/dupes). Open gap: expert-validated relationship gold. |

**Bottom line for the thesis.** The intrinsic and retrieval evidence supports the GraphRAG
approach. The extrinsic end-to-end claim does **not** hold yet — and the loss sits in the
LLM agent-orchestration layer, not in retrieval or the (provably deterministic) backend. The
priority is to **diagnose and fix that layer, not abandon the approach.**

---

## Status at a glance — all 15 experiments

| Exp | Name | Type | Status | N | Headline |
|-----|------|------|--------|---|----------|
| **E1** | Component retrieval A/B | Extrinsic | ✅ Complete | 18 / 231 | Hybrid wins (p@3 0.80); KG wins on design-intent |
| **E2** | End-to-end funnel (prompt→GDS) | Extrinsic | ✅ Complete | 24 × 3 | **GraphRAG 0.25 vs baseline 0.71 DRC-clean** |
| **A1** | Extraction / duplication | Intrinsic | ✅ Complete\* | 414 | Fragmentation quantified; gold F1 pending |
| **A2** | Faithfulness / DIA precision | Intrinsic | ✅ Complete | 40 / 1 ann. | DIA precision 0.675 → 0.84 projected post-fix |
| **A3** | Cross-document integration | Intrinsic | ✅ Complete\* | 17 docs | Multi-doc support quantified; recall pending gold |
| **A4** | Schema convergence (SEA) | Intrinsic | ✅ Complete | 116 / 15 | 60 types proposed, 1 promoted; churn quantified |
| **A5** | PDK ingestion vs DesignLibrary | Intrinsic | ✅ Complete | 34 | Node F1 = 1.00; function-edge F1 = 0.27 |
| **A6** | KG integrity + latency | Intrinsic/Ops | ✅ Complete | 2707 / 3333 | Integrity clean; review backlog 2114 (concern) |
| **B1** | Clarification (should-clarify) | Intrinsic | ⚠️ Underpowered | 6 × 3 | Recall 1.0, precision 0.5 (over-clarifies) |
| **Gate** | Topology-gate discrimination | Intrinsic | ✅ Complete | 8 + 15 | 100% catch, 0.00 false-reject |
| **Hard-E2** | Topology-gate repair value | Extrinsic | ✅ Complete | 11 × 1 | Repair value ≈ 0 on this set |
| **Variance** | KB-rebuild non-determinism | Ops | ✅ Complete | 5 builds | Edge CV 14.7%; temp=0 ≠ deterministic |
| **Ablation-k** | Gate/critic ablation | Extrinsic | ⚠️ Underpowered | 20 × 3 | Effects not significant (McNemar p≥0.375) |
| **Backend** | Layout/sim/DRC determinism | Ops | ✅ Complete | 20 | Fully deterministic; mean 2.32 s |
| **L1** | Reference-free KG health | Intrinsic | ✅ Complete | 474 | MRR 0.175 vs 0.066 corrupted control |

Legend: ✅ numbers present & interpretable · ⚠️ complete but underpowered / needs more labels
· **\*** = label-free proxy, gold pending.

---

## Extrinsic — the thesis claim
*Does the pipeline produce better designs end-to-end?*

### E1 — Component Retrieval A/B ✅ · *win*
- **Asks:** Does KG-grounded component selection beat lexical and LLM-over-JSON retrieval?
- **Finds:** Hybrid (RRF fusion) **dominates — pass@3 0.80**. KG-embed wins on design-intent
  queries (0.556 vs 0.333 pass@1); lexical wins on functional queries. The PERFORMS_FUNCTION
  gate is net-harmful; paper enrichment is retrieval-neutral.
- **Caveat:** Gold is DRAFT (pending Poon-group validation); functional stratum underpowered (N=25).
- **N:** 18 curated / 231 testbench.

### E2 — End-to-End Funnel ❌ · *regression*
- **Asks:** Prompt → instantiate → routing → models → simulation → DRC-clean: does agentic
  GraphRAG beat baseline?
- **Finds:** Baseline wins at every stage and tier: **DRC-clean 0.71 vs 0.25**. 17/24 baseline
  pass all stages vs 6/24 GraphRAG. Cost: **2.6×** latency, **4.3×** input tokens. GraphRAG is
  also unstable (0.38–0.67 non-unanimous repeats).
- **Caveat:** "Passing" = manufacturable & simulatable, not functionally correct; prompts needing
  unmodeled parts die legitimately at the models stage.
- **N:** 24 × 3 (majority vote).

### Hard-E2 — Topology-Gate Repair Value ✅ · *mixed*
- **Asks:** On hard prompts, does the formal topology gate improve end-to-end build outcomes?
- **Finds:** Repair value **≈ 0** on this buildable set — gate-on and gate-off both 1.00
  gold-match; the gate slightly hurts the downstream funnel.
- **Caveat:** Small, buildable-only subset — by construction has little left for repair to fix.
- **N:** 11 × 1.

### Ablation-k — Gate / Critic Ablation ⚠️ · *underpowered*
- **Asks:** How much do the topology gate and the critic each contribute to end-to-end success?
- **Finds:** Gate adds **+0.15** on models/sim but **p = 0.375** (not significant); critic
  contribution negligible. 35–65% of prompts flip stochastically per stage.
- **Caveat:** Underpowered — McNemar test cannot resolve the effect at this N.
- **N:** 20 × 3.

---

## KG-construction diagnostics — intrinsic
*Is the knowledge graph built correctly?*

### A1 — Extraction / Duplication ✅\* · *mixed*
- **Asks:** How much does node extraction over-fragment or duplicate the same real entity?
- **Finds:** Near-duplicate clusters quantified at cos ≥ 0.80 (26 Component, 17 Property…);
  semantic fragmentation flagged.
- **Caveat:** Label-free — gold F1 / P-R still pending hand annotation.
- **N:** 414 nodes.

### A2 — Faithfulness / DIA Precision ✅ · *win*
- **Asks:** Are extracted relationship triples actually supported by the source documents?
- **Finds:** DIA precision **0.675** (Wilson CI [0.52, 0.80]); typed edges **0.875**, generic
  RELATED_TO only **0.54**. Demoting RELATED_TO projects **0.84**; live KB confirms it down to
  3.8–5.4% of edges.
- **Caveat:** N=1 annotator / N=1 paper; recall unmeasured; post-fix precision projected, not
  re-annotated.
- **N:** 40 triples, 1 annotator.

### A3 — Cross-Document Integration ✅\* · *mixed*
- **Asks:** Does the KG fuse entities that appear across multiple papers rather than siloing them?
- **Finds:** Multi-document support quantified (Architecture 0.76, Component 0.59…) — entities do
  span documents.
- **Caveat:** Cross-doc edge recall needs gold link annotation.
- **N:** 17 docs / 367 entities.

### A4 — Schema Convergence (SEA) ✅ · *mixed*
- **Asks:** Does the self-evolving schema stabilize, or keep inventing one-off relationship types?
- **Finds:** **60** distinct types proposed, **1** promoted (CONTAINS_COMPONENT); 43/60 confined
  to a single document — mostly transient churn.
- **Caveat:** Growth curve is document-order-dependent; order-free resampling pending.
- **N:** 116 obs / 15 docs.

### A5 — PDK Ingestion vs DesignLibrary ✅ · *mixed*
- **Asks:** Does PDK ingestion faithfully reproduce the ground-truth DesignLibrary (nodes, ports,
  params, function edges)?
- **Finds:** Node P/R/**F1 = 1.00**; ports **0.97**, parameters **0.74**. But PERFORMS_FUNCTION
  **F1 = 0.27** (P 0.21 / R 0.40) — this bounds the E1 KG ceiling.
- **Caveat:** Function-name P/R conflates true error with vocabulary mismatch (no synonym map yet).
- **N:** 34 modules.

### A6 — KG Integrity + Latency ✅ · *mixed*
- **Asks:** Is the live graph structurally clean, and are queries fast?
- **Finds:** **0** orphans / dupes / self-loops / schema violations / missing embeddings; queries
  **1.4–2.1 ms** median.
- **Caveat:** Concerns — residual RELATED_TO 126 (3.8%) and a ReviewItem backlog of **2114**
  awaiting human review. Latency is one snapshot, not a scaling curve.
- **N:** 2707 nodes / 3333 edges.

### L1 — Reference-Free KG Health ✅ · *win*
- **Asks:** Without a gold reference, does the graph carry recoverable structural regularity?
- **Finds:** Link-prediction **MRR 0.175** on the clean graph vs **0.066** on a corrupted control
  (**+0.110** separation).
- **Caveat:** Measures internal consistency, not correctness against ground truth.
- **N:** 474 test triples.

### B1 — Clarification (should-clarify) ⚠️ · *underpowered*
- **Asks:** Does the pipeline ask for clarification exactly when a prompt is genuinely ambiguous?
- **Finds:** **Recall 1.0** (never misses ambiguity) but **precision 0.5** — it over-clarifies
  clear prompts.
- **Caveat:** N=6, single labeler — needs a rubric, a second annotator (κ), and a larger set.
- **N:** 6 × 3.

---

## Supporting & operational
*Do the numbers above stand on solid ground?*

### Gate — Topology-Gate Discrimination ✅ · *win*
- **Asks:** Does the formal topology gate catch invalid circuits while passing valid ones?
- **Finds:** **100%** catch across all 15 error classes, **0.00** false-reject — the gate is sound
  on its encoded rules.
- **Caveat:** Soundness is over the rules that are encoded; unencoded error types aren't tested here.
- **N:** 8 valid + 15 invalid.

### Backend — Layout / Sim / DRC Determinism ✅ · *win*
- **Asks:** Is the non-LLM backend deterministic, so E2 variance can be attributed upstream?
- **Finds:** **Fully deterministic** across 20 passes of a frozen netlist, mean **2.32 s** —
  confirms E2/ablation variance originates in LLM netlist generation, not the backend.
- **Caveat:** Holds for a single frozen netlist; not a stress test across diverse layouts.
- **N:** 20 passes.

### Variance — KB-Rebuild Non-Determinism ✅ · *concern*
- **Asks:** How much does the knowledge graph itself change between identical rebuilds?
- **Finds:** Edge-count **CV 14.7%**, literature-propagated functions **CV 27.4%**;
  **temperature = 0 does not** eliminate build non-determinism. Embedding-grounded retrieval stays
  byte-identical.
- **Caveat:** Implication — per-build KG retrieval numbers carry cross-run noise until this is tamed.
- **N:** 5 rebuilds.

---

## Open items & priorities

1. **Diagnose the E2 agentic regression.** The central negative result. Backend is clean and
   retrieval is fine, so the loss is in netlist generation / agent orchestration — the
   highest-value next investigation.
2. **Expert gold validation.** A1/A3 recall, A2 faithfulness (beyond N=1), and E1 testbench gold
   all depend on Poon-group annotation. Several "complete" intrinsic results are still label-free
   proxies.
3. **Drain the review backlog.** 2114 pending ReviewItems and the residual 126 RELATED_TO edges
   flagged in A6.
4. **Re-power the underpowered arms.** B1 (N=6), the E1 functional stratum (N=25), and the
   ablation McNemar (p≥0.375) need larger N and seeded multi-build runs.
5. **Tame KB build non-determinism.** Edge CV 14.7% must be controlled before per-build KG
   retrieval numbers can be treated as stable.

---

*Underlying data: `benchmark/results/`. Findings docs: `A2_FAITHFULNESS_FINDINGS.md`,
`E1_RETRIEVAL_FINDINGS.md`, `VARIANCE_FINDINGS.md`, `EXPERIMENTAL_SETUP.md`. Experiment taxonomy
per `EVALUATION_OVERVIEW.md` / `BENCHMARK_ARCHITECTURE.md`.*

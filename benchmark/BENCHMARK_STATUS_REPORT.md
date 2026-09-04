# PhIDO GraphRAG Benchmark — Status & Results Report

**Prepared:** 2026-06-30 · **Branch:** `graphRAG-implementation` (worktree `wt-graphrag`)
**Scope:** All evaluation experiments comparing the agentic, KG-backed GraphRAG pipeline against the baseline PhIDO pipeline, plus the intrinsic KG-construction and operational diagnostics that explain those results.

---

## 1. Executive Summary

The benchmark suite is **substantively complete**: every planned experiment has produced numbers, and the supporting determinism/variance audits needed to interpret them are done. The headline picture is honest and consequential:

- **Retrieval works.** Component selection improves with KG grounding for *design-intent* queries, and a **hybrid (lexical + embedding) retriever dominates everything** on the broad testbench (pass@3 = 0.80). This is the strongest positive result in the suite.
- **End-to-end agentic integration currently loses.** On the head-to-head end-to-end funnel (E2), the agentic GraphRAG pipeline reaches **0.25 DRC-clean vs the baseline's 0.71** — a decisive regression at every stage, at 2.6× the latency and 4.3× the input tokens.
- **The bottleneck is the agent control flow, not retrieval.** The backend (layout/sim/DRC) is provably deterministic; the formal topology gate catches 100% of seeded error classes with zero false rejects. The losses concentrate in the LLM-driven netlist-generation / agent-orchestration layer, which also shows high run-to-run variance (38–67% unstable repeats).
- **KG construction is structurally sound but partially gold-blind.** Integrity checks are clean (0 orphans/dupes/self-loops), PDK node ingestion is perfect (F1 = 1.00), and the RELATED_TO-precision fix landed. The chief open item is **expert-validated gold** — current precision/recall on relationships rests on N=1 annotator / N=1 paper, and several A-series metrics are label-free proxies pending annotation.

**Bottom line for the thesis:** the intrinsic and retrieval evidence supports the GraphRAG approach; the extrinsic end-to-end claim does **not** hold yet and is the priority to diagnose and fix (agent orchestration), not abandon.

---

## 2. Status at a Glance

| Exp | Name | Type | Status | N | Headline |
|-----|------|------|--------|---|----------|
| **E1** | Component retrieval A/B | Extrinsic | ✅ Complete | 18 curated / 231 testbench | Hybrid wins (p@3 0.80); KG wins on design-intent |
| **E2** | End-to-end funnel (prompt→GDS) | Extrinsic | ✅ Complete | 24 × 3 | **GraphRAG 0.25 vs baseline 0.71 DRC-clean** |
| **A1** | Extraction / duplication | Intrinsic | ✅ Complete (label-free) | 414 nodes | Fragmentation quantified; gold F1 pending |
| **A2** | Faithfulness / DIA precision | Intrinsic | ✅ Complete | 40 triples, 1 annotator | DIA precision 0.675 → 0.84 projected post-fix |
| **A3** | Cross-document integration | Intrinsic | ✅ Complete (label-free) | 17 docs, 367 entities | Multi-doc support quantified; recall pending gold |
| **A4** | Schema convergence (SEA) | Intrinsic | ✅ Complete | 116 obs / 15 docs | 60 proposed types, 1 promoted; churn quantified |
| **A5** | PDK ingestion vs DesignLibrary | Intrinsic | ✅ Complete | 34 modules | Node F1 = 1.00; function-edge F1 = 0.27 |
| **A6** | KG integrity + latency | Intrinsic/Ops | ✅ Complete | 2707 nodes / 3333 edges | Integrity clean; review backlog 2114 (concern) |
| **B1** | Clarification (should-clarify) | Intrinsic | ⚠️ Complete, underpowered | 6 × 3 | Recall 1.0, precision 0.5 (over-clarifies) |
| **Gate** | Topology-gate discrimination | Intrinsic | ✅ Complete | 8 valid + 15 invalid | 100% catch, 0.00 false-reject |
| **Hard-E2** | Topology-gate repair value | Extrinsic | ✅ Complete | 11 × 1 | Repair value ≈ 0 on this set |
| **Variance** | KB-rebuild non-determinism | Ops | ✅ Complete | 5 builds | Edge CV 14.7%; temp=0 ≠ deterministic |
| **Ablation-k** | Gate/critic ablation | Extrinsic | ✅ Complete | 20 × 3 | Effects not significant (McNemar p≥0.375) |
| **Backend** | Layout/sim/DRC determinism | Ops | ✅ Complete | 20 passes | Fully deterministic; mean 2.32 s |
| **L1** | Reference-free KG health | Intrinsic | ✅ Complete | 474 test triples | MRR 0.175 vs 0.066 corrupted control |

Legend: ✅ numbers present & interpretable · ⚠️ complete but underpowered / needs more labels.

---

## 3. Headline Experiments (Extrinsic — the thesis claim)

### E1 — Component Retrieval A/B  ✅
*Does KG-grounded selection beat lexical and LLM-over-JSON retrieval?*

| Metric | Lexical | KG (functional) | KG-embed | Hybrid (prod) |
|--------|--------:|----------------:|---------:|--------------:|
| Curated-18 pass@1 | 0.333 | 0.444 | **0.556** | — |
| Curated-18 coverage@3 | 0.444 | 0.509 | **0.676** | — |
| Testbench pass@1 / p@3 / MRR | 0.411 / 0.545 / 0.526 | 0.329 / 0.398 / 0.420 | 0.385 / 0.519 / 0.513 | **0.437 / 0.797 / 0.611** |
| Functional-query pass@1 (N=25) | **0.320** | 0.160 | 0.080 | 0.200 |

**Conclusions:**
- **KG wins on design-intent** queries (kg-embed 0.556 vs lexical 0.333 pass@1); **ties** on named-component queries; **lexical wins** on functional queries.
- **Hybrid (RRF fusion) dominates** all single methods — complementary failure modes.
- **Design finding:** the PERFORMS_FUNCTION gate is *net-harmful* (popular generic functions over-match); embedding-only ranking is strictly better (+0.125 pass@1).
- **Paper enrichment is retrieval-neutral** (Δ ≈ 0), robust across rebuilds — corpus/query domain mismatch.

**Caveats:** gold is DRAFT (pending Poon-group validation); functional stratum underpowered (N=25); KB non-determinism adds cross-run noise on the KG arm (kg-embed is byte-stable).

### E2 — End-to-End Funnel (Agentic GraphRAG vs Baseline)  ✅
*Prompt → instantiate → routing → models → simulation → DRC-clean. N=24, K=3, majority vote.*

| Stage | Baseline | GraphRAG | Δ |
|-------|---------:|---------:|---:|
| Instantiate | 0.96 | 0.75 | −0.21 |
| Routing OK | 0.96 | 0.58 | −0.38 |
| Models | 0.88 | 0.29 | −0.58 |
| Sim success | 0.88 | 0.29 | −0.58 |
| **DRC-clean (end-to-end)** | **0.71** | **0.25** | **−0.46** |

- **Baseline beats agentic at every stage and every complexity tier** (L1–L4); 17/24 baseline pass all stages vs 6/24 for GraphRAG.
- **Cost:** GraphRAG 126.4 s / 148k in / 30.7k out vs baseline 48.5 s / 34.6k in / 10.8k out (2.6× / 4.3× / 2.8×).
- **Instability:** GraphRAG non-unanimous-repeat rate 0.38–0.67 vs baseline 0.12–0.17.

**Caveat:** "passing" = manufacturable & simulatable, **not** functionally correct; prompts needing unmodeled parts legitimately die at the `models` stage.

---

## 4. KG-Construction Diagnostics (Intrinsic)

- **A1 — Extraction/duplication** (414 nodes): near-duplicate clusters quantified at cos≥0.80 (26 Component, 17 Property, …); semantic fragmentation flagged. *Gold F1/P-R still pending hand annotation — current signal is label-free.*
- **A2 — Faithfulness/DIA** (40 triples, 1 annotator, paper #01): **DIA precision 0.675** (Wilson CI [0.52, 0.80]); typed edges 0.875, generic RELATED_TO only 0.54. **Post-fix projection 0.84** after demoting generic RELATED_TO; live KB confirms RELATED_TO down to 3.8–5.4% of edges. *N=1 annotator / N=1 paper; recall unmeasured; post-fix precision projected, not re-annotated.*
- **A3 — Cross-document integration** (17 docs, 367 entities): multi-document support quantified (Architecture 0.76, Component 0.59 …). *Cross-doc edge recall needs gold links.*
- **A4 — Schema convergence (SEA)** (116 obs / 15 docs): 60 distinct proposed types, **1 promoted** (CONTAINS_COMPONENT); 43/60 types confined to a single document (transient churn). *Growth curve is document-order-dependent; order-free resampling pending.*
- **A5 — PDK ingestion** (34 modules): **node P/R/F1 = 1.00**; ports 0.97, parameters 0.74; **PERFORMS_FUNCTION F1 = 0.27** (P 0.21 / R 0.40), IMPLEMENTS presence 0.50. *Function-name P/R conflates true error with vocabulary mismatch (no synonym map) — this bounds the E1 KG ceiling.*
- **A6 — Integrity + latency** (2707 nodes / 3333 edges): **0 orphans / 0 dupes / 0 self-loops / 0 schema violations / 0 missing embeddings**; queries 1.4–2.1 ms median. **Concerns:** RELATED_TO residual 126 (3.8%) and **ReviewItem backlog 2114** awaiting human review. *Latency is a single snapshot, not a scaling curve.*

---

## 5. Supporting & Operational Studies

- **Topology-gate discrimination** (8 valid + 15 invalid): **100% catch across all 15 error classes, 0.00 false-reject** — the formal gate is sound on encoded rules.
- **Hard-E2 (gate repair value)** (N=11): repair value **≈ 0** on this buildable set (gate-on and gate-off both 1.00 gold-match); gate slightly hurts downstream funnel. *Small, buildable-only subset.*
- **Ablation sweep over k** (20 × 3): topology gate +0.15 on models/sim (**p=0.375, not significant**); critic contribution negligible; 35–65% of prompts flip stochastically per stage. *Underpowered.*
- **Backend determinism** (20 passes, frozen netlist): **fully deterministic**, mean 2.32 s — confirms the E2/ablation variance originates upstream in LLM netlist generation, not the backend.
- **Variance campaign** (5 KB rebuilds): edge-count **CV 14.7%**, literature-propagated functions CV 27.4%; **temperature=0 does not eliminate** build non-determinism; embedding-grounded retrieval stays byte-identical.
- **L1 reference-free health** (474 test triples): clean-graph link-prediction **MRR 0.175 vs 0.066 corrupted control** (+0.110 separation) — recoverable structural regularity. *Measures consistency, not correctness.*
- **B1 clarification** (6 × 3): **recall 1.0, precision 0.5** — never misses ambiguity but over-clarifies clear prompts. *N=6, single labeler — needs a rubric, a second annotator (κ), and a larger set.*

---

## 6. Open Items & Priorities

1. **Diagnose the E2 agentic regression** — the central negative result. Backend is clean and retrieval is fine, so the loss is in netlist-generation / agent orchestration. This is the highest-value next investigation.
2. **Expert gold validation** — A1/A3 F1-recall, A2 faithfulness (beyond N=1), and E1 testbench gold all depend on Poon-group annotation. Several "complete" intrinsic results are currently label-free proxies.
3. **Drain the review backlog (2114 ReviewItems)** and clear residual RELATED_TO (126) flagged in A6.
4. **Re-power the underpowered arms** — B1 (N=6), E1 functional stratum (N=25), ablation McNemar (p≥0.375) — with larger N and seeded multi-build runs.
5. **Tame KB build non-determinism** (CV 14.7%) before treating per-build KG retrieval numbers as stable.

---

*Source data: `benchmark/results/` and the findings docs `A2_FAITHFULNESS_FINDINGS.md`, `E1_RETRIEVAL_FINDINGS.md`, `VARIANCE_FINDINGS.md`, `EXPERIMENTAL_SETUP.md`. Experiment taxonomy per `EVALUATION_OVERVIEW.md` / `BENCHMARK_ARCHITECTURE.md`.*

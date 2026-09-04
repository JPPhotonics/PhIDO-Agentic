# Knowledge-Graph Evaluation — Literature Review

> Status: **Reference doc for the thesis benchmark.** Compiled 2026-06-10 via a
> fact-checked deep-research pass (27 sources fetched, 132 candidate claims, 25
> adversarially verified 3-vote, 0 killed). Companion to `BENCHMARK_ARCHITECTURE.md`
> — §6 below maps every finding onto that benchmark's L0–L3 layers.
>
> Scope: how domain-specific KGs (biomedical, materials science, scholarly/engineering,
> hardware) are evaluated, across four dimensions, with balanced treatment of the
> gold-standard feasibility problem.

---

## 0. The one-paragraph takeaway

The literature evaluates domain-specific KGs along four interlocking dimensions —
**intrinsic quality**, **extrinsic/task-based**, **construction-pipeline**, and **human/
LLM-as-judge** — and it broadly *agrees with the premise that exhaustive gold annotation
is infeasible* for specialized domains. The defensible response is not "annotate
everything" but **trade exhaustiveness for statistical confidence and automation**: a
small, statistically *designed* human-audited sample for precision (with confidence
intervals), distant supervision from a curated reference for recall, reference-free
proxies for scale, and LLM-as-judge for the long tail beyond any reference. This is
exactly the posture `BENCHMARK_ARCHITECTURE.md` already takes (L1 reference-free at
scale, L2 gold only as a small anchor, L3 extrinsic as the headline).

---

## 1. Intrinsic KG quality

**Canonical metric set.** Precision / recall / F-measure against a *manually labeled
subset* — a **partial gold standard**, not exhaustive. For error-detection / correction
tasks the metrics shift to **accuracy and AUC**. The human-sourced partial gold standard
is high quality but **costly, so it is deliberately kept small**.
— Cimiano & Paulheim, *Knowledge Graph Refinement: A Survey* (Semantic Web Journal, 2017).
[SW-160218](https://journals.sagepub.com/doi/10.3233/SW-160218)

**The rigorous reframing (most directly useful).** KG accuracy = *statistically estimating
the proportion of correct triples from a human-annotated sample*. The core tension is
**statistical confidence vs. annotation cost**. Better sampling designs (cluster →
weighted → two-stage → stratified) cut annotation cost by **up to 60% (static KGs) / 80%
(evolving KGs)** while preserving the same confidence-interval guarantees.
— Gao et al., *Efficient Knowledge Graph Accuracy Evaluation* (PVLDB 2019).
[arXiv:1907.09657](https://arxiv.org/abs/1907.09657) ·
[PVLDB PDF](http://www.vldb.org/pvldb/vol12/p1679-gao.pdf)

> **Implication for the thesis:** you never annotate the whole KG. You annotate a
> *statistically designed sample* and report accuracy ± CI. This is the principled
> backbone for the A1–A4 "gold sub-graph" in the benchmark — and it directly answers the
> "gold-annotated set is infeasible" worry: the right unit is a *sample with error bars*,
> not a complete ground truth.

---

## 2. Extrinsic / task-based evaluation

**Task-dependence is the headline caution.** A KG's measured *value* depends on which
downstream task you pick: **RAG wins single-hop/detail queries; GraphRAG wins multi-hop**
— complementary, no consistent winner. Choosing only single-hop QA would systematically
undersell a graph-structured KG.
— Han et al., *RAG vs. GraphRAG: A Systematic Evaluation* (2025).
[arXiv:2502.11371](https://arxiv.org/html/2502.11371v3)

**Answer-entity coverage** (the cheap, powerful extrinsic completeness proxy). Measure the
fraction of downstream *answer entities* present in the constructed KG — only **~65.8%
(HotpotQA) / 65.5% (NQ)** in the study above — which directly explains downstream
underperformance. (See §6.2 for how this operationalizes for PhIDO.)

**Holistic pipeline scoring.** GraphRAG-Bench evaluates construction + retrieval +
generation *jointly*, and adds a **reasoning-coherence track** (does a correct answer
reflect correct reasoning, or a lucky guess?) separate from final-answer accuracy.
— *GraphRAG-Bench* (ICLR'26). [arXiv:2506.02404](https://arxiv.org/pdf/2506.02404)

**⚠️ Link-prediction benchmark hygiene.** FB15k/WN18 are inflated by reverse/duplicate
relations (test leakage) and Cartesian-product relations solvable by trivial rules,
producing **19–175% accuracy overestimation**; this motivated FB15k-237 / WN18RR. If you
use link prediction extrinsically, de-leak first.
— Akrami et al., *Realistic Re-evaluation of KG Completion Methods* (SIGMOD 2020).
[arXiv:2003.08001](https://arxiv.org/pdf/2003.08001)
*(The "much less accurate after cleaning" magnitude passed only 2-1 in verification —
treat the direction as solid, the exact magnitude as the soft point.)*

---

## 3. Construction-pipeline evaluation

**Extraction P/R.** Entity and relation extraction scored against a reference, standard
NLP-style — the A1/A5 region of the benchmark.

**Ontology / schema alignment has a mature standardized benchmark: OAEI** (Ontology
Alignment Evaluation Initiative). Yearly controlled evaluations since 2004; auto-scores
P/R/F against per-track reference alignments. Crucially it has **technical/scientific
tracks**: biomedical (Anatomy: Mouse Anatomy vs NCI Thesaurus; Bio-ML over Mondo/UMLS),
**Materials Science (MSE)**, pharmacogenomics, biodiversity, and KG matching
(DBpedia/NELL). Closest standardized precedent for schema-conformance evaluation in an
engineering domain.
— [oaei.ontologymatching.org](https://oaei.ontologymatching.org/)

---

## 4. Human & LLM-as-judge protocols

**Retrospective sampling-based audits** are *the* standard human protocol: methods emit
tens of thousands of candidate axioms, so judges label only a **sample** — feasibility
forces sampling (Cimiano & Paulheim, above).

**LLM-as-judge** increasingly scores relations *outside* the reference set (see §5,
WikiCausal), but carries documented hazards — most notably **position bias**: reversing
the order of two presented items can flip the verdict. Mitigation is order-swapping with
conservative aggregation, but it only partially fixes the problem.
— Position bias replicated in *RAG vs. GraphRAG* (above) and *Judging the Judges*
([arXiv:2411.15594](https://arxiv.org/pdf/2411.15594)).

**Inter-annotator agreement (IAA)** — quantitative reliability of human annotation. See §4a.

---

## 4a. Inter-annotator agreement in technical/scientific domains

> From a focused, source-verified research pass. Numbers read from primary source text
> unless flagged *(unverified — secondary)*; those few should be checked against the
> original PDF before citing verbatim in the thesis.

**Reported IAA values** (the entity > relation gap is the consistent pattern):

| Domain | Task | Metric | Value | Source |
|---|---|---|---|---|
| Scientific IE (SciER, 2024) | Entity (in-domain) | κ | **94.2%** | [arXiv:2410.21155](https://arxiv.org/html/2410.21155v1) |
| SciER | Relation (in-domain) | κ | **70.8%** | same |
| SciER | Entity / Relation (out-of-domain) | κ | **74.1% / 73.8%** | same |
| SciERC (Luan 2018) | Entity / Relation / Coref | κ | **76.9 / 67.8 / 63.8** *(unverified — secondary)* | [D18-1360](https://aclanthology.org/D18-1360/) |
| Materials (MuLMS, 2023) | Entity (NER, relaxed) | pairwise F1 | **77.9** (range 40–100) | [arXiv:2310.15569](https://arxiv.org/html/2310.15569) |
| MuLMS | Relation (avg) | Cohen's κ | **0.61** (range 0.00–0.88 by type) | same |
| Polymers (PolyIE, 2023) | Entity / Relation (avg) | pairwise F1 | **0.89 / 0.84** | [arXiv:2311.07715](https://arxiv.org/pdf/2311.07715) |
| Polycrystal synth (PcMSP, 2022) | Entity, round 1 → round 2 | Fleiss' κ | **56.4 → 69.8** | [EMNLP-F 2022.446](https://aclanthology.org/2022.findings-emnlp.446/) |
| PcMSP | Relation, round 1 → round 2 | Fleiss' κ | **48.5 → 53.6** | same |
| Fuel cells (SOFC-Exp, 2020) | Experiment-sentence detection | Cohen's κ | **0.75** | [arXiv:2006.03039](https://arxiv.org/abs/2006.03039) |
| Materials (MaterioMiner, 2024) | Entity (NER) | Fleiss' κ | **0.733** | [arXiv:2408.04661](https://arxiv.org/pdf/2408.04661) |
| Biomedical concepts (CRAFT, 2012) | Concept/entity | trained-to | annotators trained to ~**80%** | [BMC Bioinf 13:161](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-161) |
| Biomedical RE — **expert** | Relation | Fleiss' κ / α | **0.65 / 0.66** (substantial) | [PMC7706181](https://pmc.ncbi.nlm.nih.gov/articles/PMC7706181/) |
| Biomedical RE — **crowd (MTurk)** | Relation | Fleiss' κ / α | **0.20 / 0.20** (slight–fair) | same |

**Landis & Koch (1977) interpretation scale** (confirmed, [NBK92287](https://www.ncbi.nlm.nih.gov/books/NBK92287/table/executivesummary.t2/)):
<0 poor · 0.00–0.20 slight · 0.21–0.40 fair · 0.41–0.60 moderate · **0.61–0.80 substantial** · 0.81–1.00 almost perfect. *(The bands are admittedly arbitrary; interpret relative to task difficulty and category count.)*

**Why relation IAA systematically lags entity IAA** (every multi-task corpus above shows it):
- **Combinatorial dependency** — a relation is correct only if *both* argument spans are correct, so entity-boundary disagreement propagates into and compounds relation disagreement.
- **Relation-type confusability** — semantically adjacent labels are hard (MuLMS `usedTogether` κ=0.20, `conditionEnvironment` κ=0.00) vs. concrete ones (`propertyValue` κ=0.88).
- **Implicit / long-distance relations** — cross-sentence and inferred links need deeper domain reasoning than spotting a named mention.
- **Domain expertise becomes load-bearing** — biomedical RE: non-experts collapse to κ≈0.20, experts hold κ≈0.65.

**Protocols that demonstrably raise IAA** (with before/after evidence):
- **Iterative guideline refinement + adjudication rounds** — PcMSP's two-round protocol (warm-up + discussion) raised agreement on *every* layer round-1→round-2 (entity 56.4→69.8, relation 48.5→53.6). Cleanest in-domain before/after.
- **Train annotators to a target before production** — CRAFT trained to ~80% on prior material; single shared guideline + minimizing subjective span choices.
- **Lead-annotator + ≥2 independent annotators + tooling + pre-training** with guidelines in an appendix (SciER).

**Crowd vs. expert** — for *technical relation* annotation, naive crowdsourcing is unreliable: experts reached κ≈0.65 while raw MTurk reached κ≈0.20, and adding crowd to the rater didn't help ([PMC7706181](https://pmc.ncbi.nlm.nih.gov/articles/PMC7706181/)); on-site rater cost ~2× crowd per item. Counterpoint: CrowdTruth (Dumitrache et al., ACM TiiS 2017, [10.1145/3152889](https://dl.acm.org/doi/10.1145/3152889)) shows that *modeling disagreement as signal* (heavy redundancy + ambiguity-aware aggregation, not naive majority vote) can approach expert quality at lower cost. Net: experts or expert-validated hybrid pipelines are the defensible choice for relations.

**"Acceptable to publish" threshold** — **α/κ ≥ 0.80** broadly reliable; **0.667 ≤ α < 0.80** acceptable for *tentative* conclusions (Krippendorff's cutoff); **κ ≥ 0.67** is the commonly cited NLP-corpus floor. In Landis-Koch terms, **"substantial" (0.61–0.80)** is the practical bar for hard IE tasks, with "almost perfect" expected for objective entity-level annotation.

### Recommendation for PhIDO (photonics/PIC — a first-of-its-kind corpus)
A defensible thesis target is **substantial agreement: κ (or pairwise-F1) ≥ 0.70–0.80 on
entities/components, ≥ 0.60 on relations/netlist connections** — reported *separately* for
the two tasks and *per relation type*, since relation IAA will lag (materials-science
analogs are the closest precedent: entities ~0.74–0.94, relations ~0.55–0.71). Report the
**PcMSP/SciER protocol**: a written guideline document → a warm-up/training round on shared
documents → ≥2 independent **domain-expert** annotators (relations need PIC knowledge) +
a lead adjudicator → IAA measured across **≥2 rounds to show before/after improvement** →
disagreement resolved by discussion. Convention: since κ is technically inapplicable to
span detection, report **average pairwise F1 for entity spans** and **Cohen's/Fleiss' κ
for the categorical relation-type decisions**, anchored explicitly to Landis-Koch and the
κ ≥ 0.67 NLP-corpus norm. *This directly sizes the annotation effort for the benchmark's
A1–A4 gold sub-graph and B4/B5 taxonomy sets.*

---

## 5. The gold-standard problem — ranked toolkit

The central practical concern: exhaustive gold annotation is infeasible for specialized
technical domains. Ranked by defensibility when you *cannot* build complete ground truth:

| Approach | What it gives you | Cost | Key limitation |
|---|---|---|---|
| **Sampling-based estimation** (Gao et al.) | Accuracy with confidence intervals from a small sample | Low–med (60–80% savings) | Estimates **precision** well; recall needs a reference of true facts |
| **Distant supervision from a curated KG** (WikiCausal) | Recall vs. pre-existing relations, no fresh annotation | Low | **Requires an existing curated reference KG** |
| **LLM-as-judge for novel relations** (WikiCausal hybrid) | Precision of relations *outside* the reference | Very low | Position/verbosity bias; needs calibration |
| **Reference-free proxy — LP-Measure** | Quality score with *no* gold standard or human labour | Near-zero | Formally measures only consistency/redundancy |
| **Silver standard** (KG-as-its-own-test) | Cheap completion evaluation | Near-zero | **Assumes the KG is already correct** → cannot detect errors |

**The most defensible pattern is a hybrid**, exemplified by **WikiCausal** (Hassanzadeh,
IBM, ISWC 2024, [arXiv:2409.00331](https://arxiv.org/pdf/2409.00331)): **distant
supervision from a curated KG for recall** + **LLM-as-judge for precision of novel
relations** ("Could {cause} result in {effect}?", majority vote) — *explicitly to avoid
manual/crowd annotation*. Pair with **LP-Measure** (Cao et al., NLPIR 2023,
[ACM](https://dl.acm.org/doi/fullHtml/10.1145/3639233.3639357)) — remove a small fraction
of triples, measure recovery via MRR/Hit@k — as a fully automated reference-free sanity
check. Empirically separates good from (synthetically corrupted) bad KGs, though it
formally captures only consistency/redundancy.

### ⚠️ The catch for photonics/hardware
Every *recall-side* shortcut above (distant supervision, silver standard, OAEI alignment)
**presupposes an existing curated reference KG** (Wikidata, UMLS, Mondo). **Photonics/PIC
design has no such reference KG.** So they only partially transfer. The honest fallback:
1. **Expert-built seed/partial gold**, evaluated with **sampling-based precision
   estimation** (Gao) → precision ± CI without exhaustive truth.
2. **LP-Measure-style reference-free consistency** → automated health proxy at scale
   (the benchmark's L1).
3. **Extrinsic answer-entity coverage** against PhIDO's *own* downstream design task →
   completeness signal that sidesteps the missing-reference-KG problem entirely, because
   you control the task. **This is likely the strongest move** (see §6.2).

---

## 6. Mapping to the PhIDO benchmark

### 6.1 The four dimensions ↔ the L0–L3 layers
| Literature dimension | `BENCHMARK_ARCHITECTURE.md` layer | Notes |
|---|---|---|
| Intrinsic, reference-based (P/R/F vs gold) | **L2** | A1–A4 gold sub-graph; use Gao sampling, report ± CI |
| Intrinsic, reference-free (consistency, faithfulness) | **L1** | evidence-grounded edge faithfulness ≈ LP-Measure's spirit; precision-like, no recall truth needed |
| Extrinsic / task-based | **L3** | E1 retrieval A/B, E2 end-to-end funnel — the headline |
| Construction-pipeline (extraction P/R, alignment) | **L2 (A1, A5)** | A5 cheap because PDK is enumerable; OAEI is the alignment precedent |
| Human / LLM-as-judge | crosses **L2/L1** | retrospective sampling audits; LLM-judge faithfulness with order-swap to fight position bias |

The benchmark's stance — *"KG is instrumental; extrinsic is the headline, intrinsic are
diagnostics"* — is exactly what the task-dependence and answer-entity-coverage literature
supports: a KG with excellent entity-F1 that doesn't move the design task is a failure.

### 6.2 Operationalizing **answer-entity coverage** for PhIDO

The open-domain metric (fraction of downstream answer entities present in the KG; 65.8% in
HotpotQA) **splits into two layers for PhIDO**, and the split is the key insight:

**(i) Presence coverage** — *is the required component a node in the KG at all?*
- "Answer entities" for a design prompt = the **set of components a correct design uses**.
  Derive the gold set from the `GETTING_STARTED_EXAMPLE_OUTPUTS/Level 1–4` reference
  outputs and (for E1) the source component per query.
- `presence_coverage = |gold components ∩ KG nodes| / |gold components|`.
- **For PhIDO this should be ≈100%**, because the KG ingests the *enumerable* PDK (A5) —
  unlike WikiCausal/HotpotQA where the KG is extracted from free text and misses entities.
  So presence coverage is a near-trivial **A5 sanity check**, not the interesting number.
  *(Caveat: ≈100% only if the gold prompts draw exclusively from PDK components. If a
  prompt implies a component not in the ~26 selectable DesignLibrary modules / 262-component
  GDSFactory set, presence coverage exposes a real library gap — worth reporting.)*

**(ii) Functional / retrievable coverage** — *given a functional query ("a component that
does X"), does the KG's structure surface the right node?*
- `retrievable_coverage@k = |gold components returned in top-k by KG retrieval| / |gold|`.
- **This is where the real gap lives for PhIDO**: the challenge is not presence (PDK is
  enumerable) but whether the `PERFORMS_FUNCTION` / `BASED_ON_PRINCIPLE` edges connect the
  query *intent* to the node. This **is** E1/B2 — so answer-entity coverage reframes E1
  `pass@k` as a coverage/completeness ceiling, and explains E2 failures upstream of DRC
  (a design can't succeed if a required component is never retrieved).

**Why this matters for the thesis:** it converts the literature's "construction
completeness" diagnostic into a metric PhIDO can compute *self-labeled* (no manual
annotation), and it cleanly separates the trivial part (presence, ≈100%, A5) from the
load-bearing part (functional retrievability, the KG advantage E1/B2 is designed to show).
It also gives a principled *ceiling* statement: end-to-end E2 success ≤ retrievable
coverage of the required component set.

### 6.3 Recommended evaluation posture (defensible without exhaustive gold)
1. **L3 extrinsic** (E1/E2) is the headline — self-labeling, no manual gold. Report
   `pass@k`, set-P/R, the staged funnel; bootstrap CIs + McNemar (already in §6.3 of the
   benchmark).
2. **L2 precision via Gao-style sampling** on the A1–A4 anchor — precision ± CI, *not* a
   complete gold KG. Recall reported **only on the small anchor** (the benchmark already
   says this).
3. **L1 reference-free** — evidence-grounded faithfulness (precision-like, robust to
   "novel vs. wrong") + graph-health (dup/contradiction/orphan/schema) at scale.
4. **LLM-as-judge** for faithfulness and novel-relation precision — *always* with
   order-swapping and conservative aggregation to counter position bias; calibrate against
   the small human anchor where possible.

---

## 7. Caveats (from verification)
- Findings rest mostly on single (primary, peer-reviewed) sources each; OAEI, the sampling
  estimators, and the link-prediction-leakage findings have the strongest independent
  corroboration.
- GraphRAG numbers (coverage %, single-vs-multi-hop win patterns) are fast-moving and
  dataset-specific — treat the *qualitative* task-dependence as durable, not the figures.
- LP-Measure's good-vs-bad separation was shown on *synthetically* corrupted KGs;
  generalization to naturally occurring errors is somewhat optimistic, and it formally
  measures only consistency/redundancy.
- LLM-as-judge position bias is well-replicated; order-swapping only *partially* fixes it.
- No surviving claim quantified typical IAA (kappa/alpha) for technical-domain KG
  annotation — addressed by the focused pass in §4a.

## 8. Open questions carried forward
- Achievable IAA (kappa, Krippendorff's α) for entity/relation annotation in
  photonics/hardware, and crowd-vs-expert cost/quality tradeoff. *(→ §4a)*
- Do cheap automated proxies (LP-Measure, LLM-judge precision) *correlate* with rigorous
  sampling-based human accuracy on the same KG? (Calibrate the cheap metrics against the
  small human anchor.)
- For a domain with no curated reference KG, the most defensible recall estimator when no
  partial reference of true relations exists (pooled judgments? expert-built seed only?).
- Robustness of GraphRAG-Bench reasoning-coherence scoring (itself LLM-as-judge) when
  transferred from open-domain QA to engineering/design QA.

## 9. Source list
**Primary:**
- Cimiano & Paulheim, *KG Refinement: A Survey*, SWJ 2017 — https://journals.sagepub.com/doi/10.3233/SW-160218
- Gao et al., *Efficient KG Accuracy Evaluation*, PVLDB 2019 — https://arxiv.org/abs/1907.09657 · http://www.vldb.org/pvldb/vol12/p1679-gao.pdf
- Hassanzadeh, *WikiCausal*, ISWC 2024 — https://arxiv.org/pdf/2409.00331
- Cao et al., *LP-Measure (Assessing KG Quality via Link Prediction)*, NLPIR 2023 — https://dl.acm.org/doi/fullHtml/10.1145/3639233.3639357
- Han et al., *RAG vs. GraphRAG: A Systematic Evaluation*, 2025 — https://arxiv.org/html/2502.11371v3
- *GraphRAG-Bench*, ICLR'26 — https://arxiv.org/pdf/2506.02404
- Akrami et al., *Realistic Re-evaluation of KG Completion*, SIGMOD 2020 — https://arxiv.org/pdf/2003.08001
- OAEI — https://oaei.ontologymatching.org/
- Marchesin et al., PVLDB vol.17 — https://www.vldb.org/pvldb/vol17/p2392-marchesin.pdf
- *Judging the Judges* (LLM-judge position bias) — https://arxiv.org/pdf/2411.15594

**Secondary / supporting:**
- Adaptive sampling (UAI 2018) — https://www.cs.purdue.edu/homes/yexiang/publications/uai-18-adaptive-final.pdf
- IAA reference (kappa/alpha) — https://mbrenndoerfer.com/writing/inter-annotator-agreement-kappa-alpha-reliability
- LLM evaluators (eugeneyan) — https://eugeneyan.com/writing/llm-evaluators/
- Biomedical/materials KG case studies — https://www.sciencedirect.com/science/article/pii/S2001037024003386 · https://link.springer.com/article/10.1186/s13040-022-00311-z

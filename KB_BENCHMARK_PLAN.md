# KB Benchmark — Execution Plan & Change Record

> Status: **Plan locked, implementation not started (2026-06-11).** This is the execution-level
> companion to `BENCHMARK_ARCHITECTURE.md` (the design) and `KG_EVALUATION_LITERATURE.md` (the
> lit review). It is the "change document with open decisions" that reconciles both against what is
> actually buildable given the code and the **zero-guaranteed-expert-hours** constraint. Where this
> doc and `BENCHMARK_ARCHITECTURE.md` disagree, **this doc wins** — it reflects later decisions
> (2026-06-10/11) and code reconnaissance the design doc predates.
>
> Companion docs: `BENCHMARK_ARCHITECTURE.md` (full L0–L3 design), `EVALUATION_OVERVIEW.md` (readable
> overview), `KG_EVALUATION_LITERATURE.md` (fact-checked KG-eval lit review), `CLAUDE.md` (baseline
> pipeline — note the BM25 correction in §8 below).

---

## 1. Purpose & what changed

`BENCHMARK_ARCHITECTURE.md` was declared design-complete 2026-06-08. Two things happened after:
the **KG-eval lit review** (2026-06-10) surfaced refinements, and a **scoping/recon session**
(2026-06-10) made five execution decisions and confirmed code realities that move several design
assumptions. This plan folds both in. The thesis claim is unchanged and remains **extrinsic-primary**:
*KG grounding measurably improves the photonic design task vs. the baseline, and the advantage grows
with search-space size.* What changed is **how we get ground truth** (no expert hours) and **what we
must build first** (GDSFactory integration).

## 2. The five locked scoping decisions (2026-06-10)

1. **KG must be wired into the pipeline's Phase-4 component selection.** Today Phase 4 uses lexical
   `search_components` (pdk_catalog); the KG retrievers (`kg_server.search_concepts` /
   `search_pdk_by_function`) exist but are standalone. Wiring them in is new engineering, not config.
2. **Benchmark retrieval at 262-component scale** (full GDSFactory generic library, per
   `Supplemental_Experiment_Description.md`). This **requires integrating GDSFactory into PhIDOv1
   first** — "Phase −1" (see §7), which gates everything downstream.
3. **Use only existing baseline arms** — lexical `search_components` + LLM-over-JSON (`llm_search`) +
   KG. **Do not build BM25.** It never existed; `CLAUDE.md` is wrong to call `llm_search`
   "BM25+embedding" (correction logged in §8).
4. **Paper knowledge must actually flow into component retrieval** via `enrich_pdk_from_components`
   (**Direction A**: copy a Component's paper-derived edges down to the `PDK_Cell` that IMPLEMENTS it,
   at flat `confidence=0.6`). The headline is therefore a **multiplicative chain**:
   paper-edge quality × IMPLEMENTS-linking quality × enrichment. The decisive test of this decision is
   the ablation in §4 (KG-with-enrichment vs KG-without).
5. **Zero guaranteed expert hours.** The annotator (Tony) is not a domain expert. This deletes the
   original sampling-based human anchor and the §4a IAA campaign, and forces the gold strategy in §3.

## 3. Ground-truth strategy under zero expert hours

The constraint is turned into the posture the lit review argues is *most* defensible for a domain with
no curated reference KG: **trade exhaustive gold for automation + a curated reference + a calibrated
judge.** Three tiers:

**Tier 1 — fully automatic / self-labeled (no judgment):**
- **E1** retrieval — each query is generated *from* a known source component → answer known by construction.
- **E2** funnel — DRC + SAX self-label.
- **A5** PDK ingestion — the DesignLibrary is machine-readable ground truth (enumerable).
- **L1** graph-health — duplication / contradiction / orphan / schema-conformance, all computed.
- **`GenerativeOntology`** — a textbook reference KG (16 primitives × closed vocabularies) supplies
  **recall + precision** on `PERFORMS_FUNCTION` / `BASED_ON_PRINCIPLE` / `HAS_PROPERTY` /
  `USES_COMPONENT`. This is the curated reference photonics was assumed to lack (answers lit-review C6).

**Tier 2 — judgment that does NOT need a PIC expert:**
- A1 extraction and **A2 evidence-grounded faithfulness** are **textual-entailment** tasks ("does the
  stored quote support this triple?"), not "is this physically true." Judged by an **LLM-judge**,
  **calibrated against `GenerativeOntology`** (no fresh human labels), with **order-swap + multi-vote**
  to counter the documented position bias. For paper edges using **out-of-ontology** concepts the judge
  is uncalibrated — those feed the expert judge-validation anchor (§5c).

**Tier 3 — needs an expert → DEFERRED:** edge **scientific correctness** (beyond text-supported) and
A4 promotion-quality / "is this a *good* abstraction." Not claimed in this thesis.

**Synthetic error injection** supplies the B4/B5 formal-gate labels: take known-valid topologies/params
(gold examples + enumerable PDK constraints), inject taxonomy-spanning errors → every label known by
construction. The same corrupted graph is the validation control for LP-Measure (§6).

**Reliability story (replaces the κ targets):** human(non-expert)↔LLM-judge agreement on a small set +
**LLM-judge self-consistency under order-swap**. The κ≥0.70/0.60 IAA targets assumed ≥2 experts → dropped.

**Stated scope boundary:** *we evaluate edge **faithfulness** (text-supported), not edge **scientific
correctness***, exactly parallel to the existing *"DRC-clean ≠ functionally correct."* The thesis
headline (extrinsic E1/E2, self-labeled) is fully defensible with zero experts; the expert-only parts
were always diagnostics.

## 4. The two headline experiments (restated for this scope)

### E1 — Retrieval A/B (the core claim)
- **Arms (3):** lexical `search_components` / LLM-over-JSON `llm_search` / **KG-grounded** — same LLM,
  same descriptions, same queries; the KG is the only treatment.
- **Tiered corpus (decision (a)):** real curated docstrings + SAX models for the **~27 modeled
  DesignLibrary components** (retrieval targets, E2-capable); **signature-grounded LLM-synthesized**
  descriptions for the **~235 distractors** (retrieval-only, *identical across arms* → fairness preserved).
- **Scale points 50 / 131 / 262** → the **scaling hypothesis**: KG advantage grows with distractor
  density. Framed precisely as **"KG robustness to distractor density"** (paper-enrichment lives mostly
  on the relevant ~27, not the distractors — stated, not overclaimed).
- **Queries:** paraphrase **+ functional/indirect** ("a component that does X") — required for the KG
  advantage to be observable.
- **Metrics:** pass@1/3, MRR/avg-rank, tokens/query, **set P/R** (compositional), and
  **`retrievable_coverage@k`** with the ceiling statement **E2 success ≤ retrievable coverage of the
  required component set** (lit-review C3).
- **The decisive ablation (tests decision #4):** **KG-with-paper-enrichment vs KG-without
  (ontology+PDK only)** — fully self-labeled, isolates the paper contribution.

### E2 — End-to-end A/B (the funnel)
- **~27 modeled components only** (the 262 have no SAX models → retrieval-only).
- Baseline pipeline vs GraphRAG pipeline, reported as the staged funnel
  `instantiate → routing_ok → SAX models → sim-success → DRC-clean`, complexity-stratified (Level 1–4),
  self-labeled via DRC.
- **Caveat:** funnel = manufacturable & simulatable, **not functionally correct** (deferred, tied to
  the `circuit_optimizer`).

## 5. Resolved design questions (a)/(b)/(c)

- **(a) GDSFactory integration depth → tiered design.** As in §4 E1. Keeps the full 50/131/262 scaling
  curve without hallucination (distractor descriptions are *signature-grounded*, not free-form).
  *Feasibility flag:* gdsfactory is **not yet installed** in `.venv`; signature metadata must be
  enumerated as the first Phase −1 task (§7).
- **(b) SEA order-invariance → descope from headline, pilot-gated.** Keep type-growth curve + per-paper
  churn (free from the single canonical rebuild) as the convergence diagnostic. Order-invariance is
  gated behind a cheap **k-repeat stochastic-churn-floor** measurement at fixed canonical order
  (order-effects are meaningful only if they exceed the no-seed noise floor); descope to future work if
  the floor is too high. Patch `process_papers.py` `glob` → `sorted()` regardless.
- **(c) Human validation → small guaranteed *judge-validation* anchor (≤5 annotators), designed for the
  floor (updated 2026-06-12).** Recruit **≤5** grad students/alumni, each labeling **faithfulness**
  ("does the highlighted quote support this triple?") on a triplet sample — primarily from **their own
  paper(s)** (deep expertise, frictionless <15-min session) and **oversampled on out-of-ontology edges**
  (where `GenerativeOntology` can't calibrate the judge).
  - **Primary use = validate the LLM-judge** (expert↔judge agreement); the validated judge then measures
    KG-wide precision. **No headline depends on a pooled precision CI** — with ≤5 document-clusters,
    clustered inference is fragile (≈4 df; effective n ≈ #documents/ICC).
  - **Reporting is conditional/post-hoc:** always per-paper accuracies + judge-agreement (descriptive,
    works at any cluster count); add a pooled df-adjusted **cluster t-interval** *only if* enough clusters
    materialize, captioned exploratory.
  - **Mandatory up front:** log `document_id` + `annotator_id` + per-item timing on **every** label —
    this is what lets us run whichever analysis the realized cluster count supports; unrecoverable later.
  - **Faithfulness framing** (not "is this correct?") keeps scope-consistent (Tier 2) and dampens
    author-generosity bias; convenience-sample representativeness stated as a limitation.
  - **Upside, absorbed without rework:** since faithfulness is entailment (authorship not strictly
    required), recruits *could* each label 2–3 papers → cluster count ~10–15 → the optional pooled CI
    tightens (more df). Whatever doesn't materialize → falls back to the zero-expert automatic+ontology
    backbone unchanged. (Open: whether any recruit will label >1 paper — design is robust either way.)

## 6. Lit-review refinements folded in

- **C1 — dissolved.** Automating the judge removes the human fixed cost that justified Gao's
  cluster/stratified sampling. → **Judge all edges** (or a large stratified sample if compute bites);
  report **per-edge-type precision + a CI reflecting judge-calibration error**, not sampling error.
- **C3 — adopted.** `retrievable_coverage@k` + the E2 ceiling (in §4 E1).
- **C5 — adopted (secondary).** **LP-Measure** (remove a fraction of triples → MRR/Hit@k recovery) as a
  fully-automated reference-free L1 sanity check, **validated on the synthetic-corruption control**, with
  the **Akrami de-leak caveat** (strip reverse/duplicate relations first). Secondary to evidence-grounded
  faithfulness; build if time allows.
- **C8 — deferred.** GraphRAG-Bench reasoning-coherence track → future work (itself LLM-judge-based;
  open-domain-validated only; not needed for the headline).
- Already covered by prior decisions: **C2 = decision #5**; **C4** (order-swap + multi-vote) already in
  the judge protocol; **C6** answered by `GenerativeOntology`; **C7** partially answered via §5c.

## 7. Phase −1 — GDSFactory integration (gates everything)

1. **Fix the environment** — gdsfactory is absent from `.venv` and `uv run` cannot re-sync (missing
   graphviz dev headers → `pygraphviz` build failure; see README's kfactory/graphviz notes). Install
   system `graphviz libgraphviz-dev` + the pinned gdsfactory/kfactory, confirm `import gdsfactory`.
2. **Enumerate the generic library** — `inspect.signature()` + `component.ports` over the generic PDK
   cells to confirm each exposes enough metadata (params/ports/category) to ground synthesis on.
3. **Signature-grounded description synthesis** for the ~235 distractors; freeze as shared infra
   (identical across arms); spot-check a sample for plausibility.
4. **Wire the KG retrievers into Phase-4 selection** (decision #1) and **`enrich_pdk_from_components`**
   (decision #4), behind a flag so the enrichment ablation (§4) is a toggle.

## 8. Corrections to existing docs (apply when touching them)
- **`CLAUDE.md`:** `llm_search` is **not** "BM25+embedding" — there is no BM25 (decision #3).
- **`BENCHMARK_ARCHITECTURE.md` §6.3:** "fixed temp/seed for determinism" is **not achievable** — no
  seed/temperature anywhere in the code → report **k-repeat variance**, not determinism.

## 9. Build order / backlog
0. **Phase −1** (§7) — environment + GDSFactory + KG wiring + enrichment toggle. *Gates all.*
1. **E1 retrieval harness** — query generation (paraphrase + functional), 3 arms, pass@k/MRR/coverage,
   the enrichment ablation. Cleanest central result, fully self-labeled.
2. **Headless baseline runner** — `main` is Streamlit-driven; GraphRAG has `run_pipeline()`.
3. **E2 funnel harness** — on the ~27 modeled set; DRC self-labeling (commit `8e819f2`).
4. **`GenerativeOntology` calibration set** for the LLM-judge (+ order-swap, multi-vote).
5. **Synthetic error-injection sets** for B4/B5 + the LP-Measure corruption control.
6. **SEA pilot** — `glob → sorted()`, k-repeat churn floor; multi-order only if floor is low.
7. **Expert judge-validation anchor + labeling tool** (§5c) — **BUILT** at `benchmark/labeling/`:
   zero-hosting, zero-install, async **static-HTML** labeler (one triplet/screen, highlighted evidence
   quote, binary + skip, keyboard/progress, `localStorage` autosave, Download→`labels_<id>.json`) +
   `make_labeling_bundles.py` (export per-annotator HTML from Neo4j; excludes `confidence` to avoid
   anchoring) + `ingest_labels.py` (per-paper rate, expert↔judge agreement+κ, clustered CI, overlap,
   timing). Faithfulness-framed; ≤5 annotators; out-of-ontology-weighted. **Export→label→import**, flat
   JSON, no Neo4j write-back. *(Pivoted from Streamlit — Streamlit needs a server, which violates the
   no-hosting/async constraint.)*

## 10. Stated limitations (carry into the thesis)
- **Faithfulness ≠ scientific correctness** (§3) — edges are judged text-supported, not physically true.
- **Manufacturable ≠ functionally correct** (E2 caveat) — funnel does not verify the requested spec.
- **LLM-judge** carries residual position/verbosity bias (mitigated, not eliminated) and is calibrated
  only where `GenerativeOntology` has coverage.
- **No determinism** — results reported as k-repeat variance.
- **Single canonical model** held fixed across arms; model-sensitivity = future work.

## 11. Decision log
- **2026-06-10:** Five scoping decisions locked (§2). Gold strategy redesigned around zero expert hours
  (§3). Code recon: 262 retrieval-only / ~27 modeled; no seed/temp; `process_papers.py` unsorted glob;
  only snapshot = full wipe.
- **2026-06-11 (a) RESOLVED:** tiered E1 corpus (real ~27 + signature-grounded-synthesized ~235); keeps
  the 50/131/262 scaling curve. Verified gdsfactory not installed → Phase −1 genuinely not started.
- **2026-06-11 (b) RESOLVED:** order-invariance descoped from headline, pilot-gated on the stochastic
  churn floor; type-growth + churn kept; `glob → sorted()`.
- **2026-06-11 (c) RESOLVED:** human spot-check = optional upside, prioritized drop-in queue; no
  dependency.
- **2026-06-11 lit-review refinements folded:** C1 dissolved → judge-all + per-type precision ± CI;
  C3 coverage ceiling adopted; C5 LP-Measure adopted (secondary, de-leaked); C8 reasoning-coherence
  deferred.
- **2026-06-12 (c) REFINED — expert ground-truth scoped to a ≤5-annotator *judge-validation* anchor,
  not a precision-CI claim.** With ≤5 document-clusters, clustered inference is fragile (≈4 df,
  effective n ≈ #docs/ICC) → experts **validate the LLM-judge** (expert↔judge agreement primary), the
  validated judge measures the KG. Out-of-ontology-weighted; **faithfulness-framed** ("label your own
  paper") to dampen author-generosity bias. Pooled precision CI = conditional/post-hoc (df-adjusted
  cluster t-interval, exploratory) only if clusters suffice. **Mandatory metadata:**
  `document_id`/`annotator_id`/per-item timing on every label. Design is robust to the unresolved
  "will a recruit label >1 paper?" question — built for the floor, multi-paper absorbed as upside.
- **2026-06-13: labeling tool BUILT** at `benchmark/labeling/` (export `make_labeling_bundles.py` +
  static-HTML `labeler_template.html` + analysis `ingest_labels.py` + README/recruit instructions).
  Pivoted Streamlit → **static self-contained HTML** to satisfy the no-hosting/async constraint:
  recruit opens an `.html` with their triples baked in, labels offline, emails back a JSON. `confidence`
  withheld from annotators (anti-anchoring); out-of-ontology oversampling; flat-JSON only (no KG
  write-back). Verified: scripts compile, ingest runs on synthetic data, HTML injection escapes
  `</script>` and excludes confidence. **Next = Phase −1** (env + GDSFactory) and populating the KB so
  real bundles can be generated.

# A2 Faithfulness — First Expert Round (Poon, paper #01)

Durable record of the first returned expert annotation bundle and the KG-construction
improvements it motivates. Companion to `E1_RETRIEVAL_FINDINGS.md`. The raw
`benchmark/labeling/results_poon.json` (ingest report) and `poon_triple_join.json`
(per-triple table) are gitignored under `results/`-style transience, so this doc is the record.

## What was measured
- **Annotator:** Joyce Poon (PI). **Paper:** `01_Silicon_photonics_for_visible_and_near_ir_spectrum`
  (an AOP **review** — broad multi-topic coverage; relevant caveat below).
- **Task:** relationship-validity ("Is this relationship correct for the work in this paper?",
  Correct/Incorrect/Unsure). 40 triples, all 40 labeled, 0 skips. Median 8.7 s/item (not rushed).
- **Judge:** DIA's own `is_related`/`confidence` verdicts. Every triple in the bundle is a
  **DIA-approved** edge, so the judge label is uniformly "supported" → this round validates
  **DIA precision** (RECALL — DIA-rejected edges — is a separate future arm).

## Headline result
- **Expert validity rate = 27/40 = 0.675** (13 rejected). Wilson 95% CI [0.52, 0.80] (single cluster).
- **Expert↔judge agreement = 0.675**; **κ = 0** (judge degenerate — all "supported", as designed).
  Agreement-rate is the validation number per plan. ⇒ **DIA precision ≈ 0.68 on this paper.**
- **Out-of-ontology subset (24 items): agreement 0.542** — DIA over-accepts worst exactly where
  the ontology can't check it (the load-bearing validation number).

## The failure pattern is sharp — `RELATED_TO` is the precision sink
| Relation | supported / total | rate |
|---|---|---|
| HAS_PROPERTY | 9 / 9 | **1.00** |
| BASED_ON_PRINCIPLE | 3 / 4 | 0.75 |
| PERFORMS_FUNCTION | 2 / 3 | 0.67 |
| **typed edges combined** | **14 / 16** | **0.875** |
| **RELATED_TO** | **13 / 24** | **0.54** |

- **11 of 13 rejections are `RELATED_TO`.** ooo (≈RELATED_TO) rejected 46% vs in-ontology 12%.
- **Confidence does NOT discriminate:** mean conf rejected 0.75 vs supported 0.80, and a rejected
  edge (`lasers --RELATED_TO--> lithium niobate`) scored **0.95**. Thresholding confidence will
  not cleanly remove the bad edges.
- Representative rejections: `bandgap (Property) --RELATED_TO--> {tunable optical couplers / optical
  filters / programmable lattice filter}` (a hub Property spuriously fanned out), `modulators
  --RELATED_TO--> {hybrid laser / Microlasers}`, `lasers --RELATED_TO--> photon detection`. These are
  cross-topic **co-occurrences** in a review paper, not real relations. The 2 typed misses:
  `adiabatic taper --BASED_ON_PRINCIPLE--> Spot Size Conversion` (SSC is a *function*, type-confused)
  and `modulators --PERFORMS_FUNCTION--> linear and nonlinear optical functions` (over-broad).

## Root causes (grounded in `dia_agent/dia_agent.py`)
1. **Confidence is a topical-similarity proxy, not a faithfulness score.**
   `combined_conf = 0.45·similarity + 0.15·evidence + 0.15·schema_ok + 0.25·llm_conf` (lines ~500-505).
   **45% is raw embedding similarity** → two topically-close entities score high regardless of whether
   the *specific relation* holds (hence the 0.95 false edge). This is why confidence won't filter.
2. **`RELATED_TO` is the catch-all committed "for connectivity."** When the LLM proposes a novel type
   or no schema type fits, the edge is committed as `RELATED_TO` with `schema_ok = 0.0` (lines ~493-494)
   and bypasses type-conformance (`queue_allowed` always true for RELATED_TO, line ~533). It is the
   dumping ground — and the precision sink.
3. **Edge evidence is entity-level pooled, not relation-specific.** The prompt guard ("reject if quotes
   don't support the relation", `prompts.py:20`) operates on pooled entity quotes, so it cannot actually
   see whether the *head→rel→tail* is supported. (`reextract_quotes.py` already found ~59% of edges have
   no directly-supporting sentence.)
4. **`SEMANTIC_THRESHOLD = 0.50` is low** (line 203) → many topically-similar-but-unrelated candidate
   pairs enter verification. A broad **review** paper (#01) maximizes spurious co-occurrence — likely a
   worst case; focused research papers should fare better.

## First-round improvements (prioritized)
- **B (biggest win): demote `RELATED_TO`.** Route generic RELATED_TO to the review queue instead of
  committing to the main graph, OR require relation-specific evidence before committing. Removes the
  bulk of the 46% ooo error at once; typed-edge precision (0.875) is already acceptable.
- **A: recalibrate confidence.** Cut the similarity weight (0.45 → ~0.10–0.15), raise `llm_conf` +
  `evidence` + `schema_ok`. Turns confidence into a usable precision filter (currently it isn't one).
- **C: relation-specific evidence at construction time.** Run the per-triple quote check
  (`reextract_quotes.py` logic) inside DIA so the prompt's support guard sees the actual triple.
- **D: tighten candidate generation.** Raise `SEMANTIC_THRESHOLD` (~0.65) and/or cap per-entity
  RELATED_TO fan-out to curb hub over-connection (bandgap/lasers/modulators).
- **E: light specificity/type check on typed edges** to catch the 2 typed misses (function-as-principle).

## Changes implemented (2026-06-24, `dia_agent/dia_agent.py`)
Both gated by constants in `infer_relationships` so they are cleanly ablatable for the thesis.
- **(A) Confidence recalibration.** `combined_conf` weights changed from
  `0.45·sim + 0.15·evid + 0.15·schema + 0.25·llm` to **`0.10·sim + 0.25·evid + 0.20·schema + 0.45·llm`**
  (approved branch; rejected/queue-gate branch `0.55/0.15/0.30·llm` → `0.15/0.25/0.60`). Confidence now
  tracks the LLM's relation-level judgment + evidence + schema conformance rather than embedding
  similarity. Constants `W_SIM/W_EVID/W_SCHEMA/W_LLM` + `WR_*`.
- **(B) Demote generic RELATED_TO.** Generic catch-all RELATED_TO (LLM fell back to it, `schema_ok=0`)
  is routed to the human-review queue (`reason="generic_related_to_demoted"`) instead of committed to
  the main graph. **Novel-type placeholders (`is_novel_type`) are deliberately NOT demoted** — they
  carry a recorded `RelationshipObservation` and must stay as RELATED_TO so SEA Phase-4
  (`_recategorize_related_to_edges`) can promote and re-type them; demoting them would silently break
  type promotion. Toggle `COMMIT_GENERIC_RELATED_TO=False` (set True to restore old behaviour / ablate).
- **Projected impact on Poon's slice** (labeled data, not a rebuild): 10/11 rejected RELATED_TO are
  generic → demoted; 11/13 *valid* generic RELATED_TO also demoted → routed to review (recoverable, not
  deleted). Committed-graph precision **0.675 → 16/19 = 0.842**, trading recall on the catch-all (E1
  showed RELATED_TO/enrichment is retrieval-neutral, so the served-graph cost is small).
- **Validation pending a KB rebuild** (Neo4j down; multi-hour). (A) cannot be re-scored from the bundle
  (needs per-edge similarity/llm_conf components); (B) impact above is a projection on the 40 labeled edges.

## Post-rebuild validation (2026-06-25, full clean rebuild with A+B live)
Full reset + 17/17-paper rebuild + PDK ingest, A+B active. **KB: 2931 nodes / 2293 edges; 34 PDK_Cell,
45 IMPLEMENTS.** Mechanism confirmed on the live graph:
- **(B) fires:** `RELATED_TO` is now only **123/2293 edges (5.4%)** — 70 VSA-path + 53 DIA novel-type
  placeholders (the generic DIA catch-all is gone). **128 generic RELATED_TO demoted** to the review
  queue (`ReviewItem reason='generic_related_to_demoted'`), separate from the 2228 pre-existing
  `semantic_verification_rejected` items.
- **Paper #01 (Poon's paper):** committed = 103 edges, of which **only 7 are RELATED_TO (7%)** —
  5 VSA + 2 DIA novel-type; **17 generic RELATED_TO demoted**. The error class that drove Poon's
  rejections (10/11 were generic RELATED_TO) is demoted out of the committed graph.
- **(A) live:** DIA committed edges carry the recalibrated confidence (n=1081, mean 0.888, range
  0.504–0.991).

**What this validates vs not:** the *mechanism* (A+B) is confirmed active and the committed-graph
composition shifted decisively to typed edges (#01 now 93% typed). The committed-precision **number**
(projected 0.842) is NOT yet re-measured — the rebuild is non-deterministic (no seed) so the new triples
have new element-ids and Poon's old labels don't map. A measured post-fix precision needs a fresh label
round (new bundle from this KB). Honest claim: A+B remove the dominant error class from the served graph;
the precision lift is projected, pending re-annotation.

## Bonus defect found + fixed: SEA type promotion was broken (`schema_registry.py`)
This build was the first to actually have a type to promote — SEA validated `CONTAINS_COMPONENT`
(Phase 1–3: 63 clusters → 1 gate-pass → LLM-validated) — but **Phase 4 promotion crashed** on a Cypher
syntax error: `MERGE … SET … ON CREATE SET …` (line ~327). `ON CREATE SET` must precede the bare `SET`.
**Fixed** (reordered ON CREATE SET before SET; dry-run-verified against live Neo4j). Pre-existing bug,
independent of A+B; takes effect next SEA pass. Note: this means novel-type placeholders never got
materialized in any prior build either — relevant to the "SEA promoted 0 types" history.

## Caveats (do not over-read)
- **N=40, one paper, one annotator** — directional pilot, not powered. #01 is a broad **review** =
  likely the worst case for co-occurrence inference. The other 4 annotators (focused research papers)
  are needed to see whether RELATED_TO precision is paper-type-dependent.
- The RELATED_TO signal (11/13 rejections, mechanistically explained by the confidence formula + the
  connectivity catch-all) is strong and self-consistent enough to act on now, while the rest return.
- This validates DIA **precision** only; **recall** (DIA-rejected edges) is unmeasured.

## Recall: three quantities, only one of which the "rejected-edge" arm can reach (2026-06-28)
A natural next arm is to sample DIA's **discarded** relationships, judge them with the same experts, and
treat expert-"valid" verdicts as false negatives. This is worth doing, **but it is not system recall**,
and the distinction must be stated to avoid over-claiming. Decompose: a true relationship reaches the
committed KG only if it clears two stages —
1. **Extraction/proposal recall (R_extract)** — PPC extracts the entities and VSA/DIA propose the edge as
   a candidate *at all*.
2. **Gate recall (R_gate)** — DIA's semantic verifier *accepts* it rather than rejecting/demoting it.

End-to-end **system recall ≈ R_extract × R_gate.** The rejected-edge arm samples only *proposed-then-
discarded* edges, so it estimates **R_gate alone** — recall **conditional on having been proposed**. It is
**blind to never-proposed relationships** (the R_extract term), which are invisible to any method that
inspects pipeline outputs (committed *or* discarded). So the rejected-edge arm is an **upper bound on
system recall**; the gap is exactly the extraction-stage misses, usually the larger loss. Report it as
**gate recall**, never as system recall.

**Sharper still — the rejected pool is 100% `RELATED_TO`** (measured on the 2026-06-28 KB): of the review
queue, **1970 `semantic_verification_rejected` + 134 `generic_related_to_demoted`, all `RELATED_TO`; zero
typed rejections.** Architectural cause: DIA's `is_related` gate runs **only on its global-inference
proposals, which are `RELATED_TO` connectivity catch-alls** (e.g. a Component→Property pair proposed as
`RELATED_TO`, not typed). **Typed edges (`PERFORMS_FUNCTION`, `HAS_PROPERTY`, …) come from VSA's direct
text extraction and are committed without a reject-and-queue step** — so dropped/wrong typed candidates
are never persisted. Consequence: the rejected-edge arm can only measure recall of the **`RELATED_TO`-
inference gate**; it says nothing about typed-edge recall (no typed-rejection population exists to sample).
Per-annotator rejected-pool sizes (all `RELATED_TO`): Poon 129, Sharma 67, Mu 115, Xue 73, Liu 211 — ample
for a gate-recall sample, though false negatives are rare (most rejects are genuinely bad), so a usable CI
needs a sizable N and a `reason`-stratified split (semantic-rejected vs A+B-demoted; the latter is the
direct audit of whether the A+B demotion discarded valid edges).

**To measure true system recall (incl. never-proposed + typed misses): source-grounded sampling.** Anchor
the denominator in the **paper text**, not pipeline outputs: stratified-sample passages/claims from each
annotator's paper → expert (or LLM-proposes/expert-adjudicates) names the relationship(s) each passage
asserts, *blind to the KG* → check KG presence. Recall = (asserted relations found in KG) / (asserted in
sampled passages). This is the only design that catches both never-proposed and typed misses and has a
well-defined sampling frame; it is a **heavier, separate annotation task** (not just more triples in a
bundle), so its per-expert cost should be scoped before committing. **Recommended framing for the thesis:**
precision (measured) + `RELATED_TO`-gate recall (cheap add-on, interleave discarded edges **blind** into
the existing bundles so experts can't anchor) constitute a well-scoped faithfulness result, *provided*
recall is labelled as gate recall with the R_extract / typed-edge limitation stated; the source-grounded
arm is the rigorous upgrade for a true system-recall claim.

## Full 5-annotator round — precision + gate recall (2026-07-22)
All five bundles returned (Poon #01, Sharma #08, Mu #05, Xue #12, Liu #13), each **40 DIA-approved
(`arm=precision`) + 40 DIA-rejected (`arm=recall`)** edges, blind-interleaved (400 triples total).
Roszko is a **redundant double-label of Xue's bundle** used for IAA only (below) and is **excluded**
from these counts to avoid double-weighting one paper. Join: expert `labels_*.json` ↔ DIA verdict via
`bundles/bundle_meta_*.json` (`arm`, `relation`, `is_out_of_ontology`); DIA verdict is the `is_related`
gate (`supported` = committed edge, `not_supported` = "Rejected by semantic verifier").

**Non-response handling (matters — see the correction note):** expert labels take four values —
`supported`, `not_supported`, `skip`, and `null` (blank, never viewed: `time_ms=0`). Both `skip` and
`null` are **non-responses** and are **excluded** from denominators; only `supported`/`not_supported`
count. **All 35 blanks are Poon's** — she completed 45/80 items in a ~9-min session and 35 were never
reached (all `time_ms=0`, scattered by `shown_order`). Counting blanks as `not_supported` (an early bug
here) spuriously crashed Poon's precision to 0.40; corrected, it is 0.842.

Definitions (this is the **is_related gate**, not the system):
- **Precision** = expert-`supported` / (`supported`+`not_supported`) among **DIA-approved** edges = TP/(TP+FP).
- **Gate recall (UPPER BOUND on system recall)** = TP/(TP+FN), FN = expert-`supported` among
  **DIA-rejected** edges (valid edges the gate wrongly discarded). Per the decomposition above, this is
  blind to never-proposed and dropped-typed edges (the R_extract term), so it **overstates** system recall.
  (Recall is unaffected by the blank bug — blanks fell in the TN cell, not TP/FN.)

**Pooled (n=174 precision-arm, 176 recall-arm labelled; blanks+skips excluded):**
- **Precision = 123/174 = 0.707**, Wilson 95% CI [0.635, 0.769].
- **Gate recall (UB) = 123/176 = 0.699**, Wilson 95% CI [0.627, 0.762].

| Annotator | Paper (type) | TP | FP | FN | TN | Precision (n) | Gate recall (n) |
|---|---|---|---|---|---|---|---|
| Poon | #01 (review) | 16 | 3 | 15 | 11 | **0.842** (19) | 0.516 (31) |
| Sharma | #08 (research) | 25 | 12 | 4 | 32 | 0.676 (37) | 0.862 (29) |
| Mu | #05 (research) | 21 | 19 | 6 | 34 | **0.525** (40) | 0.778 (27) |
| Xue | #12 (research) | 35 | 4 | 12 | 26 | **0.897** (39) | 0.745 (47) |
| Liu | #13 (research) | 26 | 13 | 16 | 23 | 0.667 (39) | 0.619 (42) |

Precision spans 0.53–0.90. **The review paper (#01) is NOT the worst case** (corrected) — Poon's #01 is
now the *highest* (0.842); the lowest is Mu #05, a focused research paper. So gate precision is **not
cleanly paper-type-dependent** the way the N=40 pilot suggested; the earlier "review = worst case" read
was an artifact of the blank-as-rejection bug. (Poon's 0.842 rests on only n=19 — CI [0.62, 0.94] — so
weight it accordingly; the pooled 0.707 rests on n=174.)

### The recall pool is ~entirely `RELATED_TO` — "gate recall" is really the `RELATED_TO`-gate recall
| Relation | TP | FP | FN | TN | Precision (n) | Gate recall (n) |
|---|---|---|---|---|---|---|
| RELATED_TO | 22 | 10 | 53 | 139 | 0.688 (32) | **0.293** (75) |
| HAS_PROPERTY | 50 | 14 | 0 | 0 | 0.781 (64) | 1.000\* (50) |
| PERFORMS_FUNCTION | 16 | 14 | 0 | 0 | **0.533** (30) | 1.000\* (16) |
| BASED_ON_PRINCIPLE | 20 | 4 | 0 | 0 | 0.833 (24) | 1.000\* (20) |
| CONTAINS_COMPONENT | 9 | 5 | 0 | 0 | 0.643 (14) | 1.000\* (9) |
| USES_COMPONENT | 6 | 4 | 0 | 1 | 0.600 (10) | 1.000\* (6) |

\* **Artifact, not a result.** Typed edges have **zero rejections in the pool** (FN=TN≈0) because DIA's
reject-and-queue step runs *only* on its `RELATED_TO` global-inference proposals; typed edges come from
VSA direct extraction and are committed without a rejectable step. So typed "recall = 1.0" means "no
typed-rejection population exists to sample," **not** "the gate keeps all valid typed edges" — dropped
typed candidates are never persisted (the R_extract loss) and are invisible here. The recall number is
only interpretable for `RELATED_TO`: **0.293** (of RELATED_TO edges experts call valid, the gate kept 29%).

By ontology status: **in-ontology (typed)** precision 0.719 (92/128) [0.64, 0.79], recall 1.000\*;
**out-of-ontology (~RELATED_TO)** precision 0.674 (31/46) [0.53, 0.79], recall 0.369 (31/84) [0.27, 0.48].

### Two findings (corrected)
1. **Typed-edge precision is lower than the pilot suggested, and `PERFORMS_FUNCTION` is the weak spot.**
   The 40-triple Poon round gave typed precision 0.875; on all five annotators it is **0.719 in-ontology**,
   with **`PERFORMS_FUNCTION` the clear weak spot at 0.533** (n=30). Demoting `RELATED_TO` alone will *not*
   get the committed graph clean — typed edges need a specificity/type check (improvement **E**).
   *`PERFORMS_FUNCTION` FP pattern* (14 FPs, all high-confidence 0.8–0.99 → confidence won't filter):
   the dominant error is **tail-is-a-Component, not a Function** — `active photonic circuit
   --PERFORMS_FUNCTION--> {Light Source, Optical Switch, Photodetector}` (c≈0.98), `power monitor -->
   Power Splitter`, `cantilever structures --> Modulator`; these should be `CONTAINS_/USES_COMPONENT`.
   The rest are **whole-system over-attribution** — `neurophotonic probe --> focus the emitted light`
   (c=0.99), `SiN waveguides --> electro-optic modulation` — a function of a sub-part assigned to the
   container/material. **Caveat for E:** the mistyped tails are already tagged `Design_Function` by
   upstream PPC typing (so the edge is schema-conformant) — a naive "is the tail Function-typed?" check
   passes them. E must verify the tail is a *genuine action/function* (not a device noun mistyped as one)
   and prefer a structural relation (`CONTAINS_/USES_COMPONENT`) when the quote is a composition statement.
   Full annotated FP list + design implications: `benchmark/labeling/FP_LIST_FOR_E.md`.
2. **A+B is directionally VALIDATED, not falsified (reversed from the pre-correction read).** Pooled gate
   precision rose **0.675 → 0.707**, and on Poon's own paper the projected **0.842 landed exactly**
   (16/19). The A+B demotion works both structurally (committed graph shifted to typed; demoted generic
   RELATED_TO correctly quarantined in the recall arm) **and** on the precision number. The reason pooled
   precision doesn't reach 0.84 is **not** a failure of A+B — it's that **typed-edge precision (0.72, with
   `PERFORMS_FUNCTION` at 0.53) is now the binding constraint**. Net: B fixed the RELATED_TO sink;
   typed-edge specificity (E) is the next lever.

### Build provenance (resolved 2026-07-22)
These bundles are from the **post-A+B (2026-06-25) clean rebuild**, regenerated 2026-06-29. Three
independent signals agree:
- **Reason codes only A+B/E emit:** recall arm carries `generic_related_to_demoted` (43) and
  `schema_nonconformant_typed_demoted` (1) alongside `semantic_verification_rejected` (156). These are
  the A+B/E demotion tags — absent from any pre-fix build.
- **Recalibrated (A) confidence:** precision-arm confidence n=200, range 0.585–1.000, mean 0.879 — matches
  the post-A live committed range (doc: n=1081, mean 0.888, range 0.504–0.991), not the similarity-dominated
  pre-A formula.
- **Git/timestamps:** `bundle_meta_*.json` dated 2026-06-29 14:15, commit `4f43af5` ("KG-construction
  fixes + benchmark variance synthesis; regen annotator bundles"), i.e. *after* the 2026-06-25 A+B
  rebuild and the `be3e8a3` recall-arm generator.

So the precision arm is the **post-demotion committed graph** (demoted edges are in the recall arm, not
here), and 0.63 is the honest post-A+B committed gate precision — the 0.842 projection is falsified on
this build. Repro: `bundles/bundle_meta_*.json` + `labels_*.json`.

## Inter-annotator agreement (Roszko ✕ Xue, 2026-07-22)
Roszko independently re-labelled Xue's exact 80-triple bundle (source `12_Xue_…`), 80/80 shared, no
overlap gaps — a genuine double-label for IAA (user-confirmed independent; flags corroborate:
`reasoning_viewed_with_prior_label = 0` for both, `reasoning_viewed` = 0 Roszko / 2 Xue).
- **Cohen's κ = 0.639** (substantial), n = 71, `skip` excluded, raw agreement 83.1%. **[headline]**
- Robustness: κ = 0.660 (n=69) also excluding Xue's 2 reasoning-viewed items; κ = 0.523 (n=80) with
  `skip` as a third category.
- Disagreement is **asymmetric** — Xue is the stricter judge (9 support→not_support vs. 3 the other way),
  quantifying the strictness axis behind the `RELATED_TO` precision sink rather than random noise.

| | Xue: supported | Xue: not_supported |
|---|---|---|
| **Roszko: supported** | 39 | 9 |
| **Roszko: not_supported** | 3 | 20 |

Reportable claim: the gold support judgments reproduce across independent domain experts (κ ≈ 0.64),
so the precision/recall numbers above rest on a reproducible ground truth, not one annotator's idiosyncrasy.

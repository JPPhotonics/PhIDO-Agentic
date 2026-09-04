# E1 retrieval benchmark — consolidated findings

Component-retrieval evaluation for the GraphRAG KG: does KG-grounded retrieval beat the lexical
catalog search at Phase-4 component selection (**plan decision #1**), and does **paper enrichment**
add retrieval value (**decision #4**)? This doc consolidates every E1 run (several raw reports in the
gitignored `benchmark/results/` were overwritten by later runs; numbers below are reconstructed from
those runs' logs + the conversation record). Single-run point estimates — extraction/retrieval has no
seed, so read small differences as noise.

## TL;DR
- **Decision #1 — CONFIRMED, conditionally.** KG retrieval beats lexical on **functional / indirect**
  queries (where the part isn't named); they tie on **named** queries (token overlap already suffices).
- **Decision #4 — paper enrichment is retrieval-NEUTRAL.** Once edges are confidence-weighted,
  `kg+enrich == kg`; enrichment neither helps nor hurts. (Unweighted, it slightly *hurts*.)
- **The shipped hybrid (lexical + confidence-weighted KG, RRF-fused) DOMINATES both arms** on every
  stratum and metric — RRF is complementary, not a compromise.

## Setup
- **KG / PDK:** 34 `PDK_Cell`s (DemoPDK), the buildable universe; same set is the candidate pool for
  all arms. (The 262-component generic library is retrieval-only / not wired — see Caveats.)
- **Arms:**
  - `lexical` — token-overlap over **name+description only** (`pdk_catalog_server.search_components`;
    the KG-derived `function` field is withheld to avoid leaking the answer).
  - `kg` — `PDK_Cell` embedding similarity + `PERFORMS_FUNCTION` reachability. `binary` = functional
    hit is all-or-nothing; `weighted` = functional boost scales with the **max edge confidence**
    (native 0.8–1.0 ≫ literature-inferred 0.6), so low-confidence enrichment edges can't outrank
    native structure.
  - `kg+enrich` — same, *including* paper-propagated (`provenance=inferred_from_literature`) edges.
  - `hybrid(prod)` — the shipped `mcp_servers/component_retrieval.retrieve_candidates`: lexical +
    confidence-weighted KG, **Reciprocal-Rank-Fusion**, top-8, lexical fallback if Neo4j is down.
- **Gold:** the domain-correct PDK module(s) per query, **arm-independent** (not the pipeline's own
  pick). Two sources: 18 hand-curated single-type intents (gold confirmed vs the GETTING_STARTED
  worked examples' `2_CS.txt`), and 231 queries decomposed from the 103 Testbench prompts via a DRAFT
  `TYPE_TO_MODULES` dictionary + LLM phrase→type typing (`build_e1_queries_from_testbench.py`).
- **Metrics** (`benchmark/metrics.py`): pass@1, pass@3, MRR, coverage@3. For the pipeline, **pass@3 /
  coverage is the relevant number** — the orchestrator hands the LLM the top-8 and it selects.

## The runs, in order

### Run 1 — synthetic queries (leaky) → NON-DIAGNOSTIC
91 queries (paraphrase "a {name}" + functional "performs {Design_Function}"), gold = source cell, and
the candidate `function` field exposed to lexical. Both query kinds were token-matchable, so lexical
won everything — a query-design artifact, not evidence against KG.

| arm | pass@1 | functional pass@1 | paraphrase pass@1 |
|---|---|---|---|
| lexical | 0.451 | 0.211 | 0.853 |
| kg | 0.231 | 0.123 | 0.412 |
| kg+enrich | 0.209 | 0.105 | 0.382 |

**Lesson:** queries answerable by token overlap can't reveal the KG's value; need indirect queries +
hide the leaky `function` field.

### Run 2 — 18 curated leak-resistant intent queries → KG WINS
Design-intent phrasing (e.g. "convert on-chip optical power into a photocurrent"), lexical sees only
name+description. *(saved: `benchmark/results/e1_retrieval.md`)*

| arm | pass@1 | pass@3 | MRR | coverage@3 |
|---|---|---|---|---|
| lexical | 0.333 | 0.500 | 0.452 | 0.444 |
| kg | 0.444 | 0.556 | 0.585 | 0.463 |
| kg+enrich | **0.500** | 0.556 | **0.594** | **0.509** |

The flip vs Run 1 is the point: lexical wins token-matchable queries, KG wins real intents.

### Run 3 — 231 Testbench queries (overall) → TIED
Realistic prompts, but they mostly *name* components ("a 2×2 MZI"), so lexical stays competitive on
the aggregate. *(saved: `benchmark/results/e1_retrieval_e1_queries_testbench.md`)*

| arm | pass@1 | pass@3 | MRR |
|---|---|---|---|
| lexical | 0.411 | 0.545 | 0.526 |
| kg | 0.424 | 0.511 | 0.518 |
| kg+enrich | 0.407 | 0.511 | 0.500 |

### Run 4 — stratified named vs functional (binary KG) → KG helps functional, enrichment HURTS
*(overwritten on disk; from run log + conversation)* 159 named, 72 functional.

| stratum | metric | lexical | kg (binary) | kg+enrich (binary) |
|---|---|---|---|---|
| named | pass@1 | **0.491** | 0.478 | 0.478 |
| functional | pass@1 | 0.236 | **0.306** | 0.250 |
| functional | MRR | 0.359 | **0.388** | 0.348 |

KG > lexical on functional; **enrichment is net-negative** — its flat-0.6 edges, given equal binary
weight, add false functional matches that displace the gold.

### Run 5 — stratified with confidence-weighting → enrichment NEUTRAL (decision #4 resolved)
*(overwritten on disk; from run log + conversation)* functional stratum (N=72):

| arm | pass@1 | pass@3 | MRR |
|---|---|---|---|
| lexical | 0.236 | 0.361 | 0.359 |
| kg+enrich (binary) | 0.250 | 0.333 | 0.348 |
| **kg (weighted)** | **0.319** | 0.375 | 0.405 |
| **kg+enrich (weighted)** | **0.319** | 0.375 | 0.396 |

Confidence-weighting removes the harm (0.250→0.319) **and** lifts the KG arm. `kg(weighted) ==
kg+enrich(weighted)` ⇒ **enrichment adds no retrieval lift.** Why: for the queried telecom cells the
paper-derived functions are largely *redundant* with native PDK structure, and the papers' genuinely
novel content is about components not in the PDK/queries.

### Run 6 — production hybrid validation → hybrid DOMINATES
The shipped `retrieve_candidates` (lexical + weighted-KG, RRF-fused) as an arm.
*(saved: `benchmark/results/e1_stratified.md`)*

| stratum | metric | lexical | kg (weighted) | **hybrid (prod)** |
|---|---|---|---|---|
| overall | pass@1 | 0.411 | 0.433 | **0.558** |
| overall | pass@3 | 0.545 | 0.506 | **0.797** |
| overall | MRR | 0.526 | 0.522 | **0.692** |
| named | pass@1 | 0.491 | 0.484 | **0.591** |
| named | pass@3 | 0.629 | 0.566 | **0.881** |
| functional | pass@1 | 0.236 | 0.319 | **0.486** |
| functional | pass@3 | 0.361 | 0.375 | **0.611** |

RRF rewards anything *either* arm ranks well, and lexical/KG succeed on *different* queries, so the
hybrid exceeds both — even on named queries. **pass@3 ≈ 0.80 overall** is the pipeline-relevant figure
(the LLM selects from the top-8).

## Decision outcomes
- **#1 (KG vs lexical):** supported — KG wins functional/indirect retrieval; the **hybrid** is the
  right production choice and is now wired into Phase-4 (`RETRIEVAL_BACKEND=hybrid`, default).
- **#4 (paper enrichment):** **retrieval-neutral** for in-PDK telecom components. Value, if any, lies
  in non-retrieval tasks (literature grounding, novel-component reasoning, the faithfulness KG) — not
  this benchmark. (Separately, the KG is majority-*inferred*; that drove the annotation reframe — see
  the labeling docs / memory, out of scope here.)

## Corpus ↔ query domain alignment (the root reason #4 reads neutral)
Measured against the KG (2026-06-23): the queries target **33/34** PDK cells; **24/33** received
*some* paper-derived (`inferred_from_literature`) enrichment, but **lopsidedly** — concentrated on
rings (`mrr_1x1`/`mrr_2x2` ≈39 each), heaters (35/19), `straight` (33), `photodetector` (30),
`_gc` (18), directional couplers (14/13), i.e. the topics the Poon-group papers cover — and **zero**
for `tw_mzm`, `mzi_1x2_pindiode_cband` (an L3 gold), `mzi_1x1*`, `polarization_splitter_rotator`,
`laser`, `crossing`. The corpus's actual subjects (paper-derived Architecture names) are a mix of
generic photonics (Mach-Zehnder, micro-ring modulators, splitter trees, meshes) and **Poon-group
specialty** (implantable/four-shank/eight-beam probes, grating-based light emitters, microlasers) —
i.e. **visible/NIR neural-probe/laser photonics**, a different domain from the **generic-telecom**
queries+PDK.

Consequence: the KG retrieval's real signal is the **DemoPDK** (which matches the queries); the
**paper corpus** contributes little *relevant* signal for telecom queries, and what it does add
overlaps components the PDK already describes natively. So **E1 as built tests PDK-grounded
retrieval, not paper-augmented retrieval**, and the decision-#4 "neutral" is substantially a
corpus↔query **domain mismatch** — not proof that paper enrichment can't help retrieval.

## Caveats / limitations
- **DRAFT gold.** `TYPE_TO_MODULES` + the LLM phrase→type typing need Poon-group sign-off; the
  testbench's permissive gold sets inflate absolute numbers vs the 18-curated single-target gold.
- **Underpowered.** Functional N=72; deltas are a handful of queries — directionally consistent, not
  a powered estimate.
- **34-cell scale.** The 262-component generic library would test distractor density but is
  retrieval-only (not buildable end-to-end), and would need ingesting into both indices.
- **Non-deterministic.** No seed; small cross-run differences (e.g. kg(weighted) 0.319 vs 0.433
  across runs) are noise — treat per-paper/per-run numbers as a distribution.

## A fair (domain-matched) test of paper-augmented retrieval — proposed
To actually test whether literature knowledge helps retrieval, align corpus ↔ PDK ↔ queries:
- **Controlled corpus swap (recommended):** hold the telecom PDK + queries fixed; compare enrichment
  from (a) the current off-domain Poon-group corpus vs (b) a **telecom/C-band Si-photonics corpus**
  (papers about MZI modulators, MMI splitters, ring filters — the queried components). If enrichment
  helps with (b) but not (a), the mismatch — not enrichment per se — is the cause.
- **Or domain-flip:** build queries + a retrieval-only component set in the *corpus's* domain
  (visible/NIR, neural probes, OPAs, grating emitters) so the paper knowledge is on-topic. Needs a
  visible/neural-probe component library (DemoPDK is telecom); retrieval-only (those parts aren't
  end-to-end buildable).
- **Pre-check (RAN 2026-06-23, `run_e1_corpus_covered.py`) — result: REDUNDANCY, not dilution.**
  Restricting the weighted ablation to the **219 queries whose gold cell the corpus enriched**
  (literature `PERFORMS_FUNCTION` edges), `kg+enrich` vs `kg` is **Δpass@1=+0.000, Δpass@3=+0.000,
  Δmrr=-0.010** — i.e. enrichment is inert *even where the corpus overlaps the queries*. So the
  neutral result is genuine redundancy with native PDK structure (the paper edges duplicate
  functions the PDK already encodes; confidence-weighting keeps the 0.6 edges below native ones),
  not dilution by uncovered targets. (Covered/uncovered split is lopsided 219/12 — gold is a set,
  so a query is 'covered' if any gold member is enriched; the UNCOVERED arm n=12 is uninformative.)

## Reproduce
```
CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_retrieval.py [benchmark/e1_queries_testbench.json]
CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_stratified.py
# query set (testbench): benchmark/build_e1_queries_from_testbench.py
```
Reports land in `benchmark/results/` (gitignored). Code: `benchmark/{kg_retrieve,run_e1_retrieval,
run_e1_stratified,build_e1_queries_from_testbench}.py`, `mcp_servers/component_retrieval.py`.

## Provenance
Track-C commits: `2bf4d73` (harness + 18-curated), `f5d90a1` (testbench + stratified), `47ce6b3`
(confidence-weighted retriever; decision #4), `23bb725` (hybrid Phase-4 wiring), `545178d` (hybrid
validation).

## Rebuild comparison (2026-06-25) — A+B faithfulness build vs prior build
Re-ran the full E1 suite on the post-A+B clean rebuild (2931 nodes / 2293 edges; RELATED_TO demoted,
DIA confidence recalibrated). Same query files, same 34-cell pool, arm-independent gold → only the KB
content changed. **Control: the lexical arm is byte-identical across builds** (curated 0.333/0.500/0.452;
testbench 0.411/0.545/0.526), confirming the harness/queries/gold are fixed and differences are pure KB.

| run | arm | prior (pass@1/pass@3/mrr) | new (pass@1/pass@3/mrr) |
|---|---|---|---|
| 18 curated | kg | 0.444/0.556/0.585 | 0.444/0.556/0.545 |
| 18 curated | kg+enrich | 0.500/0.556/0.594 | 0.444/0.556/0.543 |
| 231 testbench | kg(weighted) | 0.433/0.506/0.522 | **0.260/0.338/0.355** |
| 231 functional | kg(weighted) | 0.319/0.375/0.405 | **0.167/0.222/0.252** |
| 231 overall | hybrid(prod) | 0.558/0.797/0.692 | 0.498/0.667/0.619 |
| 231 functional | hybrid(prod) | 0.486/0.611/0.427 | 0.292/0.472/0.427 |
| enrichment Δ (covered) | kg+enrich − kg | +0.000/+0.000/+0.002 | +0.000/+0.000/+0.002 |

**Findings:**
1. **Decision #4 (enrichment retrieval-neutral) — robustly replicated, now stronger.** `kg == kg+enrich`
   on every set (Δpass@1=+0.000 incl. corpus-covered subset). The small curated *lift* the prior build
   showed (0.500 vs 0.444) vanished → it was noise. Build-independent result.
2. **Decision #1 (KG > lexical) — holds on curated intents, REGRESSED on testbench.** KG functional
   pass@1 0.319→0.167 (now below lexical 0.236).
   **MECHANISM — MEASURED, not conjecture (2026-06-25, instrumented the KG arm on the 72 functional
   queries): the cause is RANKING DILUTION, not sparsity / unreachability.**
   - Gold is **reachable for 48/72 = 0.667** of functional queries (semantic_search(Design_Functions,
     limit=3, thr=0.4) → PERFORMS_FUNCTION → cell). Reachability is NOT the bottleneck.
   - But the functional traversal pulls a **median of 15 cells (of 34)** per query (58/72 pull ≥10),
     because (i) `semantic_search` often matches *irrelevant* Design_Functions ("1x2 MMIs" → depletion /
     Internal Gain / VOA; the same "popular" functions recur) and (ii) those functions are over-connected.
     All boosted cells tie on the functional term (×1000) → gold lands at **median rank 12**. Rank of the
     48 reachable: 12 at rank 1, 4 at 2–3, 7 at 4–8, **25 at >8** → reproduces pass@1 0.167 / pass@3 0.222.
   - Same over-connection / vocabulary-fragmentation mechanism documented for the prior build, worse here.
     A+B touch only generic RELATED_TO + confidence *values*; neither removes PERFORMS_FUNCTION edges.
   **NOT KNOWN:** the prior build's PERFORMS_FUNCTION quantity / dilution (KB wiped → unmeasurable), so
   WHY prior scored 0.319 vs 0.167 cannot be attributed with data. The earlier "sparser functional layer
   (57 vs ~74 Design_Functions)" line was CONJECTURE and is WRONG on mechanism — breadth/dilution, not sparsity.
   (Also: the named/functional split has a plural bug — `\bmmi\b` misses "MMIs" — so named queries leak
   into "functional"; affects both builds equally.)
3. **Hybrid(prod) still dominates** overall/named (lexical half is stable); functional edge narrowed only
   because its KG half weakened this build.

**METHODOLOGICAL HEADLINE:** the functional-KG number is (a) **fragile to KB-rebuild non-determinism** and
(b) limited by a **retriever-design issue** — over-broad Design_Function matching → dilution — which is
independently fixable (see the "Reducing Design_Function over-matching" section). The enrichment-neutral
result (#4) is robust across builds. KG retrieval should be reported as a distribution over seeded/temp=0
rebuilds before the functional-KG advantage is treated as a stable claim. New raw reports:
`benchmark/results/e1_*.{json,md}` (gitignored), run log `benchmark/results/e1_rerun_20260625.log`.

## Reducing Design_Function over-matching (2026-06-25) — the functional gate is NET-HARMFUL
Tested retriever variants on the current KB (no rebuild), sharing one set of embedding searches; gold
fixed, only scoring changes. Stratifier plural bug fixed here (`\bk s?\b`) → testbench splits 206 named /
25 functional (was 159/72 — the old "functional" stratum was mostly named-plurals). Driver:
`benchmark/run_e1_dfmatch_ablation.py`; log `benchmark/results/dfmatch_ablation.log`.

| variant | testbench overall p@1/p@3/mrr | curated-18 overall |
|---|---|---|
| baseline (funcs sim≥0.4 top3, boost=max edge_conf, ×1000 gate) | 0.260/0.338/0.355 | 0.444/0.556/0.545 |
| simw (boost ×= func_sim) | 0.147/0.234/0.269 | 0.333/0.500/0.464 |
| thr55_lim2 (sim≥0.55, top2) | 0.229/0.255/0.320 | 0.444/0.500/0.517 |
| idf (down-weight over-connected funcs) | 0.229/0.260/0.316 | 0.278/0.444/0.397 |
| soft_add (functional as soft signal, no ×1000 gate) | 0.273/0.351/0.375 | 0.444/0.556/0.545 |
| **sem_only (PDK_Cell embedding ONLY, no functional gate)** | **0.385/0.519/0.513** | **0.556/0.722/0.663** |

**FINDINGS:**
1. **No reweighting/thresholding of the function match helps** — all of simw/thr/idf LOSE vs baseline.
   `simw` hurts MOST because the wrong functions are matched with *high* similarity (popular functions
   like depletion/VOA embed close to many queries), so similarity-weighting amplifies the wrong ones.
   ⇒ the dilution is not fixable by reweighting the existing query→Design_Function matches.
2. **Dropping the functional gate entirely (`sem_only`) is strictly best** — both query sets, both strata
   (never worse, incl. genuine functional). Testbench overall +0.125 p@1 / +0.181 p@3 over baseline;
   curated +0.111 / +0.166. The PERFORMS_FUNCTION functional-boost is net-harmful: its dilution outweighs
   any signal.
3. **`sem_only` (a KG-embedding retriever) crushes lexical on design-intents** (curated overall 0.556 vs
   lexical 0.333) and roughly matches lexical on named testbench queries (0.385 vs 0.411). This is a
   CLEANER decision-#1 win than the functional-boost arm ever showed.
4. **Synthesis with #4:** paper enrichment is retrieval-neutral AND the functional-graph matching is
   net-harmful ⇒ on this KB the **PERFORMS_FUNCTION layer provides no positive retrieval value; the KG's
   retrieval value is entirely in the PDK_Cell embeddings.** RECOMMENDATION: the KG retrieval arm (and the
   hybrid's KG half) should rank by cell embedding, not the Design_Function functional traversal.

CAVEATS: DRAFT gold, single build, post-hoc selection of sem_only (though it is the zero-parameter / least
over-fittable variant). Curated n=18. To confirm stability, `sem_only` is included as an arm in the
seeded multi-build variance campaign.

## Fix re-run on the real drivers (2026-06-25) — functional gate OFF (`kg_embed`) wired into the suite
Added `functional=False` to `make_kg_retriever` (embedding-only) and re-ran the full drivers with a new
`kg_embed` arm + a `hybrid_embed` (lexical + kg_embed, RRF) alongside the originals. Stratifier plural bug
fixed → strata are 206 named / 25 functional. Log: `benchmark/results/e1_rerun_fix_20260625.log`.

| set / stratum | lexical | kg (functional) | **kg_embed (FIX)** | hybrid(prod) | hybrid_embed |
|---|---|---|---|---|---|
| curated-18 (p@1/p@3/cov@3) | 0.333/0.500/0.444 | 0.444/0.556/0.509 | **0.556/0.722/0.676** | — | — |
| testbench overall (p@1/p@3/mrr) | 0.411/0.545/0.526 | 0.260/0.338/0.355 | **0.385/0.519/0.513** | 0.498/0.667/0.619 | 0.394/0.688/0.566 |
| named (206) | 0.422/0.563/0.537 | 0.282/0.359/0.376 | 0.422/0.563/0.553 | **0.524**/0.709/0.647 | 0.427/**0.733**/0.600 |
| functional (25) p@1 | **0.320** | 0.080 | 0.080 | 0.280 | 0.120 |

**FINDINGS:**
1. **Standalone KG arm: the fix is a large real gain** (curated p@1 0.444→0.556, cov@3 0.509→0.676;
   testbench 0.260→0.385) — confirms on the real drivers that the functional gate was net-harmful.
2. **kg_embed > lexical on design-intents** (0.556 vs 0.333, cov@3 0.676 vs 0.444) and **= lexical on
   named** (0.422/0.563). Cleaner decision-#1 win than the functional arm.
3. **REVERSAL on "KG wins functional":** on the CLEAN 25-q functional stratum **lexical wins (0.320)**,
   both KG variants poor (0.080). The prior "KG wins functional" was a stratifier-pollution artifact
   (named-plurals). Caveat n=25.
4. **Hybrid is NUANCED — don't blanket-swap.** hybrid(prod) keeps best p@1/mrr (0.498/0.619); hybrid_embed
   best p@3 (0.688/0.733 named). kg_embed beats kg(functional) standalone, but in RRF the functional KG
   half adds more because its errors are DECORRELATED from lexical (fusion rewards diversity); kg_embed is
   correlated with lexical. ⇒ drop the functional gate for a standalone KG retriever; the hybrid needs its
   own A/B (and the variance campaign) before changing production.
CAVEATS: single build, DRAFT gold, functional n=25. kg_embed/hybrid_embed added to the variance campaign.

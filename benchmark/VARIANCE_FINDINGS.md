# KB-rebuild variance — consolidated findings

How much does a single KB rebuild move the numbers? The build pipeline has **no seed**, so every
rebuild of the same 17-paper corpus produces a different graph. This campaign rebuilt the KB **5
times** — 3 at production temperature (`prod`), 2 at temperature 0 (`temp0`) — and ran the full E1
retrieval suite after each. Companion to `E1_RETRIEVAL_FINDINGS.md` (which flagged "report KG
retrieval as a distribution over rebuilds" — this quantifies that distribution). Data:
`benchmark/results/variance/build_*/*.json`; regenerate the tables with
`benchmark/variance_report.py` → `benchmark/results/variance_report.md`.

## TL;DR
- **The KB build is materially non-deterministic, and temperature 0 does NOT remove it.** Edge
  count ranges **2298–3205** across the 5 builds (CV ≈ 15%); the two `temp0` builds *alone* still
  differ by **298 edges** (2907 vs 3205). So the variance is not just LLM sampling temperature.
- **The variance is localized.** It lives entirely in the **paper-inferred** graph structure. Arms
  that traverse it (`kg`, `kg+enrich`, `kg(weighted)`, `hybrid(prod)`) swing build-to-build; arms
  grounded in the **deterministic PDK_Cell embeddings** (`lexical`, `kg_embed`, `hybrid_embed`) are
  **byte-identical across all 5 builds** (0.000 range, to 16 decimals).
- **Implication:** report every KG-derived absolute number as mean ± spread over ≥3 rebuilds (or
  with an explicit single-build caveat). Cross-arm comparisons *within one build* remain valid
  (same KB); absolute levels are a distribution.

## Structural non-determinism (per-build KB composition)
Same corpus + same pipeline, no seed:

| field | mean | min–max (all 5) | CV% | temp0 min–max |
|---|---|---|---|---|
| nodes | 3215 | 2733–3814 | 14.3% | 3381–3814 |
| edges | 2653 | 2298–3205 | 14.7% | 2907–3205 |
| design_fn nodes | 66 | 52–83 | 18.3% | 70–83 |
| PERFORMS_FUNCTION | 579 | 473–712 | 15.3% | 609–712 |
| HAS_PROPERTY | 931 | 745–1122 | 17.1% | 1065–1122 |
| RELATED_TO | 131 | 80–173 | 27.3% | 156–173 |
| pf_native (PDK) | 72 | 64–81 | **9.8%** | 73–81 |
| pf_inferred (PDK) | 86 | 66–126 | **27.4%** | 77–126 |

- **Stable end:** `pf_native` (native PDK PERFORMS_FUNCTION) is the lowest-CV edge field (9.8%) —
  it derives from the deterministic library, not the LLM. **`pf_inferred` (literature-propagated)
  is the highest (27.4%)** — the inferred layer is where rebuilds disagree most.
- **temp0 is not deterministic:** the two temp0 builds differ on every field (e.g. PERFORMS_FUNCTION
  609 vs 712). Non-temperature variance sources: embedding/threshold ties in candidate generation,
  Python dict/iteration order, and async interleaving during ingest.

## Retrieval-metric non-determinism (E1)
Worst-case single-arm swings across builds (the cost of trusting one build):

| set / arm | metric | mean | min–max | range |
|---|---|---|---|---|
| curated-18 / kg | pass@1 | 0.467 | 0.333–0.556 | **0.222** |
| curated-18 / kg | pass@3 | 0.589 | 0.444–0.722 | **0.278** |
| testbench / kg | pass@3 | 0.442 | 0.368–0.506 | 0.139 |
| stratified functional / kg(weighted) | pass@1 | 0.120 | 0.080–0.160 | 0.080 (CV 24%) |
| stratified overall / hybrid(prod) | pass@3 | 0.771 | 0.693–0.805 | 0.113 |
| **lexical / kg_embed / hybrid_embed** | **all** | — | **identical** | **0.000** |

- On the 18-query curated set the `kg` arm swings up to **0.28 pass@3** between builds — larger than
  most of the cross-arm differences the E1 study reports. Any single-build curated number is inside
  this band.
- The small-n **functional** stratum (n≈25) has the highest CV (24% on `kg(weighted)` pass@1):
  small denominator × inferred-edge variance.
- **`hybrid(prod)` inherits variance from its KG half** (overall pass@3 0.693–0.805); `hybrid_embed`
  does not (stable), because its KG half is the embedding arm.

## The sharp result: variance ⇄ the paper-inferred layer
The split is clean and mechanistic:

| stable across builds (0.000 range) | varies across builds |
|---|---|
| `lexical` (name+desc only, no KG) | `kg`, `kg+enrich` (functional traversal) |
| `kg_embed` (PDK_Cell embeddings only) | `kg(weighted)` (weighted functional traversal) |
| `hybrid_embed` (lexical + kg_embed) | `hybrid(prod)` (lexical + weighted KG) |

`kg_embed` is a **deterministic function of the 34 PDK_Cell embeddings** (re-created identically each
build from the library, deterministic embedding model) and the fixed queries — so it cannot vary, and
empirically does not. Everything that varies routes through `PERFORMS_FUNCTION` edges, whose inferred
portion is the highest-CV structural field. **This converges with the E1 finding that the
`PERFORMS_FUNCTION` functional traversal is net-harmful** (`E1_RETRIEVAL_FINDINGS.md` → "Reducing
Design_Function over-matching"): the functional layer is not only retrieval-negative on average, it is
also the **sole source of build-to-build retrieval non-determinism**. Ranking by PDK_Cell embedding
(`kg_embed`) is both higher-scoring *and* reproducible.

## Caveats
- **n=5 builds (3 prod + 2 temp0).** CVs are directional; temp0 has only 2 points so its min–max is a
  range, not a powered estimate of residual variance.
- **Same corpus/queries/gold across builds** (verified by the byte-identical `lexical` arm) → the
  observed spread is pure KB-build non-determinism, not harness drift.
- The campaign holds the DRAFT testbench gold fixed; absolute levels still carry the `TYPE_TO_MODULES`
  gold caveat from the E1 study. Variance (the quantity of interest here) is gold-independent.

## Recommendations
1. **Report KG numbers as mean ± spread over ≥3 rebuilds**, or state "single build, ±~15% expected."
2. **Prefer the embedding-grounded retriever (`kg_embed`)** — it is the reproducible *and* stronger
   arm; the functional traversal adds variance and (per E1) no net retrieval value.
3. **If determinism is required** (e.g. a frozen thesis artifact), freeze a single KB snapshot and
   pin it; do not assume temp0 yields a reproducible graph.
4. **Future:** seed the build (where the pipeline allows) and re-measure residual variance to separate
   LLM-sampling from the structural (tie/order/async) sources.

## Reproduce
```
python benchmark/variance_report.py        # aggregates results/variance/build_*/*.json
# campaign itself: benchmark/run_variance_campaign.sh  (5 rebuilds + E1 suite; multi-hour)
```

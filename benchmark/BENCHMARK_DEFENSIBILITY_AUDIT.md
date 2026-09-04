# Benchmark Defensibility Audit

_Author: internal review · Date: 2026-07-01 · Branch: `graphRAG-implementation` (wt-graphrag)_

## Purpose & method

An adversarial pre-defense review of the benchmarking framework: where could a thesis
committee credibly attack the current test setup? Findings are ranked by severity, each
with file/line evidence and a minimal fix. This is a methodology critique, not a claim
that the underlying system is bad — several problems are presentation/statistics issues
that the honest status prose (`BENCHMARK_STATUS_REPORT.md`) already half-admits but the
headline artifacts do not reflect.

**Claim under test** (`EVALUATION_OVERVIEW.md:15-26`):
> KG grounding measurably improves the photonic design task vs. the baseline, and the
> advantage grows with search-space size.

Every finding is judged against whether the setup can defend *that* sentence.

---

## 0. Headline correction found during this audit (act on this first)

The E2 headline reported `McNemar p = nan` on all five funnel stages, in **both** the
current run (`results/e2_headline.md`, N=8/K=1) and the larger `*_pre_gate` run
(N=24/K=3). This is a **display bug**, not a statistics failure:

- `run_e2_headline.py:166` formatted the p-value with `.get('p', float('nan'))`, but
  `stats.mcnemar` returns the key **`'p_value'`** (`stats.py:62`). The key never matched,
  so every cell printed `nan`. (The ablation scripts use `['p_value']` correctly —
  `run_ablation_sweep.py:83`, `run_ablation_sweep_k.py:150` — which is why only the
  headline was affected.)
- **Fixed** in this audit (`run_e2_headline.py:166`, `'p' → 'p_value'`). The two headline
  `.md` files must be regenerated to reflect it.

Recomputing McNemar from the stored per-prompt results (exact two-sided binomial, majority
vote over K) gives the numbers that *should* have been on the headline:

**N=24, K=3 (`e2_headline_pre_gate_20260629.json`) — the larger run:**

| stage        | baseline | graphrag | b (gr wins) | c (base wins) | exact p |
|--------------|:--------:|:--------:|:-----------:|:-------------:|:-------:|
| instantiate  | 0.96 | 0.75 | 0 | 5 | 0.0625 |
| routing_ok   | 0.96 | 0.58 | 0 | 9 | **0.0039** |
| models       | 0.88 | 0.29 | 0 | 14 | **0.0001** |
| sim_success  | 0.88 | 0.29 | 0 | 14 | **0.0001** |
| drc_clean    | 0.71 | 0.25 | 0 | 11 | **0.0010** |

**N=8, K=1 (`e2_headline.json`) — the current "headline":**

| stage        | baseline | graphrag | b | c | exact p |
|--------------|:--------:|:--------:|:-:|:-:|:-------:|
| instantiate  | 0.88 | 1.00 | 1 | 0 | 1.0000 |
| routing_ok   | 0.75 | 0.50 | 0 | 2 | 0.5000 |
| models       | 0.75 | 0.12 | 0 | 5 | 0.0625 |
| sim_success  | 0.75 | 0.12 | 0 | 5 | 0.0625 |
| drc_clean    | 0.75 | 0.12 | 0 | 5 | 0.0625 |

**Interpretation that matters for the thesis:** fixing the bug does *not* rescue the
claim — it produces a **statistically significant negative result** at N=24 (drc_clean
p=0.0010; every discordant pair favors the baseline, `b=0` at every stage). The N=8 run is
non-significant (p=0.0625) *only because N was cut*, not because the effect disappeared.
Headlining the N=8/K=1 file therefore both (a) loses significance and (b) discards the
run-to-run instability the N=24/K=3 run had measured (GraphRAG non-unanimous rate
0.38–0.67). See Tier 1, items 1–2.

---

## Tier 1 — Indefensible as currently stated

### 1. The headline shipped no valid significance test
Both headline `.md` files show `McNemar p = nan` everywhere (root cause + fix in §0). A
committee opening the primary result table and seeing `nan` on the make-or-break
comparison is the worst possible first impression. **Fix:** regenerate both reports with
the corrected formatter; report exact-test p with discordant counts (b, c).

### 2. The "corrected" headline is weaker than the run it replaced
The current headline (`e2_headline.md`, 2026-06-30 11:44) is **N=8, K=1**; the larger run
(`e2_headline_pre_gate_20260629.md`) is **N=24, K=3**. Post-whitelist-gate GraphRAG still
collapses (drc_clean 0.12 vs 0.75, Δ −0.62); the gate moved only *instantiate*
(0.75→1.00), cosmetic because the failure relocates to `models`. Presenting the smaller,
unrepeated file as the headline while the larger repeated file is renamed `*_pre_gate` and
shelved reads as selecting the quieter dataset after seeing results.
**Fix:** the headline must be the largest repeated run (N=24, K=3). If the simulatable
whitelist is a legitimate fairness control, apply it to *that* dataset and re-report — do
not reduce N. State plainly that the extrinsic claim does not currently hold
(`BENCHMARK_STATUS_REPORT.md:13` already concludes this).

### 3. LLM-judge circularity + a degenerate validation
The faithfulness judge defaults to `gpt-4o` (`judge.py:100`); the system-under-test model
is not pinned to differ from it, so if any pipeline/KG calls use an OpenAI model the judge
grades its own family (self-preference). The single human-validated slice gave **κ = 0**
(`A2_FAITHFULNESS_FINDINGS.md:19-22`) because the annotated pool was 100% DIA-accepted
edges (no negatives) — so the 0.675 "agreement" measures expert recall of accepted edges,
not judge reliability. No pre-registered calibration on a balanced set.
**Fix:** judge with a different model family; validate on a class-balanced pool
(accepted + rejected); report κ with CI; treat κ=0 as blocking.

### 4. No expert-validated gold underpins any headline number
- E1 `TYPE_TO_MODULES` is a hand-drafted table "pending Poon-group sign-off" with
  documented judgment calls (`build_e1_queries_from_testbench.py:35-59`).
- E1 query *types* are LLM-extracted, so extraction bias propagates into gold membership.
- E2 tiers are "heuristic, pending Poon-group review" (`e2_prompts.json:2`).
- A2 rests on **N=1 annotator, N=1 paper**; precision 0.675→0.84 is a **projection, never
  re-annotated** (`BENCHMARK_STATUS_REPORT.md:28,87`).

**Fix:** obtain sign-off on E1 `TYPE_TO_MODULES` and E2 tiers before quoting any number as
final; report A2 with re-annotated precision or drop the 0.84.

---

## Tier 2 — Serious, will draw sustained questioning

### 5. Whole-system confound defeats the actual thesis sentence
The claim is about *KG grounding*, but the E2 agentic arm bundles
"KG grounding + topology gate + critic" vs a bare baseline (`e2_headline.md` fairness
note), so the Δ cannot be attributed to KG grounding. The attribution ablations are
underpowered: gate p=0.375, critic negligible, 35–65% stochastic flips at N=20
(`BENCHMARK_STATUS_REPORT.md:99`); clarification at N=6 (`:103`).
**Fix:** scope the headline claim to "the agentic system as a whole," or power the
KG-on/KG-off ablation to significance.

### 6. The dependent variable doesn't match the claim
E2 "pass" = manufacturable & simulatable, explicitly **not functionally correct**
(`EVALUATION_OVERVIEW.md:98-107`). A design that misses the requested FSR/splitting ratio
still scores as a pass, so even a positive E2 would not establish "better designs."
**Fix:** acknowledge as a validity gap, or add per-prompt target-response checks on a
subset (already flagged as future work tied to `circuit_optimizer`).

### 7. "Advantage grows with search space" is contradicted by the data
Paper enrichment is **retrieval-neutral (Δ≈0)** and the `PERFORMS_FUNCTION` gate is
**net-harmful** (`E1_RETRIEVAL_FINDINGS.md`; `BENCHMARK_STATUS_REPORT.md:63`), because the
corpus (Poon-group visible/NIR neuro-photonics) is domain-mismatched to the telecom
DemoPDK queries. The E1 win comes from *embeddings*, not the KG's paper content — which
undercuts the novel contribution and the "grows-with-search-space" clause directly.
**Fix:** evaluate on a domain-matched corpus/PDK, or retire the clause.

### 8. KB-build nondeterminism is unquantified in the headline
Edge-count CV 14.7%, function-propagation CV 27.4%, and **temp=0 does not remove it**
(`variance_report.py:206`; `BENCHMARK_STATUS_REPORT.md:36,101`). E1 runs once per build and
is not repeated within a build, so arm numbers carry cross-build noise never folded into a
CI; the arm ordering could flip across rebuilds.
**Fix:** report E1 arm metrics as mean ± CI across the 5 builds (data in
`results/variance/`); state which orderings are stable.

### 9. Multiple comparisons with no correction
50–100+ metric/arm/stage combinations across E1/E2/A/B with no Bonferroni/Holm/FDR in
`stats.py`. At per-test α=0.05, ~5–10 expected false positives family-wide.
**Fix:** pre-register 2–3 primary hypotheses, correct the rest, lead with effect sizes and
CIs rather than p-values.

---

## Tier 3 — Disclosure / reproducibility hygiene

- **Numbers may not reproduce from current code:** `baseline_runner.py` and
  `e2_runner.py` are modified since the last benchmark commit; `kb_stats.json` is
  stale/untracked. Re-run and commit before the defense.
- **Leakage risk:** E1/E2 prompts derive from `Testbench_modified.csv`, overlapping the
  GETTING_STARTED examples the system was built against (`e2_prompts.json:2`). Disclose the
  overlap; confirm the system was not tuned on these exact prompts.
- **Several "complete" A-series results are label-free proxies** (A1, A3) or single-
  snapshot (A6 latency; 2114-item ReviewItem backlog). Fine as diagnostics — not as
  validated quality evidence.

---

## What is defensible (for balance)

- Statistical infrastructure is real and appropriate where used: Wilson, bootstrap,
  McNemar, cluster-t (`stats.py`); ablation scripts wire p-values correctly.
- B3 topology gold explicitly avoids self-evaluation (`b3_eval.py:13`).
- Backend determinism is proven (20 passes, `run_backend_determinism.py`), correctly
  localizing E2 variance to the LLM/orchestration layer.
- Metric definitions (pass@k, MRR, set-P/R, approx-GED, gate catch/false-reject) are sound
  with no arbitrary thresholds.
- The honest status prose already admits most of Tier 1–2. The gap is that the *headline
  artifacts* don't match the prose.

---

## Priority fixes (in order)

1. Regenerate both E2 headline reports with the corrected McNemar formatter, and restore
   **N=24/K=3** as the headline (never headline N=8/K=1). Report exact p with (b, c).
2. Re-run the judge with a non-OpenAI model on a class-balanced validation set; report κ.
3. Reframe the thesis sentence to what the data supports: embedding-based selection helps
   (E1); the end-to-end and "grows-with-search-space" claims are **not yet supported** and
   must be presented as such — not as a scoring artifact.

The most dangerous combination is items 1+2: a committee that notices the headline was
switched to a smaller, unrepeated dataset that *still* shows the system losing will
question the integrity of the whole evaluation, not just E2.

---

## Appendix — evidence index

| Finding | File:line |
|---|---|
| McNemar display bug | `run_e2_headline.py:166`; `stats.py:62` (`p_value` key) |
| Correct McNemar usage elsewhere | `run_ablation_sweep.py:83`; `run_ablation_sweep_k.py:150` |
| Current headline N=8/K=1 | `results/e2_headline.md` |
| Larger run N=24/K=3 | `results/e2_headline_pre_gate_20260629.md` |
| Whole-system confound note | `results/e2_headline.md` (Fairness controls) |
| Judge model default | `judge.py:100` |
| κ=0 validation | `A2_FAITHFULNESS_FINDINGS.md:19-22` |
| E1 draft gold | `build_e1_queries_from_testbench.py:35-59`; `e1_queries.json:2` |
| E2 tier labels draft | `e2_prompts.json:2` |
| A2 N=1 / projection | `BENCHMARK_STATUS_REPORT.md:28,87` |
| Ablation underpowered | `BENCHMARK_STATUS_REPORT.md:99,103` |
| E2 validity caveat | `EVALUATION_OVERVIEW.md:98-107` |
| Enrichment neutral / domain mismatch | `E1_RETRIEVAL_FINDINGS.md`; `BENCHMARK_STATUS_REPORT.md:63` |
| KB nondeterminism | `variance_report.py:206`; `BENCHMARK_STATUS_REPORT.md:36,101` |
| Backend determinism | `run_backend_determinism.py` |

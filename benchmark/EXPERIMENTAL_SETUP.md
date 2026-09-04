# Benchmark — detailed experimental setup (non-KG / pipeline features)

> Per-benchmark protocol: **what we evaluate**, **how we evaluate it**, **how the data is
> generated/curated**, the implementing module, and current runnability. Companion to
> `KB_BENCHMARK_PLAN.md` (decisions) and `BENCHMARK_ARCHITECTURE.md` (layer design). The KG
> diagnostics (A1–A6) and the E1 *KG arm* are specified in those docs; this file covers the
> pipeline/non-KG features: E1 baseline arms, B1, B3, B4, B5, E2/B6, C.

## 0. Cross-cutting protocol (applies to every benchmark below)

- **Arms & fairness control.** Baseline = the linear pipeline; GraphRAG = the agentic
  pipeline. The **same canonical LLM is fixed across arms**, the **same component
  descriptions** (the shared `pdk_corpus`), and the **same prompts** — so the KG is the only
  variable. Model-sensitivity is future work.
- **No determinism → k-repeat variance.** There is no seed/temperature control in the
  pipeline, so every prompt is run **k times (k≈3–5)** and we report mean ± k-repeat spread,
  never a single run.
- **Complexity stratification.** Every result is broken out by **Level 1–4** tiers (defined by
  the `GETTING_STARTED` examples; Testbench prompts tagged to a tier).
- **Paired statistics.** Same prompts through both arms → **McNemar** for pass-rate/funnel
  comparisons, **bootstrap CIs** for pass@k and continuous metrics, **Wilson** for single
  proportions, **cluster-t** for the small expert anchor. (All in `stats.py`.)
- **Isolation convention.** Each stage is evaluated **twice**: once on **clean gold input**
  (isolates the stage) and once **end-to-end** (compounding, realistic).
- **Ground-truth economy.** Maximize self-labeling (DRC/sim, query-from-component, enumerable
  PDK) and synthetic injection (labels known by construction); reserve hand-curation for the
  B3 gold topologies and the gate error-taxonomies.

---

## 1. E1 — component retrieval (baseline arms; KG arm in `KB_BENCHMARK_PLAN.md`)

**What we evaluate.** Whether component selection returns the correct component(s) for a
query, and how that degrades as the candidate pool grows. (The full claim — KG beats baseline,
advantage grows with pool size — needs the KG arm; the two baseline arms are evaluable now.)

**How we evaluate.**
- **Conditions:** arms = {lexical token-overlap, LLM-over-JSON `llm_search`, (KG-grounded)} ×
  PDK sizes {50, 131, 262} × query kinds {paraphrase, functional}.
- **Procedure:** each arm returns a ranked candidate-ID list per query; score the top-k against
  the gold component(s). Same corpus/descriptions/queries across arms.
- **Metrics:** pass@1, pass@3, MRR/avg-rank, set-P/R (compositional), `retrievable_coverage@k`,
  tokens/query; broken out by query kind and pool size. **Stats:** bootstrap CI on pass@k;
  paired McNemar across arms on the same queries; scaling = trend of (KG − baseline) vs size.
- **Implements:** `e1_retrieval.py` (arms, scoring, `run_scaling`) + `pdk_corpus.py` (corpus).

**Data generation / curation.**
- **Candidate corpus (self-built):** 260 generic gdsfactory cells (distractor tier,
  *signature-grounded* deterministic descriptions) + ~27 DesignLibrary components (modeled
  tier, retrieval targets, from docstrings). One shared corpus across arms.
- **Queries (auto-generated, self-labeled):** generated **from** each modeled component, so the
  gold answer is the source component — no manual labels. *Paraphrase* ("a {name}") and
  *functional* ("a component that {function}") forms. Optional LLM-diversified paraphrases.
- **Compositional queries:** multi-component prompts (from the gold circuits / Testbench) with
  the gold **component set** as the target → set-P/R.
- **Sizes:** subsample the distractor pool to 50/131/262, always including the gold (the
  scaling curve is the distinctive result).

**Runnable now?** ✅ baseline arms (needs only an LLM key + a small runner). KG arm needs the KB.

---

## 2. B1 — interpreter / clarification

**What we evaluate.** Does the interpreter ask for clarification **iff** the prompt is
ambiguous, is the extracted intent correct, and does clarification improve end-to-end success?

**How we evaluate.**
- **Conditions:** the interpreter under test (baseline `verify_input_clarity` / GraphRAG
  `interpreter_agent`); for the ablation, **clarification on vs off** inside E2.
- **Procedure:** run the interpreter on each labeled query → record (a) clarify? decision and
  (b) extracted intent. Compute the 2×2 should-clarify confusion vs the ambiguity label, and
  field-level intent match vs gold. For the ablation, run E2 with clarification **on** (the
  `UserSimulator` answers from the hidden spec) vs **off** and compare the funnel.
- **Metrics:** should-clarify precision/recall/F1/accuracy; intent field-accuracy; E2 funnel
  Δ (ablation). **Stats:** Wilson on P/R; McNemar on the E2 ablation.
- **Implements:** `clarification_eval.py` (`should_clarify_scores`, `intent_match`,
  `UserSimulator`).

**Data generation / curation.**
- **Clear/ambiguous set (semi-automatic, labeled by construction):** start from clear Testbench
  prompts; create **ambiguous variants by ablating one key attribute** (splitting ratio, port
  count, band, bandwidth). The variant is labeled *ambiguous*; the removed information becomes
  its `hidden_spec` (drives the user simulator) and the full prompt yields the `gold_intent`.
  Seed of 6 ships in the module; target ~40–60 spanning tiers.
- **Ground truth:** ambiguity label (by construction); gold DesignIntent (from the full prompt
  / GETTING_STARTED reference).

**Runnable now?** ✅ on the baseline interpreter (LLM key, no KB). Ablation needs the E2 runner.

---

## 3. B3 — schematic builder (topology correctness)

**What we evaluate.** Whether the builder wires the **correct** topology — not merely a valid
one (validity ≠ correctness) — and whether better selection yields better wiring.

**How we evaluate.**
- **Conditions:** both arms (baseline p300 vs `schematic_builder_server`), on a curated
  gold-topology subset; **isolation** = feed gold component selection as clean input.
- **Procedure:** run the schematic stage on each prompt → predicted netlist; score vs the gold
  topology (node-id-agnostic). Separately, run **netlist validity + port-compatibility** on the
  full prompt suite.
- **Metrics:** typed-edge F1, component F1, graph-edit-distance (correctness subset);
  validity-rate + port-compat-rate (full suite). **Stats:** bootstrap CI on edge-F1, per-tier.
- **Implements:** `topology_eval.py` (parser + scorer).

**Data generation / curation.**
- **Gold topologies (curated):** the 4 `GETTING_STARTED` Level 1–4 `4_SG.txt` netlists are
  parsed automatically; **extend by hand-drawing** correct circuit graphs for ~10–15
  representative Testbench prompts into the same netlist format. This is the main hand-curation
  cost for the pipeline benchmarks.
- **Ground truth:** the curated gold topologies.

**Runnable now?** 🟡 scorer + 4 gold topologies ready; needs predicted netlists (run the
builder) + more curated golds.

---

## 4. B4 — Clingo topology gate (formal method)

**What we evaluate.** Which topology/architecture **error classes** the ASP gate catches
(per-class catch rate), its **coverage gap** (real error classes *not encoded*), false-reject
on valid designs, soundness, and its contribution to the E2 funnel.

**How we evaluate.**
- **Conditions:** the gate run on a labeled **invalid** set + a labeled **valid** control;
  plus a gate **on/off ablation** in E2.
- **Procedure:** feed each labeled `DesignIntent` to `validate_topology` → caught/not. Compute
  per-error-class catch rate, the coverage gap, false-reject on valids, and whether any *truly*
  invalid case passes (soundness). Ablation: E2 funnel (routing → sim → DRC) gate on vs off.
- **Metrics:** per-class catch rate (Wilson CI), coverage-gap list, false-reject rate,
  soundness violations, E2 funnel Δ. **Stats:** Wilson; McNemar on the funnel.
- **Implements:** `gate_eval.py` (the §6.6 template). **⚠ adapter gap:** the real gate reasons
  over **DesignIntent-level facts**, not the port-level graphs in `inject_topology.py`; the
  error taxonomy + injector must be **re-derived at the DesignIntent level from the encoded
  `.lp` rules** before it benchmarks *this* gate.

**Data generation / curation.**
- **Error taxonomy (curated from the rules):** read `clingo_validator.design_intent_to_facts`
  + the `.lp` rules to enumerate the encoded error classes; the **coverage gap** is precisely
  the real classes those rules *don't* encode.
- **Labeled set (synthetic injection):** take valid DesignIntents (from gold prompts / pipeline
  output) → inject one taxonomy-spanning error each (wrong component count, missing required
  architecture element, invalid intent-level connectivity) → labels known by construction.
  Valid controls = the unperturbed DesignIntents.

**Runnable now?** 🟡 `clingo` installed, no KB needed; methodology + scorer ready, but needs the
DesignIntent-level taxonomy/injector aligned to the rules.

---

## 5. B5 — AR parameter gate (formal method, external)

**What we evaluate.** Which **parameter-error classes** the AWS Bedrock Automated-Reasoning gate
catches, its false-reject rate, and — uniquely — its **cost / latency / availability**.

**How we evaluate.**
- **Conditions:** gate on a labeled invalid + valid parameter set; gate on/off ablation in E2.
- **Procedure:** feed labeled parameter sets to `validate_parameters` → caught/not; per-class
  catch + false-reject; **record AR cost/latency** per call; **cache** responses for
  reproducibility; run on a **representative subset** to bound cost.
- **Metrics:** per-class catch rate, false-reject, soundness; AR cost/latency/availability (an
  explicit operational finding); E2 funnel Δ. **Stats:** Wilson; McNemar.
- **Implements:** `inject_parameter.py` + `gate_eval.py` (+ an AR result-caching layer to build).

**Data generation / curation.**
- **Parameter-error taxonomy + injection (synthetic, labels by construction):** from the
  enumerable PDK parameter bounds — out-of-range, wrong-units, type-mismatch, missing-required,
  grid-violation, incompatible-pairing. Valid controls = real PDK parameter sets.

**Runnable now?** ❌ injector + scorer ready; needs AWS Bedrock credentials + the caching layer.

---

## 6. E2 / B6 — end-to-end success funnel (the headline)

**What we evaluate.** Whether the whole GraphRAG pipeline yields more
manufacturable/simulatable designs than baseline, and **where designs die**. *Caveat: passing
= manufacturable & simulatable, NOT functionally correct.*

**How we evaluate.**
- **Conditions:** both whole pipelines on the Testbench prompts (sim/DRC stages on the ~27
  modeled set), k-repeat, tier-stratified. **Ablation / arm axes (all via `e2_runner` config,
  KB-independent):** builder mode **`single_shot` vs `iterative`** (the iterative `CircuitGraph`
  builder) vs `auto`; Clingo topology gate on/off; AR gate on/off; critic rounds (2 vs 0);
  clarification on/off. Each axis is a paired arm comparison through `e2_funnel.compare_arms`.
- **Procedure:** run each prompt through each pipeline → record the staged outcomes
  `instantiate → routing_ok → models → sim_success → DRC-clean` → funnel + died-at histogram;
  paired per-stage comparison between arms.
- **Metrics:** per-stage pass rate (Wilson CI), died-at histogram, per-tier breakdown; paired
  **McNemar** + bootstrap delta-CI at each stage. **Self-labeled** (DRC + SAX).
- **Implements:** `e2_funnel.py` (scorer) + `e2_runner.py` (headless GraphRAG runner with the
  toggle config above; drives `run_pipeline_finalize` → `run_layout_simulation`). The **baseline
  linear-pipeline runner** is still to build.

**Data generation / curation.**
- **Prompts:** `Testbench.xlsx` (~100 NL design prompts), each tagged Level 1–4. **No manual
  ground truth** — DRC-clean and sim-success are self-labeling.

**Runnable now?** 🟡 The GraphRAG runner is built and **smoke-validated** end-to-end (full funnel
green on a 1×2 MMI splitter, gates off). The **ablation arms** (builder mode / gates / critic /
clarify) run on the GraphRAG pipeline now (KB-independent). Still needed: the baseline linear runner
(for baseline-vs-GraphRAG), AWS Bedrock (to exercise the AR gate), token instrumentation (infra-C).

---

## 7. C — infrastructure cost A/B

**What we evaluate.** Whether the agentic overhead is justified by the extrinsic lift: token /
latency / round cost vs the linear baseline, the human review-queue burden, and the critic's
contribution.

**How we evaluate.**
- **Conditions:** agentic vs linear run logs (collected during the E2 runs); critic on/off
  ablation.
- **Procedure:** instrument both pipelines to log tokens-in/out, latency, rounds, cap-hits, and
  review-outcome per prompt; aggregate per arm; compute ratios. Critic on/off → E2 funnel Δ +
  cost Δ.
- **Metrics:** tokens/latency/rounds/cap-hit-rate per arm + agentic/linear ratios;
  review-queue split (auto/queue/reject); critic-ablation E2 Δ. (Operational — no ground truth.)
- **Implements:** `cost_aggregator.py` + logging instrumentation in the runners.

**Data generation / curation.** Run logs emitted by the instrumented pipelines during E2.

**Runnable now?** ❌ aggregator ready; needs instrumented end-to-end runs (depends on #15).

---

## 8. Build dependencies summary

| Benchmark | Data source | Ground truth | Needs to run for real |
|---|---|---|---|
| E1 baseline | self-built corpus + auto queries | source component (self) | small runner |
| B1 | Testbench-ablated set | by construction + gold intent | interpreter adapter |
| B3 | GETTING_STARTED + curated golds | curated topologies | schematic-stage run + more golds |
| B4 | rule-derived taxonomy + injection | by construction + rules | DesignIntent-level injector |
| B5 | PDK-bound injection | by construction | AWS Bedrock + cache |
| E2/B6 | Testbench (~100) | self-labeling (DRC/sim) | headless runners (#15) |
| C | pipeline run logs | n/a | instrumented runs (#15) |

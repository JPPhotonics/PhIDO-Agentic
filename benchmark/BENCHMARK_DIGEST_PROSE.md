# PhIDO GraphRAG Benchmark — Narrative Summary

*Prepared 2026-06-30 · branch `graphRAG-implementation` · distilled from
[`BENCHMARK_STATUS_REPORT.md`](BENCHMARK_STATUS_REPORT.md).*

## Overview

The benchmark suite asks a single question from many angles: does the agentic,
knowledge-graph-backed **GraphRAG** pipeline produce better photonic designs than the
**baseline** PhIDO pipeline, and is the knowledge graph it depends on built soundly enough to
trust? Fifteen experiments answer that question across three layers — the extrinsic end-to-end
claim, the intrinsic quality of the knowledge graph, and the operational studies that tell us
whether the other numbers rest on stable ground. Every planned experiment has produced numbers,
and the determinism and variance audits needed to interpret them are complete.

The headline picture is honest and consequential. Retrieval, the component the graph was meant to
improve, genuinely works: a hybrid lexical-plus-embedding retriever reaches a pass@3 of 0.80 and
dominates every single-method baseline, and KG grounding wins specifically on design-intent
queries. But the full end-to-end pipeline currently loses. On the head-to-head funnel from prompt
to manufacturable layout, GraphRAG reaches a DRC-clean rate of just 0.25 against the baseline's
0.71, and it does so at 2.6× the latency and 4.3× the input tokens. Crucially, that loss is not
in retrieval and not in the backend — both are clean — so it concentrates in the LLM-driven agent
orchestration layer. **For the thesis, the implication is that the intrinsic and retrieval
evidence supports the GraphRAG approach, while the extrinsic end-to-end claim does not yet hold
and should be diagnosed and fixed at the orchestration layer rather than abandoned.**

## The extrinsic claim: does it build better designs?

The most positive result in the entire suite is component retrieval (**E1**). The question is
whether KG-grounded selection beats plain lexical search and an LLM reading raw component JSON.
The answer is nuanced but favourable: a hybrid retriever that fuses lexical and embedding
rankings via reciprocal-rank fusion dominates all single methods, reaching pass@3 of 0.80 on the
231-query testbench, because the methods fail in complementary ways. KG-embedding retrieval wins
specifically on *design-intent* queries (pass@1 of 0.556 versus lexical's 0.333), ties on
named-component queries, and loses to lexical on functional queries. Two design findings fell out
of this: the `PERFORMS_FUNCTION` gate is net-harmful — popular generic functions over-match, and
embedding-only ranking is strictly better — and paper enrichment turned out retrieval-neutral,
a corpus-versus-query domain mismatch. The chief caveat is that the gold labels are still draft,
pending Poon-group validation, and the functional stratum is underpowered at N=25.

The end-to-end funnel (**E2**) is the central negative result and the one to take seriously. It
runs the full pipeline — prompt, instantiation, routing, model assignment, simulation, and finally
DRC-clean layout — over 24 prompts, three times each, with majority vote. The baseline beats the
agentic pipeline at every single stage and across every complexity tier from L1 to L4: instantiate
0.96 versus 0.75, routing 0.96 versus 0.58, models and simulation 0.88 versus 0.29, and the
bottom-line DRC-clean rate 0.71 versus 0.25. Seventeen of 24 baseline runs pass all stages against
only six for GraphRAG. The agentic pipeline is also markedly less stable, with a non-unanimous
repeat rate of 0.38 to 0.67 versus the baseline's 0.12 to 0.17. One honest qualifier applies to
both arms: "passing" here means *manufacturable and simulatable*, not functionally correct, so
prompts that require unmodeled parts die legitimately at the models stage.

Two smaller extrinsic studies probe the formal topology gate. The repair-value study
(**Hard-E2**) finds that on a buildable set of eleven hard prompts the gate's repair value is
essentially zero — gate-on and gate-off both reach a 1.00 gold-match, and the gate slightly hurts
the downstream funnel — though this is a small, buildable-only subset that by construction leaves
little for repair to fix. The ablation sweep (**Ablation-k**) attributes contributions to the gate
and the critic: the gate adds about +0.15 on the models and simulation stages, but at p = 0.375
the effect is not significant, the critic's contribution is negligible, and 35 to 65 percent of
prompts flip stochastically per stage. Both arms are underpowered.

## Is the knowledge graph built correctly?

The intrinsic diagnostics examine the graph itself. Extraction and duplication (**A1**) over 414
nodes quantifies near-duplicate clusters at cosine similarity ≥ 0.80 — 26 Component clusters, 17
Property clusters, and so on — flagging semantic fragmentation, though the gold F1 and
precision-recall figures still await hand annotation, so this signal is currently label-free.

Faithfulness (**A2**) is the most developed intrinsic result and is encouraging. Asking whether
extracted relationship triples are actually supported by their source documents, the
document-intelligence-agent precision comes out at 0.675 with a Wilson confidence interval of
[0.52, 0.80]. The structure of that number matters: typed edges are faithful at 0.875, while the
generic `RELATED_TO` edge sits at only 0.54 and drags down the aggregate. Demoting `RELATED_TO`
projects precision up to 0.84, and the live KB confirms that edge type has fallen to between 3.8
and 5.4 percent of all edges. The caveats are real, however — this rests on a single annotator and
a single paper, recall is unmeasured, and the post-fix precision is projected rather than
re-annotated.

Cross-document integration (**A3**) confirms that entities genuinely span the 17-document, 367-
entity corpus rather than being siloed per paper (Architecture support 0.76, Component 0.59), with
cross-document edge recall still needing gold links. Schema convergence (**A4**) examines whether
the self-evolving schema stabilizes or keeps inventing one-off relationship types: of 60 distinct
proposed types only one, `CONTAINS_COMPONENT`, was promoted, and 43 of the 60 are confined to a
single document — mostly transient churn — though the growth curve is sensitive to document order
and order-free resampling is still pending.

PDK ingestion (**A5**) compares the ingested graph against the ground-truth DesignLibrary and
produces a split verdict that directly bounds the E1 ceiling: node precision, recall, and F1 are a
perfect 1.00, ports reach 0.97 and parameters 0.74, but the `PERFORMS_FUNCTION` edge scores only
0.27 F1 (precision 0.21, recall 0.40). That function-name score conflates true error with mere
vocabulary mismatch, since there is no synonym map yet. Integrity and latency (**A6**) over the
live graph of 2707 nodes and 3333 edges are clean — zero orphans, duplicates, self-loops, schema
violations, or missing embeddings, with query latency of 1.4 to 2.1 milliseconds median — but two
operational concerns remain: a residual 126 `RELATED_TO` edges (3.8 percent) and a backlog of 2114
ReviewItems awaiting human review. The latency figure is a single snapshot, not a scaling curve.

Two more intrinsic checks round out the picture. Reference-free health (**L1**) shows the graph
carries recoverable structural regularity even without a gold reference: link-prediction MRR is
0.175 on the clean graph versus 0.066 on a corrupted control, a +0.110 separation that reflects
consistency rather than correctness. Clarification behaviour (**B1**) finds the pipeline never
misses a genuinely ambiguous prompt (recall 1.0) but over-clarifies clear ones (precision 0.5);
at N=6 with a single labeler, this arm needs a rubric, a second annotator for a κ score, and a
larger set.

## Do the numbers stand on solid ground?

The operational studies exist to make the results above interpretable, and two of them are what
let us localize the E2 loss. The topology-gate discrimination study (**Gate**) shows the formal
gate is sound on its encoded rules: across eight valid and fifteen invalid circuits it catches
100 percent of all fifteen error classes with a 0.00 false-reject rate, the caveat being that
soundness is only over the rules actually encoded. Backend determinism (**Backend**) shows the
non-LLM layout, simulation, and DRC backend is fully deterministic across 20 passes of a frozen
netlist at a mean 2.32 seconds — which is the linchpin that pins the E2 and ablation variance
upstream in LLM netlist generation, not in the backend.

That upstream variance is itself measured by the KB-rebuild campaign (**Variance**). Across five
identical rebuilds, edge count varies with a coefficient of variation of 14.7 percent and
literature-propagated functions at 27.4 percent; setting temperature to zero does *not* eliminate
this build non-determinism, although embedding-grounded retrieval stays byte-identical. The
implication is operationally important: per-build KG retrieval numbers carry cross-run noise until
this non-determinism is tamed.

## Where this leaves us

The priorities follow directly from the evidence. First and most valuable is to diagnose the E2
agentic regression: with the backend clean and retrieval working, the loss must lie in netlist
generation and agent orchestration. Second, expert gold validation is the gating dependency for
several "complete" intrinsic results that are currently label-free proxies — A1 and A3 recall, A2
faithfulness beyond a single annotator, and the E1 testbench gold all wait on Poon-group
annotation. Third, the operational debt should be cleared: 2114 pending ReviewItems and the 126
residual `RELATED_TO` edges flagged in A6. Fourth, the underpowered arms — B1 at N=6, the E1
functional stratum at N=25, and the ablation whose McNemar test sits at p ≥ 0.375 — need larger N
and seeded multi-build runs. Finally, the KB build non-determinism (edge CV 14.7 percent) must be
brought under control before any per-build KG retrieval number can be treated as stable.

---

*Underlying data: `benchmark/results/`. Findings docs: `A2_FAITHFULNESS_FINDINGS.md`,
`E1_RETRIEVAL_FINDINGS.md`, `VARIANCE_FINDINGS.md`, `EXPERIMENTAL_SETUP.md`.*

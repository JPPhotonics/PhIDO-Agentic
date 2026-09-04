# E2 ablation grid — design

Reframes the two-arm E2 funnel (rigid `baseline` vs full `graphrag`) into a **feature
ablation** that attributes the end-to-end funnel outcome to each individual "new" agentic
feature, rather than to the agentic architecture as a monolith.

## Features (each a `RunConfig` axis)

| Feature | ON | OFF | Toggle(s) |
| --- | --- | --- | --- |
| **kg** | KG grounding + hybrid RRF retrieval | no grounding, lexical retrieval | `grounding_rounds=5`+`retrieval_backend="hybrid"` vs `0`+`"lexical"` |
| ~~**routing**~~ | ~~iterative tool-calling builder~~ | — | **DROPPED — see note below** |
| **gate** | Clingo topology gate | off | `enable_topology_gate` |
| **critic** | critic review loop (≤2 rounds) | off | `max_critic_rounds=2` vs `0` |

**Routing (iterative extraction) is DROPPED and treated as unimplemented.** Screening showed
iterative routing was the only feature that *degraded* correctness (edgeF1 0.667→0.563 additive,
and the all-on stack collapsed to 0.507), reproducing an earlier independent funnel/drc_clean
result. For our purposes it was never a real feature, so its bit is held **0 across every arm**:
there are no `*_routing` arms, and `full` = kg+gate+critic (== the app default, formerly
`full_minus_routing`). The grid drops from 10 → 8 agentic arms.

**KG is a two-knob feature.** The pipeline uses the KG at two points: (1) grounding rounds
during `explore_and_ask` (KG tool-calls while exploring requirements), and (2) the Phase-4
component-retrieval backend (`RETRIEVAL_BACKEND`, hybrid RRF vs lexical). "KG off" toggles
**both** so it is unambiguous regardless of whether Neo4j is live.

**AR gate is held OFF for every arm.** The AWS-Bedrock automated-reasoning parameter gate
(`enable_ar_gate`) is a no-op without Bedrock credentials, so `full−AR` / `base+AR` cannot
produce signal here. The slots are retained in the design and marked *pending Bedrock*.

## Entanglement fix (prerequisite for an orthogonal grid)

> **Now moot:** routing is dropped, so the routing×critic entanglement below no longer affects
> any live arm. Retained for the record.

Before this work, `extraction_mode="iterative"` **bypassed the critic** ("no critic for
iterative builds", `pipeline_orchestrator.py`), so the routing and critic axes were not
independent: any iterative arm had critic implicitly off, which would have made `full` and
`full−critic` identical. Fixed by adding a `max_critic_rounds`-gated `run_critic_review`
call to the iterative branch, mirroring the single-shot ordering (topology gate → critic →
selection). A critic *pass* leaves the iterative-built DesignIntent untouched; a critic
*repair* re-extracts a tool-grounded intent (dropping iterative pre-grounding), after which
component selection falls back to the LLM selector.

## Arms (feature bits = kg · routing · gate · critic; routing held 0 — dropped)

| Arm | bits | Role |
| --- | --- | --- |
| `rigid_baseline` | — | main-branch rigid pipeline (LLM top-1 docstring search, no orchestrator) |
| `base_agentic` | 0000 | orchestrator + MCP floor — all features off |
| `full` | 1011 | all features on (routing excluded) — **≡ the app default / previously-tested "full graphrag"** (Δ=−0.12 drc_clean) |
| `full_minus_kg` | 0011 | leave-one-out (marginal contribution **in context**) |
| `full_minus_gate` | 1001 | LOO |
| `full_minus_critic` | 1010 | LOO |
| `base_plus_kg` | 1000 | additive (**standalone** contribution over the floor) |
| `base_plus_gate` | 0010 | additive |
| `base_plus_critic` | 0001 | additive |

*Dropped:* all `*_routing` arms (`full` already excludes routing). *Deferred (need Bedrock):*
`full_minus_AR`, `base_plus_AR`.

The `base+X` arms have Hamming weight 1 (over the live axes) and the `full−X` arms weight 2, so
all 8 agentic configs are distinct. `full` (kg+gate+critic) is the app default and recovers the
earlier single-shot "full graphrag", giving continuity with the completed E2 testbench run.

## Reference points

- **Additive marginal** of feature X: `base+X − base_agentic` — X's standalone lift over the minimal agentic floor.
- **Leave-one-out marginal** of feature X: `full − full∖X` — X's contribution in the presence of the other three.
- **Anchors:** `full − rigid_baseline`, `base_agentic − rigid_baseline`, `full − base_agentic`.

A feature whose additive Δ is large but LOO Δ is ~0 is *redundant at the top* (its job is done
by the others); the reverse indicates a feature that *only helps once the rest are present*.

## Scale

- **Pilot:** K=1 × N=103 testbench prompts × 11 runnable arms = 1,133 pipeline executions, o3-mini.
- Then commit **K=3** to the informative arms (majority-vote over repeats absorbs the no-seed variance).

## Files

- `run_e2_ablation.py` — driver (arm matrix, incremental JSON, additive/LOO report). Env: `K_REPEATS`, `E2_MODEL`, `RIGID_MODEL`, `N_PROMPTS`, `ARMS_ONLY`, `SKIP_RIGID`, `DRY`.
- `e2_runner.py` — `RunConfig` extended with `retrieval_backend` + `grounding_rounds`; `run_one` sets/restores `RETRIEVAL_BACKEND` per arm and threads grounding into explore.
- `results/e2_ablation.json` + `.md` — raw + human-readable report.

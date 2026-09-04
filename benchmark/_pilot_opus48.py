"""Pilot: agentic arm on claude-opus-4-8 via Claude Code OAuth, edges+DOT saved, HCOLLAPSE scoring.

Purpose (per plan step C): smoke-test the full agentic pipeline on Opus 4.8 + Claude Code auth +
Opus-4.8 params, AND capture REAL agentic pred_edges to calibrate the edge-verified collapse engine
(the o1 3-cond run stored no edges). Prints raw wiring per rep so uncovered wirings (e.g. tap-style
heaters) are visible. Scoped tiny by default (subscription tier throttles). Env:
  PILOT_PROMPTS   comma-sep gold ids (default "L3_1")
  PILOT_ARM       run_agentic | run_agentic_nogate (default run_agentic_nogate)
  E2_K            reps per prompt (default 1)
Requires: PHIDO_CLAUDE_CODE_AUTH=1, E2_MODEL=claude-opus-4-8, PHIDO_HCOLLAPSE=1 (set by launcher).
"""
import json
import os

import _score_correctness as S
from mcp_servers import llm_client

# Opus 4.8 list price ($/token): $5 / MTok input, $25 / MTok output.
OPUS48_IN, OPUS48_OUT = 5.0 / 1e6, 25.0 / 1e6

PROMPTS = os.getenv("PILOT_PROMPTS", "L3_1").split(",")
ARM = os.getenv("PILOT_ARM", "run_agentic_nogate")
K = int(os.getenv("E2_K", "1"))
fn = getattr(S, ARM)
llm_client.USAGE.update({"in": 0, "out": 0, "calls": 0})  # reset the process-global tally

print(f"PILOT arm={ARM} model={S.MODEL} prompts={PROMPTS} K={K} "
      f"HCOLLAPSE={os.getenv('PHIDO_HCOLLAPSE')} CC_AUTH={os.getenv('PHIDO_CLAUDE_CODE_AUTH')}",
      flush=True)

for pid in PROMPTS:
    for k in range(K):
        r = S.rep(fn, S.prompts[pid]["prompt"], pid)
        print(f"\n=== {pid} rep{k+1}/{K} ===", flush=True)
        print(f"  status={r['status']} compF1={r.get('compF1')} edgeF1={r.get('edgeF1')} "
              f"ged={r.get('ged')} hcollapsed={r.get('hcollapsed')}", flush=True)
        print(f"  RAW node_map={r.get('pred_node_map')}", flush=True)
        print(f"  RAW edges={json.dumps(r.get('pred_edges'))}", flush=True)
        print(f"  scored pred_nodes(post-collapse)={r.get('pred_nodes')}", flush=True)

# --- Token usage + Opus 4.8 cost projection --------------------------------------------------
u = dict(llm_client.USAGE)
n_runs = max(1, len(PROMPTS) * K)
per_in, per_out = u["in"] / n_runs, u["out"] / n_runs
cost_per_run = per_in * OPUS48_IN + per_out * OPUS48_OUT
print("\n=== USAGE / COST ESTIMATE ===", flush=True)
print(f"  measured on E2_MODEL={S.MODEL} (token VOLUME; applied to Opus 4.8 LIST price)", flush=True)
print(f"  totals: calls={u['calls']} in={u['in']:,} out={u['out']:,} over {n_runs} run(s)", flush=True)
print(f"  per run: in={per_in:,.0f} out={per_out:,.0f} -> Opus4.8 ${cost_per_run:.3f}/run", flush=True)
for label, runs in [("6 L3 x K3 x 1 arm", 18), ("24 x K3 x 1 arm", 72),
                    ("24 x K3 x 3 arms", 216)]:
    print(f"  proj[{label}] = {runs} runs -> ~${cost_per_run * runs:.2f}", flush=True)
print("  CAVEATS: token counts measured on the run model (o1); Opus 4.8 tokenizes differently and "
      "adaptive thinking changes output volume -> ballpark, not exact. No prompt caching modeled "
      "(caching would cut input cost substantially on repeated PDK/KG context).", flush=True)
print("\nPILOT DONE", flush=True)

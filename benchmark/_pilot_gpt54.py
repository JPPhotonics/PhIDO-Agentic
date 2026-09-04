"""Pilot: agentic arm on gpt-5.4 (OpenAI), edges+DOT saved, HCOLLAPSE scoring.

Purpose: (1) smoke-test the full agentic pipeline end-to-end on gpt-5.4 — the raw-endpoint probe
confirmed the API surface (accepts temperature, rejects max_tokens which we never send, structured
parse works), but not the full tool-use loop; (2) capture REAL agentic pred_edges on gpt-5.4 to
calibrate the edge-verified collapse engine on L3_2..L3_6 (only L3_1 was calibrated, on the Opus
pilot). Prints raw wiring per rep so uncovered wirings (e.g. tap-style heaters) are visible.

gpt-5.4 uses the always-required OPENAI_API_KEY — no Claude Code OAuth, no Max-5x quota. Env:
  PILOT_PROMPTS   comma-sep gold ids (default the 6 L3: L3_1..L3_6)
  PILOT_ARM       run_agentic | run_agentic_nogate (default run_agentic_nogate)
  E2_K            reps per prompt (default 1)
  PILOT_PRICE_IN / PILOT_PRICE_OUT   optional $/MTok; if BOTH set, prints a cost projection
Requires (set by launcher): E2_MODEL=gpt-5.4, PHIDO_HCOLLAPSE=1.
"""
import json
import os

import _score_correctness as S
from mcp_servers import llm_client

PROMPTS = os.getenv("PILOT_PROMPTS", "L3_1,L3_2,L3_3,L3_4,L3_5,L3_6").split(",")
ARM = os.getenv("PILOT_ARM", "run_agentic_nogate")
K = int(os.getenv("E2_K", "1"))
fn = getattr(S, ARM)
llm_client.USAGE.update({"in": 0, "out": 0, "calls": 0})  # reset the process-global tally

print(f"PILOT arm={ARM} model={S.MODEL} prompts={PROMPTS} K={K} "
      f"HCOLLAPSE={os.getenv('PHIDO_HCOLLAPSE')}", flush=True)

for pid in PROMPTS:
    for k in range(K):
        r = S.rep(fn, S.prompts[pid]["prompt"], pid)
        print(f"\n=== {pid} rep{k+1}/{K} ===", flush=True)
        print(f"  status={r['status']} compF1={r.get('compF1')} edgeF1={r.get('edgeF1')} "
              f"ged={r.get('ged')} hcollapsed={r.get('hcollapsed')}", flush=True)
        print(f"  RAW node_map={r.get('pred_node_map')}", flush=True)
        print(f"  RAW edges={json.dumps(r.get('pred_edges'))}", flush=True)
        print(f"  scored pred_nodes(post-collapse)={r.get('pred_nodes')}", flush=True)

# --- Token usage (+ optional cost projection if a price is supplied) -------------------------
u = dict(llm_client.USAGE)
n_runs = max(1, len(PROMPTS) * K)
per_in, per_out = u["in"] / n_runs, u["out"] / n_runs
print("\n=== USAGE ===", flush=True)
print(f"  measured on E2_MODEL={S.MODEL}", flush=True)
print(f"  totals: calls={u['calls']} in={u['in']:,} out={u['out']:,} over {n_runs} run(s)", flush=True)
print(f"  per run: in={per_in:,.0f} out={per_out:,.0f}", flush=True)
pin, pout = os.getenv("PILOT_PRICE_IN"), os.getenv("PILOT_PRICE_OUT")
if pin and pout:
    cin, cout = float(pin) / 1e6, float(pout) / 1e6
    cost_per_run = per_in * cin + per_out * cout
    print(f"  price ${pin}/${pout} per MTok -> ${cost_per_run:.3f}/run", flush=True)
    for label, runs in [("6 L3 x K3 x 1 arm", 18), ("24 x K3 x 1 arm", 72),
                        ("24 x K3 x 3 arms", 216)]:
        print(f"  proj[{label}] = {runs} runs -> ~${cost_per_run * runs:.2f}", flush=True)
else:
    print("  (gpt-5.4 list price not supplied via PILOT_PRICE_IN/OUT -> no cost projection)", flush=True)
print("\nPILOT DONE", flush=True)

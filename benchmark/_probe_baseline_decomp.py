"""Compare baseline decomposition/stability vs the agentic arm on the same prompts.
For each prompt, run the rigid baseline N times and report the selected component set +
funnel outcome. Answers: does the baseline over-decompose / unmap / vary like agentic?"""
import os

import baseline_runner as B

MODEL = os.getenv("E2_MODEL", "o3-mini")
N = int(os.getenv("PROBE_N", "3"))
PROMPTS = {
    "L3_2": "Design a 2x2 MZI with two 2 mm long PIN-diodes, one in each arm, for phase shifting",
    "L2_2": "2 grating couplers connected by a straight waveguide",
    "L2_5": "Connect one output port of a 1x2 MMI to an input of another 1x2 MMI",
}

for pid, prompt in PROMPTS.items():
    print(f"===== {pid}: {prompt[:60]} =====", flush=True)
    for i in range(N):
        try:
            dsl = B.build_rigid_netlist(prompt, MODEL)
            comps = [v.get("component") for v in dsl.get("nodes", {}).values()]
            res = B.run_one(prompt, prompt_id=pid, level="?", model=MODEL)
            print(f"  run {i}: built={comps}  stages={res['stages']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  run {i}: EXC {type(e).__name__}: {str(e)[:120]}", flush=True)

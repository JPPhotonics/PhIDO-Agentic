"""Probe the routing_ok failures: run a multi-component prompt a few times and, per run,
dump the exported netlist's instances / routes(links) / ports plus the routing error, to
show WHY gdsfactory routing fails (phantom/invalid links vs unmapped cascade). Scratch."""
import os
import sys
import yaml
from mcp_servers.pipeline_orchestrator import (
    explore_and_ask, run_pipeline_finalize, run_layout_simulation)

MODEL = os.getenv("E2_MODEL", "o3-mini")
PROMPT = os.getenv("PROBE_PROMPT", "Design a 2x2 MZI with two 2 mm long PIN-diodes, one in each arm, for phase shifting")
N = int(os.getenv("PROBE_N", "3"))

from mcp_servers.pdk_whitelist import simulatable_modules
print(f"WARMUP {len(simulatable_modules())} modules", flush=True)

for i in range(N):
    print(f"===== run {i} =====", flush=True)
    try:
        est = None
        for ev in explore_and_ask(PROMPT, MODEL, 15, 5):
            if ev.get("type") == "clarification":
                est = ev
        pdone = None
        for ev in run_pipeline_finalize(explore_state=est, user_prompt=PROMPT, model=MODEL,
                                        max_critic_rounds=1, enable_topology_gate=True, enable_ar_gate=False):
            if ev.get("type") == "pipeline_done":
                pdone = ev["result"]
        net = (pdone or {}).get("gf_netlist_yaml")
        if not net:
            print("  no netlist", flush=True); continue
        n = yaml.safe_load(net)
        insts = {k: v.get("component") for k, v in (n.get("instances") or {}).items()}
        links = {rn: rd.get("links") for rn, rd in (n.get("routes") or {}).items()}
        print(f"  instances: {insts}", flush=True)
        print(f"  route_links: {links}", flush=True)
        print(f"  ports: {n.get('ports')}", flush=True)
        rok = None
        for ev in run_layout_simulation(net):
            if ev.get("type") == "layout_sim_done":
                rok = ev["result"]
        print(f"  routing_ok={rok.get('routing_ok')} routing_error={str(rok.get('routing_error'))[:140]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  EXC {type(e).__name__}: {str(e)[:160]}", flush=True)

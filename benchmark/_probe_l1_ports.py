"""Probe: run L1_1 (90-deg bend) N times through the agentic finalize path and, for each,
report the funnel stages AND the top-level `ports:` block of the exported netlist. Ties the
intermittent `models` failure to empty circuit ports. Diagnostic scratch, not committed."""
import os
import yaml
from mcp_servers.pipeline_orchestrator import explore_and_ask, run_pipeline_finalize, run_layout_simulation

MODEL = os.getenv("E2_MODEL", "o3-mini")
PROMPT = "A 90 degree waveguide bend"
N = int(os.getenv("PROBE_N", "4"))

# Warm the whitelist cache once (noisily builds 3 known-broken cells) so per-run timing is clean.
from mcp_servers.pdk_whitelist import simulatable_modules
print(f"WARMUP simulatable_modules -> {len(simulatable_modules())} modules", flush=True)

for i in range(N):
    print(f"START run {i}", flush=True)
    try:
        explore_state = None
        for ev in explore_and_ask(PROMPT, MODEL, 15, 5):
            if ev.get("type") == "clarification":
                explore_state = ev
        pdone = None
        for ev in run_pipeline_finalize(explore_state=explore_state, user_prompt=PROMPT, model=MODEL,
                                        max_critic_rounds=1, enable_topology_gate=True, enable_ar_gate=False):
            if ev.get("type") == "pipeline_done":
                pdone = ev["result"]
        netlist = pdone.get("gf_netlist_yaml") if pdone else None
        ports = None
        if netlist:
            try:
                ports = (yaml.safe_load(netlist) or {}).get("ports", "MISSING_KEY")
            except Exception as e:  # noqa: BLE001
                ports = f"PARSE_ERR:{e}"
        ls = None
        if netlist:
            for ev in run_layout_simulation(netlist):
                if ev.get("type") == "layout_sim_done":
                    ls = ev["result"]
        miss = (ls or {}).get("missing_models")
        print(f"[run {i}] netlist={'yes' if netlist else 'NONE'} ports={ports!r} "
              f"routing_ok={(ls or {}).get('routing_ok')} missing_models={miss}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[run {i}] EXC {type(e).__name__}: {str(e)[:160]}", flush=True)

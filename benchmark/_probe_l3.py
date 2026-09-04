"""Diagnose the L3 correctness collapse: capture agentic DesignIntent + final netlist
instances for the MZI_2x2-gold prompts (L3_1/L3_2/L3_3). Writes incrementally so a
reaped run still leaves partial results. Detached-friendly. Scratch."""
import json
import re

import yaml
from mcp_servers.pipeline_orchestrator import explore_and_ask, run_pipeline_finalize

OUT = "results/_probe_l3.json"
P = {p["id"]: p for p in json.load(open("e2_prompts.json"))["prompts"]}
TARGETS = ["L3_1", "L3_2", "L3_3", "L3_4"]
out = json.load(open(OUT)) if __import__("os").path.exists(OUT) else {}

for pid in TARGETS:
    if pid in out:
        continue
    prompt = P[pid]["prompt"]
    est = None
    for ev in explore_and_ask(prompt, "o3-mini", 15, 5):
        if ev.get("type") == "clarification":
            est = ev
    last_di = None
    done = None
    vf = None
    for ev in run_pipeline_finalize(explore_state=est, user_prompt=prompt, model="o3-mini",
                                    max_critic_rounds=1, enable_topology_gate=True,
                                    enable_ar_gate=False):
        t = ev.get("type")
        if t == "done" and isinstance(ev.get("result"), str) and "components=" in ev["result"]:
            last_di = ev["result"]
        if t == "pipeline_done":
            done = ev["result"]
        if t == "validation_failed":
            vf = ev.get("gate")
    rec = {"prompt": prompt}
    if last_di:
        rec["intent_ids"] = re.findall(r"id='([^']+)'", last_di)
        rec["intent_types"] = re.findall(r"component_type='([^']*)'", last_di)
        rec["pdk_modules"] = re.findall(r"pdk_module='([^']*)'", last_di)
        rec["di_summary"] = last_di[:600]
    if done:
        n = yaml.safe_load(done.get("gf_netlist_yaml") or "") or {}
        rec["instances"] = {k: v.get("component") for k, v in (n.get("instances") or {}).items()}
        rec["result"] = "pipeline_done"
    else:
        rec["result"] = f"validation_failed:{vf}" if vf else "no_pipeline_done"
    out[pid] = rec
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[{pid}] {rec['result']} instances={rec.get('instances')}", flush=True)
print("DONE", flush=True)

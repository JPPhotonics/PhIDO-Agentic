"""Smoke test for the three fixes before the 103-prompt run:
- whitelist exposes passive primitives (L1_5 agentic should build _mmi1x2, not empty)
- baseline apply_settings ScannerError no longer crashes (L4_1 polarization splitter)
- o3-mini 400 retry / general robustness (no uncaught crashes)
Runs both arms on a few prompts; records netlist instances or a CLEAN failure (not a crash).
Writes incrementally. Scratch."""
import json
import os
import traceback

import yaml

MODEL = "o3-mini"
OUT = "results/_smoke_fixes.json"
P = {p["id"]: p for p in json.load(open("e2_prompts.json"))["prompts"]}
CASES = ["L1_5", "L4_1", "L3_1"]
out = json.load(open(OUT)) if os.path.exists(OUT) else {}


def agentic(prompt):
    from mcp_servers.pipeline_orchestrator import explore_and_ask, run_pipeline_finalize
    est = None
    for ev in explore_and_ask(prompt, MODEL, 15, 5):
        if ev.get("type") == "clarification":
            est = ev
    done = None
    vf = None
    for ev in run_pipeline_finalize(explore_state=est, user_prompt=prompt, model=MODEL,
                                    max_critic_rounds=1, enable_topology_gate=True,
                                    enable_ar_gate=False):
        if ev.get("type") == "pipeline_done":
            done = ev["result"]
        if ev.get("type") == "validation_failed":
            vf = ev.get("gate")
    if done:
        n = yaml.safe_load(done.get("gf_netlist_yaml") or "") or {}
        return {"result": "netlist", "instances": list((n.get("instances") or {}).values())}
    return {"result": f"clean_fail:{vf}"}


def baseline(prompt):
    import baseline_runner as B
    dsl = B.build_rigid_netlist(prompt, MODEL)
    return {"result": "netlist", "instances": [v.get("component") for v in (dsl.get("nodes") or {}).values()]}


for pid in CASES:
    for arm, fn in (("baseline", baseline), ("agentic", agentic)):
        key = f"{arm}/{pid}"
        if key in out:
            continue
        try:
            out[key] = fn(P[pid]["prompt"])
        except Exception as e:  # noqa: BLE001
            out[key] = {"result": "CRASH", "exc": f"{type(e).__name__}: {str(e)[:120]}",
                        "tb": traceback.format_exc()[-400:]}
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"[{key}] {out[key]['result']} {out[key].get('instances') or out[key].get('exc','')}", flush=True)
print("DONE", flush=True)

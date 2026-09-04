"""Run ONE agentic prompt in a fresh interpreter and append its funnel result to a shared
JSON. Fresh-process-per-run avoids the cumulative-memory death that truncates in-process
batches. Diagnostic scratch, not a committed artifact.  Usage: _run_single.py <prompt_id>"""
import json
import os
import sys

import e2_runner as R

pid = sys.argv[1]
OUT = "results/_repro_n8_fresh.json"
prompts = {p["id"]: p for p in json.load(open("e2_prompts.json"))["prompts"]}
p = prompts[pid]
cfg = R.RunConfig(model=os.getenv("E2_MODEL", "o3-mini"),
                  enable_ar_gate=False, enable_topology_gate=True, max_critic_rounds=1)
try:
    res = R.run_one(p["prompt"], prompt_id=pid, level=p.get("level", "?"), config=cfg)
    rec = {"stages": res["stages"], "diag": res["diag"]}
except Exception as e:  # noqa: BLE001
    rec = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
data = json.load(open(OUT)) if os.path.exists(OUT) else {}
data.setdefault(pid, []).append(rec)
json.dump(data, open(OUT, "w"), indent=1)
print(f"[{pid}] {rec.get('stages') or rec.get('error')}", flush=True)

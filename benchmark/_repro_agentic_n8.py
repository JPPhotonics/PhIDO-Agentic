"""Ad-hoc: re-run the agentic arm on the N=8 headline prompts, K repeats, capture
stages+diag to characterize CURRENT failure modes vs the committed 2026-06-30 run.
Not a committed benchmark artifact — a diagnostic scratch script."""
import json
import os

import e2_runner as R

K = int(os.getenv("REPRO_K", "2"))
_default = "L1_1,L1_4,L2_2,L2_5,L3_2,L3_5,L4_3,L4_6"
IDS = set(os.getenv("REPRO_IDS", _default).split(","))
prompts = [p for p in json.load(open("e2_prompts.json"))["prompts"] if p["id"] in IDS]
cfg = R.RunConfig(model=os.getenv("E2_MODEL", "o3-mini"),
                  enable_ar_gate=False, enable_topology_gate=True, max_critic_rounds=1)

# Warm the whitelist cache once (noisily builds the 3 known-broken cells) before the loop.
from mcp_servers.pdk_whitelist import simulatable_modules
print(f"WARMUP simulatable_modules -> {len(simulatable_modules())} modules", flush=True)

OUT_PATH = "results/_repro_agentic_n8.json"
out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}  # merge across batches
for p in prompts:
    runs = []
    for k in range(K):
        try:
            res = R.run_one(p["prompt"], prompt_id=p["id"], level=p.get("level", "?"), config=cfg)
            runs.append({"stages": res["stages"], "diag": res["diag"],
                         "failed_at": res["cost"].get("failed_at")})
        except Exception as e:  # noqa: BLE001
            runs.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        s = runs[-1].get("stages") or runs[-1].get("error")
        print(f"[{p['id']} k{k}] {s}", flush=True)
        out[p["id"]] = runs
        json.dump(out, open(OUT_PATH, "w"), indent=1)  # incremental: survive a mid-batch death
print("DONE ->", OUT_PATH, flush=True)

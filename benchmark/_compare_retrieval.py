"""Iterative-vs-single-shot / KG-vs-lexical comparison on the N=8 E2 prompts.

Conditions (agentic arm only, K=1 each unless CMP_K set):
  A single_shot_hybrid  — extraction_mode=single_shot, RETRIEVAL_BACKEND=hybrid  (E2 default: KG forced)
  B single_shot_lexical — extraction_mode=single_shot, RETRIEVAL_BACKEND=lexical (KG off)
  C iterative           — extraction_mode=iterative  (agent CHOOSES search_pdk vs KG tools)

Resumable: skips (condition, prompt, rep) already present in the output JSON, so re-launching
after a truncation continues where it left off. Scratch, not a committed artifact.
"""
import json
import os

import e2_runner as R

MODEL = os.getenv("E2_MODEL", "o3-mini")
K = int(os.getenv("CMP_K", "1"))
OUT = "results/_compare_retrieval.json"
IDS = ["L1_1", "L1_4", "L2_2", "L2_5", "L3_2", "L3_5", "L4_3", "L4_6"]
prompts = {p["id"]: p for p in json.load(open("e2_prompts.json"))["prompts"]}

CONDITIONS = {
    "single_shot_hybrid":  ("single_shot", "hybrid"),
    "single_shot_lexical": ("single_shot", "lexical"),
    "iterative":           ("iterative",   "hybrid"),
}

# Warm whitelist once (noisy broken-cell builds) so per-run output is clean.
from mcp_servers.pdk_whitelist import simulatable_modules
print(f"WARMUP {len(simulatable_modules())} modules", flush=True)

out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for cond, (mode, backend) in CONDITIONS.items():
    os.environ["RETRIEVAL_BACKEND"] = backend
    out.setdefault(cond, {})
    cfg = R.RunConfig(model=MODEL, enable_ar_gate=False, enable_topology_gate=True,
                      max_critic_rounds=1, extraction_mode=mode)
    for pid in IDS:
        runs = out[cond].setdefault(pid, [])
        while len(runs) < K:
            try:
                res = R.run_one(prompts[pid]["prompt"], prompt_id=pid,
                                level=prompts[pid].get("level", "?"), config=cfg)
                rec = {"stages": res["stages"], "diag": res["diag"]}
            except Exception as e:  # noqa: BLE001
                rec = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            runs.append(rec)
            json.dump(out, open(OUT, "w"), indent=1)  # incremental -> resumable
            print(f"[{cond}/{pid}] {rec.get('stages') or rec.get('error')}", flush=True)
print("DONE", flush=True)

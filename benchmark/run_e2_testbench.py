"""E2 FUNNEL run over the ~103 Testbench prompts (no gold topology exists for these, so
this scores only the intrinsic funnel: instantiate -> routing -> models -> sim -> DRC).
Design-CORRECTNESS is left to MANUAL marking from the saved artifacts (set E2_OUTPUT_DIR).

Both arms share the model (o3-mini), DemoPDK, prompts, and the layout/sim/DRC tail — same
harness as run_e2_headline, but:
  - prompts = benchmark/e2_testbench_prompts.json (generated from results/e1_testbench_extraction.json)
  - RESUMABLE: skips (arm, prompt_id) pairs already at K reps in the output JSON, so a run
    killed by the environment's long-job reaper is continued by simply re-invoking.
  - K defaults to 1 (scan); raise via K_REPEATS.
"""
import json
import os
import time
from pathlib import Path

import baseline_runner as B
import e2_funnel as F
import e2_runner as R
from report import Reporter

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.getenv("E2_MODEL", "o3-mini")
K = int(os.getenv("K_REPEATS", "1"))
SRC = ROOT / "benchmark" / "results" / "e1_testbench_extraction.json"
PROMPTS_FILE = ROOT / "benchmark" / "e2_testbench_prompts.json"
OUT = ROOT / "benchmark" / "results" / "e2_testbench.json"
OUT.parent.mkdir(exist_ok=True)


def build_prompts_file():
    """Generate id/level/prompt records from the testbench extraction (idempotent).

    `level` is a rough complexity proxy = #components (capped 1..4) purely for
    stratified reporting; these prompts have no authored tier.
    """
    if PROMPTS_FILE.exists():
        return
    rows = json.loads(SRC.read_text())
    prompts = []
    for r in rows:
        ncomp = len(r.get("components", []) or [])
        prompts.append({
            "id": f"TB{r['prompt_index']:03d}",
            "level": max(1, min(4, ncomp)),
            "prompt": r["prompt"],
        })
    PROMPTS_FILE.write_text(json.dumps({"prompts": prompts}, indent=1))


def run_arm(label, runfn, prompts, store):
    store.setdefault(label, {})
    for p in prompts:
        store[label].setdefault(p["id"], [])
        while len(store[label][p["id"]]) < K:
            rep = len(store[label][p["id"]])
            t = time.time()
            try:
                r = runfn(p)
            except Exception as e:  # noqa: BLE001 — keep the batch alive
                r = {"prompt_id": p["id"], "level": p["level"], "arm": label,
                     "stages": dict.fromkeys(R.STAGES, False),
                     "cost": {"latency_s": time.time() - t, "failed_at": f"exc:{type(e).__name__}"},
                     "error": str(e)[:200]}
            r["level"] = p["level"]
            store[label][p["id"]].append(r)
            json.dump(store, open(OUT, "w"), indent=2, default=str)
            print(f"[{label:9}] {p['id']:6} L{p['level']} rep{rep + 1}/{K} "
                  f"{r['stages']} {r['cost'].get('failed_at')} {time.time() - t:.0f}s", flush=True)


def _majority(runs):
    norm = [F.normalize(r["stages"]) for r in runs]
    return {"prompt_id": runs[0]["prompt_id"], "level": runs[0].get("level", "?"),
            "stages": {s: sum(n[s] for n in norm) > len(norm) / 2 for s in F.STAGES}}


def main():
    build_prompts_file()
    prompts = json.loads(PROMPTS_FILE.read_text())["prompts"]
    store = json.loads(OUT.read_text()) if OUT.exists() else {}

    run_arm("baseline", lambda p: B.run_one(p["prompt"], prompt_id=p["id"], level=p["level"], model=MODEL),
            prompts, store)
    orch = R.RealOrchestrator()
    cfg = R.RunConfig(model=MODEL, enable_ar_gate=False, enable_topology_gate=True, max_critic_rounds=2)
    run_arm("graphrag", lambda p: R.run_one(p["prompt"], prompt_id=p["id"], level=p["level"],
                                            arm="graphrag", config=cfg, orch=orch),
            prompts, store)

    base_maj = [_majority(store["baseline"][pid]) for pid in store["baseline"] if store["baseline"][pid]]
    gr_maj = [_majority(store["graphrag"][pid]) for pid in store["graphrag"] if store["graphrag"][pid]]
    rep = Reporter(OUT.with_suffix(".md"), "E2 funnel — Testbench (~103 prompts, no gold; correctness = manual)",
                   meta={"model": MODEL, "N_prompts": len(prompts), "K_repeats": K,
                         "arms": "baseline (rigid) vs graphrag (agentic)", "raw_json": OUT.name,
                         "artifacts": os.getenv("E2_OUTPUT_DIR", "(E2_OUTPUT_DIR unset)")})
    rep.h("Funnel — per-stage pass rate (per-prompt majority over K)")
    base_sum, gr_sum = F.funnel_summary(base_maj), F.funnel_summary(gr_maj)
    rep.table(["arm", "n"] + list(F.STAGES),
              [["baseline", base_sum["n"]] + [f"{base_sum['by_stage'][s]['rate']:.2f}" for s in F.STAGES],
               ["graphrag", gr_sum["n"]] + [f"{gr_sum['by_stage'][s]['rate']:.2f}" for s in F.STAGES]])
    rep.h("Died-at histogram")
    rep.table(["arm"] + [f"died_{s}" for s in F.STAGES] + ["passed_all"],
              [["baseline"] + [base_sum["died_at"].get(f"died_at_{s}", 0) for s in F.STAGES]
               + [base_sum["died_at"].get("passed_all", 0)],
               ["graphrag"] + [gr_sum["died_at"].get(f"died_at_{s}", 0) for s in F.STAGES]
               + [gr_sum["died_at"].get("passed_all", 0)]])
    rep.h("Paired comparison (Δ = graphrag − baseline; McNemar p)")
    cmp = F.compare_arms(base_maj, gr_maj)
    rep.line(f"_n_paired = {cmp['n_paired']}_")
    rep.line("")
    rep.table(["stage", "baseline", "graphrag", "Δ", "McNemar p"],
              [[s, f"{cmp['by_stage'][s]['baseline_rate']:.2f}", f"{cmp['by_stage'][s]['graphrag_rate']:.2f}",
                f"{cmp['by_stage'][s]['delta']:+.2f}", f"{cmp['by_stage'][s]['mcnemar'].get('p_value', float('nan')):.3f}"]
               for s in F.STAGES])
    rep.save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

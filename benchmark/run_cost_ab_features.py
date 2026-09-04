"""C — cost A/B (feature-resolved): rigid baseline + agentic additive arms, uniformly metered.

Extends ``run_cost_ab.py`` from a single agentic config to the ADDITIVE ablation family, so the
token / latency / rounds cost is attributed PER FEATURE and lines up 1:1 with the human-review
quality finding (``base_plus_X − base_agentic``, §8.6). Every arm runs on ONE model
(``E2_MODEL``, default ``gpt-5.4``) through the same ``TokenMeter``, so token counts compare
apples-to-apples and the model price is a single multiplier; holding the model fixed isolates
ARCHITECTURAL cost from model choice (stated caveat: the thesis quality baseline used o1). The
full pipeline runs through layout + DRC, supplying the tokens-per-DRC-clean denominator.

Arms:    rigid_baseline + base_agentic + base_plus_{kg,gate,critic}   (additive family only).
Prompts: the graded §8.6 set (``b3_gold_v2.json``), PER_LEVEL size-spanning prompts per L3/L4
         group (min / median / max node count), so cost pairs with the graded quality per arm.
Reps:    K_REPEATS (default 2). Resumable: a re-run reuses the store and tops each pair up to K.

Run:
  E2_MODEL=gpt-5.4 K_REPEATS=2 N_PER_LEVEL=3 \
  [PILOT_PRICE_IN=<$/MTok> PILOT_PRICE_OUT=<$/MTok>] \
  PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_cost_ab_features.py
DRY=1 prints the plan (arms, selected prompts, run count) and exits.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import defaultdict

import baseline_runner as BL
import cost_aggregator as C
import e2_runner as R
from dotenv import load_dotenv
from report import Reporter
from run_e2_ablation import ARMS, ROOT, cfg_for

load_dotenv(str(ROOT / ".env"))

MODEL = os.getenv("E2_MODEL", "gpt-5.4")
K = int(os.getenv("K_REPEATS", "2"))
PER_LEVEL = int(os.getenv("N_PER_LEVEL", "3"))
PRICE_IN = os.getenv("PILOT_PRICE_IN")   # $/MTok, optional
PRICE_OUT = os.getenv("PILOT_PRICE_OUT")  # $/MTok, optional
GOLD = ROOT / "benchmark" / "b3_gold_v2.json"
OUT = ROOT / "benchmark" / "results" / "cost_ab_features.json"

# Additive family: baseline (rigid) as the anchor, base_agentic as the minimal agentic scaffold,
# then each single feature added on top so its marginal cost is base_plus_X − base_agentic.
AGENTIC_ARMS = ["base_agentic", "base_plus_kg", "base_plus_gate", "base_plus_critic"]


def load_prompts(per_level: int) -> list[dict]:
    """Size-spanning subset of the graded gold: per L3/L4 group, the min / median / max node-count
    prompts (evenly-spaced indices over the node-count-sorted group). Deterministic."""
    gold = json.loads(GOLD.read_text())["gold"]
    by: dict[str, list] = defaultdict(list)
    for e in gold:
        by[str(e["id"]).split("_")[0]].append(e)
    out: list[dict] = []
    for grp_key in sorted(by):
        grp = sorted(by[grp_key], key=lambda e: len(e.get("nodes") or []))
        m = len(grp)
        k = min(per_level, m)
        if k <= 0:
            continue
        idx = [0] if k == 1 else sorted({round(i * (m - 1) / (k - 1)) for i in range(k)})
        for j in idx:
            e = grp[j]
            out.append({"id": e["id"], "prompt": e["prompt"], "level": grp_key,
                        "n_nodes": len(e.get("nodes") or [])})
    return out


def _fail(pid, level, arm, t, exc):
    return {"prompt_id": pid, "level": level, "arm": arm, "stages": {},
            "cost": {"tokens_in": 0, "tokens_out": 0, "latency_s": time.time() - t,
                     "rounds": 0, "failed_at": "exception", "error": str(exc)[:300]}}


def run_pair(arm, prompt, *, rigid, cfg, orch, store):
    pid, level = prompt["id"], prompt["level"]
    done = len(store.get(arm, {}).get(pid, []))
    for rep in range(done, K):  # resumable: top up to K
        t = time.time()
        try:
            if rigid:
                r = BL.run_one(prompt["prompt"], prompt_id=pid, level=level, model=MODEL)
            else:
                r = R.run_one(prompt["prompt"], prompt_id=pid, level=level,
                              arm=arm, config=cfg, orch=orch)
        except Exception as e:  # noqa: BLE001 — keep the batch alive
            r = _fail(pid, level, arm, t, e)
        r.setdefault("cost", {}).setdefault("latency_s", time.time() - t)
        store.setdefault(arm, {}).setdefault(pid, []).append(r)
        json.dump(store, open(OUT, "w"), indent=2, default=str)
        c = r.get("cost", {})
        print(f"[{arm:17}] {pid:6} rep{rep + 1}/{K} "
              f"{c.get('tokens_in', 0)}+{c.get('tokens_out', 0)}tok "
              f"drc={r.get('stages', {}).get('drc_clean')} {time.time() - t:.0f}s", flush=True)


def main() -> None:
    prompts = load_prompts(PER_LEVEL)
    all_arms = ["rigid_baseline"] + AGENTIC_ARMS
    n_runs = len(all_arms) * len(prompts) * K
    if os.getenv("DRY") == "1":
        print(f"[DRY] model={MODEL} K={K} arms={all_arms}")
        print(f"      prompts ({len(prompts)}): "
              + ", ".join(f"{p['id']}(n={p['n_nodes']})" for p in prompts))
        print(f"      total runs = {n_runs}  (full pipeline through layout+DRC)")
        for a in AGENTIC_ARMS:
            print(f"        {a:17} bits(kg,routing,gate,critic)={ARMS[a]}")
        return

    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    orch = R.RealOrchestrator()
    t0 = time.time()
    for p in prompts:
        run_pair("rigid_baseline", p, rigid=True, cfg=None, orch=None, store=store)
        for arm in AGENTIC_ARMS:
            cfg = dataclasses.replace(cfg_for(ARMS[arm]), model=MODEL)
            run_pair(arm, p, rigid=False, cfg=cfg, orch=orch, store=store)
    print(f"\nall runs done in {time.time() - t0:.0f}s", flush=True)

    # ── aggregate + report ────────────────────────────────────────────
    agg = {}
    for arm, byp in store.items():
        logs = [C.to_cost_log(r) for runs in byp.values() for r in runs]
        agg[arm] = C.aggregate(logs)

    price = None
    if PRICE_IN and PRICE_OUT:
        pin, pout = float(PRICE_IN), float(PRICE_OUT)
        price = lambda a: (a["tokens_in_total"] * pin + a["tokens_out_total"] * pout) / 1e6 / max(a["n"], 1)

    # Layout routing is broken in this worktree (gdsfactory↔kfactory `route_bundle(starts=...)`
    # mismatch) → DRC-clean is uniformly 0, same as the netlist-level §8.6 run. When so, drop the
    # DRC-clean denominator and rely on cost/run (paired with the graded §8.6 quality per arm).
    drc_ok = sum(a.get("n_drc_clean", 0) for a in agg.values()) > 0

    rep = Reporter(
        OUT.with_suffix(".md"),
        "C — cost A/B, feature-resolved (rigid baseline + additive agentic arms)",
        meta={"model": MODEL, "K_repeats": K, "prompts": [p["id"] for p in prompts],
              "metered": "TokenMeter (same instrument, all arms, one model)",
              "pipeline": "full attempted; layout routing unavailable in this worktree → netlist-level"
                          if not drc_ok else "full (through layout+DRC)",
              "cost_effectiveness": "tokens/DRC-clean" if drc_ok
                          else "DRC-clean=0 (routing lib mismatch) → pair cost/run with §8.6 human quality per arm",
              "raw_json": OUT.name,
              "price_$/MTok": f"in={PRICE_IN} out={PRICE_OUT}" if price else "not supplied (token volume only)"},
    )
    rep.h("Per-arm cost")
    hdr = ["arm", "runs", "tok in/run", "tok out/run", "tok/run", "instantiated", "latency s", "rounds"]
    if drc_ok:
        hdr[6:6] = ["DRC-clean", "tok/DRC-clean"]
    if price:
        hdr.insert(5, "$/run")
    rows = []
    for arm in ["rigid_baseline"] + AGENTIC_ARMS:
        a = agg.get(arm)
        if not a or a.get("n", 0) == 0:
            continue
        n_inst = sum(1 for runs in store.get(arm, {}).values() for r in runs
                     if r.get("stages", {}).get("instantiate"))
        row = [arm, a["n"], round(a["tokens_in_total"] / a["n"]),
               round(a["tokens_out_total"] / a["n"]), round(a["tokens_per_run_mean"]),
               f"{n_inst}/{a['n']}", round(a["latency_s_mean"], 1), round(a["rounds_mean"], 2)]
        if drc_ok:
            row[6:6] = [f"{a['n_drc_clean']}/{a['n']}",
                        "∞" if a["tokens_per_success"] == float("inf") else round(a["tokens_per_success"])]
        if price:
            row.insert(5, f"{price(a):.4f}")
        rows.append(row)
    rep.table(hdr, rows)

    base = agg.get("base_agentic")
    rigid = agg.get("rigid_baseline")
    if base and rigid and rigid["n"]:
        rep.h("Headline — base_agentic ÷ rigid_baseline")
        dims = [("tokens/run", "tokens_per_run_mean")]
        if drc_ok:
            dims.append(("tokens/DRC-clean", "tokens_per_success"))
        rep.table(["dimension", "rigid", "base_agentic", "ratio"],
                  [[d, round(rigid[k]), round(base[k]),
                    f"{base[k] / rigid[k]:.2f}×" if rigid[k] else "∞"]
                   for d, k in dims])
    if base:
        rep.h("Per-feature marginal cost — base_plus_X ÷ base_agentic")
        mrows = []
        for arm in ["base_plus_kg", "base_plus_gate", "base_plus_critic"]:
            a = agg.get(arm)
            if not a:
                continue
            d = a["tokens_per_run_mean"] - base["tokens_per_run_mean"]
            r = a["tokens_per_run_mean"] / base["tokens_per_run_mean"] if base["tokens_per_run_mean"] else float("inf")
            mrows.append([arm.replace("base_plus_", "+"), round(base["tokens_per_run_mean"]),
                          round(a["tokens_per_run_mean"]), f"{d:+.0f}", f"{r:.2f}×",
                          round(a["latency_s_mean"], 1), round(a["rounds_mean"], 2)])
        rep.table(["feature", "base tok/run", "arm tok/run", "Δ tok/run", "ratio", "latency s", "rounds"], mrows)
    rep.line("")
    rep.line("_Token counts are apples-to-apples (one TokenMeter, one model). Model held fixed at "
             f"{MODEL} across both architectures to isolate architectural cost from model price; "
             "the §8.6 quality baseline used o1 (noted). Pair each feature's marginal cost with its "
             "≈0 quality delta from the human review (§8.6) for the cost-vs-benefit claim._")
    rep.save()
    print(f"wrote {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()

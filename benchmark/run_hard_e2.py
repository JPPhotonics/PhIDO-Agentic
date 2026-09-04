"""Harder-E2 ablation: does the topology gate's repair loop make a measurable difference?

For each hard prompt (`e2_hard_prompts.py`) we run the pipeline twice — gate ON vs gate OFF —
capture the FINAL DesignIntent each arm produces, and score it with ONE independent check
applied to both arms:

  * clingo_clean : the final intent satisfies architecture_rules.lp (no fundamental error).
                   gate-OFF = the LLM's raw topology-error rate on hard prompts;
                   gate-ON  = the rate AFTER the gate's feedback-driven re-extraction.
                   The paired delta = the gate's repair value (the thing the neutral end-to-end
                   ablation could not surface).
  * gold_match   : extracted clingo-type counts == the prompt's annotated gold_counts
                   (an arm-independent correctness check that does NOT rely on the gate's rules).

Scoring runs `validate_topology` POST-HOC on both arms' final intent, so the gate-OFF arm is
scored even though its pipeline never ran the gate — no circularity. Buildable prompts also go
through layout/sim/DRC for a secondary end-to-end view. Reports paired McNemar on clingo_clean
and gold_match, plus repair-round usage. Incremental-saved.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_hard_e2.py
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import e2_funnel as F
import e2_hard_prompts as HP
import e2_runner as R
from report import Reporter
from run_pilot_e2 import ROOT, NoGroundOrch
from stats import paired_mcnemar, wilson

from mcp_servers.clingo_validator import _resolve_component_type, validate_topology

MODEL = os.getenv("PILOT_MODEL", "o3-mini")
K = int(os.getenv("K_REPEATS", "1"))
OUT = ROOT / "benchmark" / "results" / "hard_e2.json"


def _counts(di) -> dict:
    return dict(Counter(_resolve_component_type(c) for c in di.components))


def _score_intent(di, gold: dict) -> dict:
    """Independent, arm-agnostic scoring of a final DesignIntent."""
    fb = validate_topology(di)
    fundamental = [f for f in fb if f.severity == "fundamental"]
    ct = _counts(di)
    return {
        "clingo_clean": len(fundamental) == 0,
        "gold_match": all(ct.get(k, 0) == v for k, v in gold.items()),
        "arch_detected": di.architecture_type,
        "n_detected": di.n_value,
        "counts": ct,
        "fired_codes": [f.context.get("clingo_error_code") for f in fb],
    }


def run_arm(p: dict, gate_on: bool, orch, model: str) -> dict:
    t0 = time.perf_counter()
    explore_state = None
    for ev in orch.explore(p["prompt"], model):
        if ev.get("type") == "clarification":
            explore_state = ev

    final_intent, pipeline_done, rounds = None, None, 0
    if explore_state is not None:
        for ev in orch.finalize(
            explore_state,
            p["prompt"],
            model,
            clarifications=None,
            max_critic_rounds=0,
            enable_topology_gate=gate_on,
            enable_ar_gate=False,
            extraction_mode="single_shot",
        ):
            if ev.get("phase") == "validation_retry":
                rounds += 1
            if ev.get("type") == "done" and hasattr(ev.get("result"), "components"):
                final_intent = ev["result"]
            if ev.get("type") == "pipeline_done":
                pipeline_done = ev["result"]

    rec: dict = {
        "id": p["id"],
        "gate_on": gate_on,
        "rounds": rounds,
        "latency_s": round(time.perf_counter() - t0, 1),
    }
    if final_intent is None:
        rec.update(
            {
                "clingo_clean": False,
                "gold_match": False,
                "arch_detected": None,
                "n_detected": None,
                "counts": {},
                "fired_codes": ["NO_INTENT"],
            }
        )
    else:
        rec.update(_score_intent(final_intent, p["gold_counts"]))

    # secondary: end-to-end funnel for buildable architectures
    if p["buildable"]:
        layout_sim = None
        if pipeline_done and pipeline_done.get("gf_netlist_yaml"):
            for ev in orch.layout_sim(pipeline_done["gf_netlist_yaml"]):
                if ev.get("type") == "layout_sim_done":
                    layout_sim = ev["result"]
        rec["stages"] = R.funnel_from_results(pipeline_done, layout_sim)
    return rec


def _rate(vec) -> tuple[float, tuple]:
    k, n = sum(vec), len(vec)
    return (k / n if n else float("nan")), wilson(k, n) if n else (float("nan"),) * 2


def main() -> None:
    prompts = HP.prompts()
    if os.getenv("DRY") == "1":
        print(
            f"[DRY] {len(prompts)} hard prompts x 2 arms x K={K} = {len(prompts) * 2 * K} runs; model={MODEL}"
        )
        for p in prompts:
            print(
                f"  {p['id']:12} {p['architecture']:14} gold={p['gold_counts']} buildable={p['buildable']}"
            )
        return

    orch = NoGroundOrch()
    store: dict = {"gate_on": {}, "gate_off": {}}
    for gate_on in (True, False):
        arm = "gate_on" if gate_on else "gate_off"
        for p in prompts:
            for rep in range(K):
                rec = run_arm(p, gate_on, orch, MODEL)
                store[arm].setdefault(p["id"], []).append(rec)
                json.dump(store, open(OUT, "w"), indent=2, default=str)
                print(
                    f"[{arm:8}] {p['id']:12} rep{rep + 1}/{K} "
                    f"clingo_clean={rec['clingo_clean']} gold={rec['gold_match']} "
                    f"arch={rec['arch_detected']} rounds={rec['rounds']} "
                    f"{rec.get('fired_codes')}",
                    flush=True,
                )

    # majority over K repeats per prompt, paired across arms
    def maj(arm, pid, key):
        v = [r[key] for r in store[arm][pid]]
        return sum(v) > len(v) / 2

    ids = list(store["gate_on"])
    rep = Reporter(
        OUT.with_suffix(".md"),
        "Harder-E2 topology-gate repair value",
        meta={
            "model": MODEL,
            "prompts": len(prompts),
            "K_repeats": K,
            "arms": "gate_on vs gate_off",
            "raw_json": OUT.name,
        },
    )

    rep.h("Per-prompt majority outcomes (gate ON | OFF)")
    rep.table(
        [
            "prompt",
            "arch",
            "gold",
            "ON clingo",
            "ON gold",
            "OFF clingo",
            "OFF gold",
            "ON rounds",
        ],
        [
            [
                i,
                store["gate_on"][i][0]["arch_detected"],
                next(p["gold_counts"] for p in prompts if p["id"] == i),
                "✓" if maj("gate_on", i, "clingo_clean") else "✗",
                "✓" if maj("gate_on", i, "gold_match") else "✗",
                "✓" if maj("gate_off", i, "clingo_clean") else "✗",
                "✓" if maj("gate_off", i, "gold_match") else "✗",
                max(r["rounds"] for r in store["gate_on"][i]),
            ]
            for i in ids
        ],
    )

    rep.h("Repair value (final-intent conformance, ON − OFF, paired McNemar)")
    rows = []
    for key in ("clingo_clean", "gold_match"):
        on = [1 if maj("gate_on", i, key) else 0 for i in ids]
        off = [1 if maj("gate_off", i, key) else 0 for i in ids]
        on_r, on_ci = _rate(on)
        off_r, off_ci = _rate(off)
        mc = paired_mcnemar(on, off)
        rows.append(
            [
                key,
                f"{on_r:.2f} {tuple(round(x, 2) for x in on_ci)}",
                f"{off_r:.2f} {tuple(round(x, 2) for x in off_ci)}",
                f"{on_r - off_r:+.2f}",
                f"{mc['p_value']:.3f}",
                f"b={mc['b']},c={mc['c']}",
            ]
        )
    rep.table(
        ["metric", "ON (CI95)", "OFF (CI95)", "Δ", "McNemar p", "discordant"], rows
    )

    repaired = [
        i
        for i in ids
        if not maj("gate_off", i, "clingo_clean") and maj("gate_on", i, "clingo_clean")
    ]
    broke = [
        i
        for i in ids
        if maj("gate_off", i, "clingo_clean") and not maj("gate_on", i, "clingo_clean")
    ]
    rounds_used = [r["rounds"] for runs in store["gate_on"].values() for r in runs]
    rep.h("Repair direction")
    rep.line(
        f"- prompts the gate REPAIRED (off-fail → on-clean): {repaired or '(none)'}"
    )
    rep.line(f"- prompts the gate BROKE (off-clean → on-fail): {broke or '(none)'}")
    rep.line(
        f"- gate-on repair rounds: mean={sum(rounds_used) / len(rounds_used):.2f}, "
        f"max={max(rounds_used)}"
    )

    build_ids = [p["id"] for p in prompts if p["buildable"]]
    if build_ids:
        rep.h("Secondary: end-to-end funnel on buildable prompts")
        frows = []
        for arm in ("gate_on", "gate_off"):
            flat = [r for i in build_ids for r in store[arm][i] if "stages" in r]
            by = F.funnel_summary(flat)["by_stage"]
            frows.append([arm] + [round(by[s]["rate"], 2) for s in F.STAGES])
        rep.table(["arm"] + list(F.STAGES), frows)

    rep.line(f"\n[ok] raw -> {OUT}")
    rep.save()


if __name__ == "__main__":
    main()

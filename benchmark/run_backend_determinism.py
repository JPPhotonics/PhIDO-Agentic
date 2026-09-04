"""Backend-determinism check: is the layout/sim/DRC tail deterministic given a FROZEN netlist?

The E2 funnel's five stages (instantiate → routing_ok → models → sim_success → drc_clean) all
run downstream of the netlist, but the netlist itself is the stochastic per-run output of the LLM
pipeline. This isolates the two: generate ONE netlist once, freeze it to disk, then run the
backend (`run_layout_simulation`) on that exact frozen netlist N times. If the five stages are
identical across all N, the backend is deterministic and the run-to-run flip-rate seen in the
ablation is attributable to NETLIST GENERATION, not the backend.

The frozen netlist is cached at results/frozen_netlist.yaml — generation (one LLM pipeline run)
happens only if that file is absent, so repeat invocations are instant and fully reproducible.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_backend_determinism.py
"""

from __future__ import annotations

import os
import time

import e2_hard_prompts as HP
import e2_runner as R
from report import Reporter
from run_pilot_e2 import ROOT, NoGroundOrch

N = int(os.getenv("N_REPEATS", "20"))
MODEL = os.getenv("PILOT_MODEL", "o3-mini")
GEN_PROMPT_ID = os.getenv("GEN_PROMPT_ID", "h_tree8")  # a simple buildable prompt
FROZEN = ROOT / "benchmark" / "results" / "frozen_netlist.yaml"
OUT_MD = ROOT / "benchmark" / "results" / "backend_determinism.md"


def _generate_one_netlist(orch, prompt: str) -> str | None:
    """Run the LLM pipeline ONCE to produce a netlist YAML (the frozen reference)."""
    explore_state = None
    for ev in orch.explore(prompt, MODEL):
        if ev.get("type") == "clarification":
            explore_state = ev
    if explore_state is None:
        return None
    for ev in orch.finalize(
        explore_state,
        prompt,
        MODEL,
        clarifications=None,
        max_critic_rounds=0,
        enable_topology_gate=True,
        enable_ar_gate=False,
        extraction_mode="single_shot",
    ):
        if ev.get("type") == "pipeline_done":
            return (ev["result"] or {}).get("gf_netlist_yaml")
    return None


def _backend_once(netlist_yaml: str) -> tuple[dict, float]:
    """Run only the layout/sim/DRC tail on a fixed netlist; return (stages, latency)."""
    from mcp_servers.pipeline_orchestrator import run_layout_simulation

    t0 = time.perf_counter()
    layout_sim = None
    for ev in run_layout_simulation(netlist_yaml):
        if ev.get("type") == "layout_sim_done":
            layout_sim = ev["result"]
    stages = R.funnel_from_results({"gf_netlist_yaml": netlist_yaml}, layout_sim)
    return stages, time.perf_counter() - t0


def main() -> None:
    orch = NoGroundOrch()

    if FROZEN.exists():
        netlist_yaml = FROZEN.read_text()
        src = f"cached {FROZEN.name}"
    else:
        prompt = next(p["prompt"] for p in HP.HARD_PROMPTS if p["id"] == GEN_PROMPT_ID)
        print(
            f"[gen] no frozen netlist — generating once from {GEN_PROMPT_ID} ...",
            flush=True,
        )
        netlist_yaml = _generate_one_netlist(orch, prompt)
        if not netlist_yaml:
            print("[err] pipeline did not produce a netlist; aborting.")
            return
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(netlist_yaml)
        src = f"generated from {GEN_PROMPT_ID}, saved to {FROZEN.name}"

    # time one pass first so we can report wall-clock honestly
    first_stages, first_dt = _backend_once(netlist_yaml)
    print(
        f"[time] one backend pass = {first_dt:.1f}s; estimated total for N={N}: "
        f"~{first_dt * N:.0f}s",
        flush=True,
    )

    runs = [first_stages]
    lat = [first_dt]
    for i in range(1, N):
        st, dt = _backend_once(netlist_yaml)
        runs.append(st)
        lat.append(dt)
        print(f"  pass {i + 1}/{N}: {st} {dt:.1f}s", flush=True)

    stages_list = list(R.STAGES)
    flips = {s: len({r[s] for r in runs}) > 1 for s in stages_list}
    all_identical = not any(flips.values())

    rep = Reporter(
        OUT_MD,
        "Backend determinism on a frozen netlist",
        meta={
            "netlist": src,
            "model_for_gen": MODEL,
            "N_backend_passes": N,
            "mean_latency_s": round(sum(lat) / len(lat), 2),
        },
    )
    rep.h("Verdict")
    rep.line(
        f"**Backend is {'DETERMINISTIC' if all_identical else 'NON-DETERMINISTIC'}** "
        f"across {N} passes on the identical frozen netlist."
    )
    rep.line("")
    rep.line(
        "- If deterministic: the ablation flip-rate (35–65% of prompts) is attributable to "
        "**netlist generation** (the stochastic LLM pipeline), not the layout/sim/DRC backend."
    )

    rep.h("Per-stage stability")
    rep.table(
        ["stage", "value (all passes)", "flipped?"],
        [
            [s, sorted({str(r[s]) for r in runs}), "YES" if flips[s] else "no"]
            for s in stages_list
        ],
    )

    rep.h("Reference netlist stages")
    rep.line(f"`{first_stages}`")
    rep.save()


if __name__ == "__main__":
    main()

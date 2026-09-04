"""E2 / ablation — headless GraphRAG runner with toggles.

Drives the agentic pipeline non-interactively for one prompt and emits the E2 funnel
stages (``instantiate → routing_ok → models → sim_success → drc_clean``) that
``e2_funnel`` consumes — toggling the non-KG components so the ablations (B1 clarification,
B4 Clingo gate, B5 AR gate, infra-C critic) reduce to "run with the toggle on vs off".

Drive sequence (mirrors ``run_pipeline`` but with full toggle control):
  explore_and_ask → run_pipeline_finalize(gates/critic/clarify toggles) → run_layout_simulation

The orchestrator is injected (``orch``) so the funnel-extraction + toggle wiring are
unit-testable with mock generators; the default ``RealOrchestrator`` calls the live
pipeline (needs an LLM key + the synced env). Each run records latency, retry rounds, where
it died, and OpenAI token usage (input/output) via ``token_meter.TokenMeter`` (infra-C).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from token_meter import TokenMeter

STAGES = ["instantiate", "routing_ok", "models", "sim_success", "drc_clean"]


def save_run_outputs(arm, prompt_id, prompt, pipeline_done, layout_sim, stages, diag) -> None:
    """Opt-in (env ``E2_OUTPUT_DIR``): persist per-run artifacts for MANUAL correctness eval.

    The E2 funnel only scores *manufacturable + simulatable*, NOT functional correctness. To
    judge correctness offline a human needs the actual design keyed to the prompt: the netlist
    YAML, the rendered GDS path, and a layout preview PNG. No-ops unless ``E2_OUTPUT_DIR`` is
    set, so default benchmark runs (and their numbers) are untouched. Shared by both arms.
    """
    out_dir = os.getenv("E2_OUTPUT_DIR")
    if not out_dir:
        return
    import base64
    import json

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{arm}__{prompt_id}__{time.time_ns()}")
    ls = layout_sim or {}
    meta = {"arm": arm, "prompt_id": prompt_id, "prompt": prompt, "stages": stages,
            "diag": diag, "gds_file_path": ls.get("gds_file_path"),
            "drc_clean": ls.get("drc_clean"), "sim_error": ls.get("sim_error")}
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    netlist = (pipeline_done or {}).get("gf_netlist_yaml")
    if netlist:
        with open(stem + ".netlist.yml", "w") as f:
            f.write(netlist)
    b64 = ls.get("gds_fig_b64")
    if b64:
        try:
            with open(stem + ".layout.png", "wb") as f:
                f.write(base64.b64decode(b64))
        except Exception:  # noqa: BLE001 — a preview is best-effort, never fail the run
            pass


@dataclass
class RunConfig:
    model: str = "o3-mini"
    enable_topology_gate: bool = True  # B4 ablation toggle
    enable_ar_gate: bool = True  # B5 ablation toggle
    max_critic_rounds: int = 2  # infra-C critic ablation (0 = off)
    clarify: bool = False  # B1 ablation: answer clarifications via user_simulator
    extraction_mode: str = (
        "single_shot"  # routing ablation: "single_shot" | "iterative" | "auto"
    )
    # KG ablation — toggled as a PAIR so "KG off" is unambiguous regardless of Neo4j
    # state: grounding_rounds=0 (no KG tool-calls during explore) AND lexical retrieval
    # backend (no KG in Phase-4 component selection). "KG on" = grounding + hybrid RRF.
    retrieval_backend: str = "hybrid"  # "hybrid" (KG+lexical RRF) | "lexical" | "kg"
    grounding_rounds: int = 5  # explore-phase KG grounding rounds (0 = off)


def funnel_from_results(pipeline_done: dict | None, layout_sim: dict | None) -> dict:
    """Map the pipeline outputs onto the five funnel stages (raw; e2_funnel normalizes)."""
    stages = dict.fromkeys(STAGES, False)
    if not pipeline_done or not pipeline_done.get("gf_netlist_yaml") or not layout_sim:
        return stages  # never produced a buildable/instantiated netlist
    stages["instantiate"] = True  # GDS rendered (layout_sim_done was emitted)
    stages["routing_ok"] = bool(layout_sim.get("routing_ok"))
    stages["models"] = not layout_sim.get("missing_models")
    stages["sim_success"] = not layout_sim.get("sim_error")
    stages["drc_clean"] = layout_sim.get("drc_clean") is True
    return stages


class RealOrchestrator:
    """Adapter to the live pipeline (lazy imports; needs LLM key + synced env)."""

    def explore(self, prompt: str, model: str, grounding_rounds: int = 5):
        from mcp_servers.pipeline_orchestrator import explore_and_ask

        return explore_and_ask(prompt, model, 15, grounding_rounds)

    def finalize(self, explore_state, prompt, model, **kw):
        from mcp_servers.pipeline_orchestrator import run_pipeline_finalize

        return run_pipeline_finalize(
            explore_state=explore_state, user_prompt=prompt, model=model, **kw
        )

    def layout_sim(self, netlist: str):
        from mcp_servers.pipeline_orchestrator import run_layout_simulation

        return run_layout_simulation(netlist)


def run_one(
    prompt: str,
    prompt_id: str = "p",
    level: str = "?",
    arm: str = "graphrag",
    config: RunConfig | None = None,
    orch=None,
    user_simulator=None,
) -> dict:
    """Run one prompt through the (toggled) pipeline; return a funnel result for e2_funnel."""
    config = config or RunConfig()
    orch = orch or RealOrchestrator()
    t0 = time.perf_counter()
    rounds = 0

    # KG-retrieval backend is read at call-time via os.getenv in component_retrieval;
    # set it per-arm and restore afterwards so arms run in one process without leaking.
    _prev_backend = os.environ.get("RETRIEVAL_BACKEND")
    os.environ["RETRIEVAL_BACKEND"] = config.retrieval_backend

    meter = TokenMeter()
    meter.__enter__()  # try/finally (not `with`) so a pipeline exception still restores the SDK patch
    try:
        # Phase 1: explore + (optional) clarification capture
        explore_state = None
        for ev in orch.explore(prompt, config.model, config.grounding_rounds):
            if ev.get("type") == "clarification":
                explore_state = ev

        clarifications = None
        if config.clarify and user_simulator is not None and explore_state:
            questions = explore_state.get("questions") or {}
            items = (
                questions.items()
                if isinstance(questions, dict)
                else enumerate(questions)
            )
            clarifications = {
                str(k): user_simulator.answer(str(v)) for k, v in items
            } or None

        # Phase 2+: finalize with the ablation toggles
        pipeline_done, failed_at, last_phase, last_error = None, None, "extraction", None
        if explore_state is not None:
            for ev in orch.finalize(
                explore_state,
                prompt,
                config.model,
                clarifications=clarifications,
                max_critic_rounds=config.max_critic_rounds,
                enable_topology_gate=config.enable_topology_gate,
                enable_ar_gate=config.enable_ar_gate,
                extraction_mode=config.extraction_mode,
            ):
                t = ev.get("type")
                if ev.get("phase"):
                    last_phase = ev["phase"]  # remember where we are for silent-stop attribution
                if ev.get("phase") == "validation_retry":
                    rounds += 1
                if t == "pipeline_done":
                    pipeline_done = ev["result"]
                elif t in ("validation_failed", "error"):
                    failed_at = t
                    last_error = ev.get("message") or ev.get("detail")
            # finalize ended without producing a netlist AND without an explicit error event:
            # attribute the silent stop to the last phase reached (was previously recorded as None)
            if pipeline_done is None and failed_at is None:
                failed_at = f"finalize_incomplete:{last_phase}"
        else:
            failed_at = "explore"

        # Phase 8: layout + sim + DRC
        layout_sim = None
        if pipeline_done and pipeline_done.get("gf_netlist_yaml"):
            for ev in orch.layout_sim(pipeline_done["gf_netlist_yaml"]):
                if ev.get("type") == "layout_sim_done":
                    layout_sim = ev["result"]
    finally:
        meter.__exit__()
        if _prev_backend is None:
            os.environ.pop("RETRIEVAL_BACKEND", None)
        else:
            os.environ["RETRIEVAL_BACKEND"] = _prev_backend
    tok = meter.snapshot()

    stages = funnel_from_results(pipeline_done, layout_sim)
    diag = _diagnostics(pipeline_done, layout_sim, last_error)
    save_run_outputs(arm, prompt_id, prompt, pipeline_done, layout_sim, stages, diag)

    return {
        "prompt_id": prompt_id,
        "level": level,
        "arm": arm,
        "stages": stages,
        "config": vars(config),
        "diag": diag,
        "cost": {
            "latency_s": time.perf_counter() - t0,
            "rounds": rounds,
            "failed_at": failed_at,
            "tokens_in": tok["tokens_in"],
            "tokens_out": tok["tokens_out"],
        },
    }


def _diagnostics(pipeline_done: dict | None, layout_sim: dict | None, last_error) -> dict:
    """Capture WHY a run lands where it does: selected modules, the modules actually built
    into the netlist, and the SAX/DRC failure payloads. Without this the funnel is just
    booleans and ``died_at_models`` is unattributable (the gap that hid the real E2 cause)."""
    diag: dict = {"last_error": (str(last_error)[:300] if last_error else None)}
    if pipeline_done:
        sel = pipeline_done.get("selection") or {}
        diag["selected_modules"] = [
            m.get("pdk_module") for m in (sel.get("mappings") or []) if m.get("pdk_module")
        ]
        diag["unmapped"] = sel.get("unmapped") or []
        # The components actually instantiated in the built netlist (ground truth for the build).
        try:
            import yaml

            net = yaml.safe_load(pipeline_done.get("gf_netlist_yaml") or "") or {}
            diag["netlist_components"] = sorted(
                {v.get("component") for v in (net.get("instances") or {}).values()
                 if isinstance(v, dict) and v.get("component")}
            )
        except Exception as e:  # noqa: BLE001
            diag["netlist_parse_error"] = f"{type(e).__name__}: {e}"[:160]
    if layout_sim:
        diag["missing_models"] = layout_sim.get("missing_models") or []
        diag["sim_error"] = layout_sim.get("sim_error")
        diag["drc_violations"] = layout_sim.get("drc_violations")
        diag["routing_ok"] = layout_sim.get("routing_ok")
    return diag


def run_suite(
    prompts: list[dict], config: RunConfig | None = None, orch=None, user_simulator=None
) -> list[dict]:
    """Run many prompts (each {id, prompt, level}); returns e2_funnel-ready result dicts."""
    return [
        run_one(
            p["prompt"],
            p.get("id", f"p{i}"),
            p.get("level", "?"),
            config=config,
            orch=orch,
            user_simulator=user_simulator,
        )
        for i, p in enumerate(prompts)
    ]

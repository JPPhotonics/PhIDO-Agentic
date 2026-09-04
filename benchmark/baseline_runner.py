"""Headless runner for the RIGID single-pass pipeline (the `main`-branch baseline).

Reproduces the rigid p100->p300 sequence (entity extraction -> component selection ->
DSL -> schematic) without Streamlit, then feeds the resulting gdsfactory netlist through the
**same `run_layout_simulation` tail** the agentic arm uses (e2_runner). Both arms therefore
share the identical layout/sim/DRC backend + `e2_funnel` scorer, so a `compare_arms(rigid,
agentic)` isolates the **LLM-integration architecture** (rigid single-pass vs agentic
orchestrator+MCP) as the only variable.

The rigid pipeline files (webapp/llm_api/utils/DemoPDK) are unchanged from `main` on this
branch, so this is a faithful baseline. Result shape matches `e2_runner.run_one`.
"""

from __future__ import annotations

import time

import e2_runner as R  # funnel_from_results, STAGES
import yaml
from token_meter import TokenMeter


# --- session: attr + item access over one dict (mirrors st.session_state usage) ---
class _Session(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _patch_token_tracking():
    """Replace llm_api's Streamlit-bound token tracker with a module-level dict (headless)."""
    from PhotonicsAI.Photon import llm_api

    tok = {"non_cached_input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    llm_api.get_session_token_usage = lambda: tok
    llm_api.reset_token_usage = lambda: tok.update(
        non_cached_input_tokens=0, cached_input_tokens=0, output_tokens=0
    )

    def _add(inp, out, is_cached=False):
        tok["cached_input_tokens" if is_cached else "non_cached_input_tokens"] += inp
        tok["output_tokens"] += out

    llm_api.add_token_usage = _add
    return tok


class NotADesignError(RuntimeError):
    """intent_classification rejected the prompt (category != 1)."""


def build_rigid_netlist(prompt: str, model: str = "o1") -> dict:
    """Run the rigid p100->p300 stages headlessly; return the populated circuit_dsl."""
    from PhotonicsAI.Photon import llm_api, utils
    from PhotonicsAI.Photon.DemoPDK import (
        footprint_netlist,
        get_params,
        get_ports_info,
        list_of_cnames,
        list_of_docs,
    )

    S = _Session(
        step_results={},
        template_selected=False,
        p100_list_of_docs=list_of_docs,
        p100_list_of_cnames=list_of_cnames,
        p100_llm_api_selection=model,
    )

    # ── p100: intent gate -> entity extraction -> preschematic ──────────
    if llm_api.intent_classification(prompt).category_id != 1:
        raise NotADesignError("intent_classification: not a layout-design prompt")
    pretemplate = llm_api.entity_extraction(prompt)
    preschematic = llm_api.preschematic(pretemplate, model)

    # ── p200: per-component search, take top-1 (the deterministic single-pass choice) ──
    # Symmetry with the agentic arm: restrict the candidate pool to the same buildable+
    # simulatable whitelist so neither arm can select a part the other structurally cannot.
    from mcp_servers.pdk_whitelist import simulatable_modules

    sim = simulatable_modules()
    keep = [i for i, nm in enumerate(list_of_cnames) if nm in sim]
    docs_w = [list_of_docs[i] for i in keep]
    cnames_w = [list_of_cnames[i] for i in keep]

    selected = []
    for c in pretemplate["components_list"]:
        r = llm_api.llm_search(c, docs_w)
        if not r.match_list:
            raise RuntimeError(f"llm_search returned no match for {c!r}")
        selected.append(cnames_w[r.match_list[0]])

    # ── p200b: build the circuit DSL (no LLM) ───────────────────────────
    circuit_dsl = {
        "doc": {
            "title": pretemplate.get("title", ""),
            "description": pretemplate.get("brief_summary", ""),
            "reference": "(link)",
            "labels": [""],
        },
        "nodes": {f"N{i}": {"component": comp} for i, comp in enumerate(selected, 1)},
        "edges": pretemplate.get("circuit_instructions", ""),
        "properties": {},
    }

    # ── p300: hand-offs + ports/params + apply_settings + DOT edges + placements ──
    S.p200_pretemplate_copy = {"components_list": pretemplate["components_list"]}
    S.p200_preschematic = preschematic
    S["p300_circuit_dsl"] = circuit_dsl
    S["p300_circuit_dsl"] = get_ports_info(S["p300_circuit_dsl"])
    S["p300_circuit_dsl"] = get_params(S["p300_circuit_dsl"])
    S["p300_circuit_dsl"] = llm_api.apply_settings(S, model)

    S["p300_dot_string_draft"] = utils.circuit_to_dot(S["p300_circuit_dsl"])
    if len(S["p300_circuit_dsl"]["nodes"]) > 0:  # webapp.py:550-565, verbatim
        S["p300_dot_string"] = llm_api.dot_add_edges(S)
        S["p300_dot_string"] = llm_api.dot_verify(S)
        for _ in range(4):
            if utils.dot_planarity(S["p300_dot_string"]):
                break
            S["p300_dot_string"] = llm_api.dot_add_edges_errorfunc(S)
            S["p300_dot_string"] = llm_api.dot_verify(S)
    else:
        S["p300_dot_string"] = llm_api.dot_add_edges_templates(S)
    S["p300_dot_string"] = llm_api.dot_verify(S)

    S["p300_circuit_dsl"] = utils.edges_dot_to_yaml(S)
    S["p300_footprints_dict"], S["p300_circuit_dsl"] = footprint_netlist(
        S["p300_circuit_dsl"]
    )
    S["p300_dot_string_scaled"] = utils.dot_add_node_sizes(
        S["p300_dot_string"],
        utils.multiply_node_dimensions(S["p300_footprints_dict"], 0.01),
    )
    S["p300_graphviz_node_coordinates"] = utils.multiply_node_dimensions(
        utils.get_graphviz_placements(S["p300_dot_string_scaled"]), 100 / 72
    )
    S["p300_circuit_dsl"] = utils.add_placements_to_dsl(S)
    S["p300_circuit_dsl"] = utils.add_final_ports(S)
    return S["p300_circuit_dsl"]


def run_one(
    prompt: str, prompt_id: str = "p", level: str = "?", model: str = "o1"
) -> dict:
    """Rigid-pipeline funnel result for one prompt (same shape as e2_runner.run_one)."""
    from mcp_servers.pipeline_orchestrator import run_layout_simulation
    from PhotonicsAI.Photon import utils

    t0 = time.perf_counter()
    failed_at = None
    netlist_yaml, layout_sim = None, None
    meter = TokenMeter()
    meter.__enter__()  # try/finally so a build exception still restores the SDK patch
    try:
        try:
            circuit_dsl = build_rigid_netlist(prompt, model)
            gf_net = utils.dsl_to_gf(circuit_dsl)
            optical = gf_net.get("routes", {}).get("optical", {})
            if optical.get("links"):
                # match the agentic netlist's route format (cross_section) so the SHARED tail can
                # route the rigid netlist too -> single-backend, apples-to-apples comparison
                optical.setdefault("settings", {})["cross_section"] = "strip"
            else:
                gf_net.pop("routes", None)  # single-component: nothing to route
            netlist_yaml = yaml.dump(gf_net, sort_keys=False)
        except NotADesignError:
            failed_at = "intent"
        except Exception as e:  # noqa: BLE001 — keep the batch alive; record where it died
            failed_at = f"build:{type(e).__name__}"

        if netlist_yaml:
            for ev in run_layout_simulation(netlist_yaml):
                if ev.get("type") == "layout_sim_done":
                    layout_sim = ev["result"]
    finally:
        meter.__exit__()
    tok = meter.snapshot()

    pipeline_done = {"gf_netlist_yaml": netlist_yaml} if netlist_yaml else None
    stages = R.funnel_from_results(pipeline_done, layout_sim)
    R.save_run_outputs("rigid_baseline", prompt_id, prompt, pipeline_done, layout_sim,
                       stages, {"failed_at": failed_at})

    return {
        "prompt_id": prompt_id,
        "level": level,
        "arm": "rigid_baseline",
        "stages": stages,
        "cost": {
            "latency_s": time.perf_counter() - t0,
            "rounds": 0,
            "failed_at": failed_at,
            "tokens_in": tok["tokens_in"],
            "tokens_out": tok["tokens_out"],
        },
    }


_patch_token_tracking()

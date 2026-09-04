"""Ablation routing test: extraction_mode actually selects the iterative builder.

Hermetic — monkeypatches the orchestrator's builder entry points + client factory so it
runs with NO LLM/network. Proves the toggle changes the pipeline's code path (not just that
a string propagates): "iterative" → build_circuit_iterative; "single_shot" → extract_design_intent;
"auto" → picks by component count vs _AUTO_ITERATIVE_THRESHOLD.

Needs the repo root on sys.path (added below) for mcp_servers; uses the synced wt-graphrag venv.
Run: PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/tests/test_iterative_builder_ablation.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # wt-graphrag root for mcp_servers

import mcp_servers.pipeline_orchestrator as po

_EXPLORE = {"messages": [], "tool_log": [], "extracted": {"components": [], "specs": []},
            "requirement_manifest": None}


def _run_mode(mode, extracted=None):
    """Drive run_pipeline_finalize with the builders stubbed; report which path was taken."""
    flags = {"iter": False, "extract": False}
    orig = (po.create_client, po.build_circuit_iterative, po.extract_design_intent)

    def fake_iter(*a, **k):
        flags["iter"] = True
        return
        yield  # make it a generator that yields nothing -> design_intent stays None -> early return

    def fake_extract(*a, **k):
        flags["extract"] = True
        return
        yield

    po.create_client = lambda model: object()        # no real client / no key needed
    po.build_circuit_iterative = fake_iter
    po.extract_design_intent = fake_extract
    try:
        state = {**_EXPLORE, "extracted": extracted or _EXPLORE["extracted"]}
        events = list(po.run_pipeline_finalize(state, "design a splitter", model="m", extraction_mode=mode))
    finally:
        po.create_client, po.build_circuit_iterative, po.extract_design_intent = orig
    return flags, events


def test_iterative_mode_routes_to_iterative_builder():
    flags, events = _run_mode("iterative")
    assert flags["iter"] is True and flags["extract"] is False
    assert any(e.get("phase") == "iterative_build" for e in events)   # emitted the iterative phase


def test_single_shot_routes_to_extract():
    flags, _ = _run_mode("single_shot")
    assert flags["extract"] is True and flags["iter"] is False


def test_auto_picks_iterative_when_large():
    big = {"components": [{} for _ in range(po._AUTO_ITERATIVE_THRESHOLD)], "specs": []}
    flags, _ = _run_mode("auto", extracted=big)
    assert flags["iter"] is True and flags["extract"] is False


def test_auto_picks_single_shot_when_small():
    flags, _ = _run_mode("auto", extracted={"components": [{}], "specs": []})
    assert flags["extract"] is True and flags["iter"] is False


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} iterative-builder ablation tests passed.")


if __name__ == "__main__":
    main()

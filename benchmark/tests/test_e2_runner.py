"""Tests for the headless E2 runner (mock orchestrator — no real pipeline/LLM)."""

from __future__ import annotations

import e2_funnel as F
import e2_runner as R


class FakeOrch:
    """Scriptable orchestrator: fixed event lists per phase; records finalize kwargs."""

    def __init__(self, explore, finalize, layout, capture=None):
        self._explore, self._finalize, self._layout = explore, finalize, layout
        self.capture = capture if capture is not None else {}

    def explore(self, prompt, model):
        yield from self._explore

    def finalize(self, explore_state, prompt, model, **kw):
        self.capture["finalize_kw"] = kw
        yield from self._finalize

    def layout_sim(self, netlist):
        yield from self._layout


EXPLORE_OK = [{"type": "clarification", "questions": {}}]
EXPLORE_NONE = [{"type": "phase", "phase": "exploring"}]


def _done(netlist="yaml"):
    return [{"type": "pipeline_done", "result": {"gf_netlist_yaml": netlist}}]


def _layout(routing=True, missing=None, sim_err=None, drc=True):
    return [{"type": "layout_sim_done", "result": {
        "routing_ok": routing, "missing_models": missing or [], "sim_error": sim_err, "drc_clean": drc}}]


def test_full_success():
    orch = FakeOrch(EXPLORE_OK, _done(), _layout(True, [], None, True))
    r = R.run_one("p", orch=orch)
    assert all(r["stages"][s] for s in R.STAGES)
    assert r["cost"]["failed_at"] is None


def test_died_at_routing():
    orch = FakeOrch(EXPLORE_OK, _done(), _layout(routing=False, missing=["x"], drc=None))
    r = R.run_one("p", orch=orch)
    assert r["stages"]["instantiate"] is True and r["stages"]["routing_ok"] is False
    assert r["stages"]["models"] is False and r["stages"]["drc_clean"] is False


def test_validation_failed_no_instantiate():
    orch = FakeOrch(EXPLORE_OK, [{"type": "validation_failed", "gate": "topology_gate"}], _layout())
    r = R.run_one("p", orch=orch)
    assert r["stages"]["instantiate"] is False
    assert r["cost"]["failed_at"] == "validation_failed"


def test_explore_dead_end():
    orch = FakeOrch(EXPLORE_NONE, _done(), _layout())
    r = R.run_one("p", orch=orch)
    assert not any(r["stages"].values()) and r["cost"]["failed_at"] == "explore"


def test_toggle_passthrough():
    cap = {}
    orch = FakeOrch(EXPLORE_OK, _done(), _layout(), capture=cap)
    cfg = R.RunConfig(enable_topology_gate=False, enable_ar_gate=True, max_critic_rounds=0,
                      extraction_mode="iterative")
    R.run_one("p", config=cfg, orch=orch)
    kw = cap["finalize_kw"]
    assert kw["enable_topology_gate"] is False and kw["enable_ar_gate"] is True and kw["max_critic_rounds"] == 0
    assert kw["extraction_mode"] == "iterative"          # iterative builder selected via config


def test_retry_rounds_counted():
    finalize = [{"type": "phase", "phase": "validation_retry"},
                {"type": "phase", "phase": "validation_retry"}] + _done()
    orch = FakeOrch(EXPLORE_OK, finalize, _layout())
    r = R.run_one("p", orch=orch)
    assert r["cost"]["rounds"] == 2


def test_ablation_feeds_e2_funnel():
    """A gate-on vs gate-off ablation flows straight into e2_funnel.compare_arms."""
    on = FakeOrch(EXPLORE_OK, _done(), _layout(drc=True))
    off = FakeOrch(EXPLORE_OK, _done(), _layout(drc=False))
    base = [R.run_one(f"p{i}", prompt_id=f"p{i}", arm="gate_off", orch=off) for i in range(3)]
    treat = [R.run_one(f"p{i}", prompt_id=f"p{i}", arm="gate_on", orch=on) for i in range(3)]
    cmp = F.compare_arms(base, treat)
    assert cmp["by_stage"]["drc_clean"]["delta"] == 1.0      # gate-on fully recovers DRC-clean here
    assert cmp["n_paired"] == 3


def test_builder_arm_ab():
    """single-shot vs iterative builder is a clean E2 arm comparison (the rescope target)."""
    ss = FakeOrch(EXPLORE_OK, _done(), _layout(drc=False))
    it = FakeOrch(EXPLORE_OK, _done(), _layout(drc=True))
    base = [R.run_one(f"p{i}", prompt_id=f"p{i}", arm="single_shot",
                      config=R.RunConfig(extraction_mode="single_shot"), orch=ss) for i in range(3)]
    treat = [R.run_one(f"p{i}", prompt_id=f"p{i}", arm="iterative",
                       config=R.RunConfig(extraction_mode="iterative"), orch=it) for i in range(3)]
    cmp = F.compare_arms(base, treat)
    assert cmp["n_paired"] == 3 and cmp["by_stage"]["drc_clean"]["delta"] == 1.0


def test_finalize_incomplete_attributed():
    """Finalize that ends with no netlist and no error event is attributed to the last phase,
    not silently recorded as failed_at=None (the blind spot that hid the E2 cause)."""
    finalize = [{"type": "phase", "phase": "component_selection"},
                {"type": "phase", "phase": "schematic_building"}]  # ...then stops, no pipeline_done
    orch = FakeOrch(EXPLORE_OK, finalize, _layout())
    r = R.run_one("p", orch=orch)
    assert r["stages"]["instantiate"] is False
    assert r["cost"]["failed_at"] == "finalize_incomplete:schematic_building"


def test_diag_captures_modules_and_missing():
    """Diagnostics expose selected modules, built netlist components, and missing SAX models."""
    done = [{"type": "pipeline_done", "result": {
        "gf_netlist_yaml": "instances:\n  c1:\n    component: mzi1\n  c2:\n    component: straight\n",
        "selection": {"mappings": [{"pdk_module": "mzi1"}, {"pdk_module": "photodetector"}],
                      "unmapped": ["laser"]}}}]
    orch = FakeOrch(EXPLORE_OK, done, _layout(missing=["via_stack_slab_m3"], sim_err="boom"))
    r = R.run_one("p", orch=orch)
    d = r["diag"]
    assert d["selected_modules"] == ["mzi1", "photodetector"]
    assert d["unmapped"] == ["laser"]
    assert d["netlist_components"] == ["mzi1", "straight"]
    assert d["missing_models"] == ["via_stack_slab_m3"] and d["sim_error"] == "boom"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} e2-runner tests passed.")


if __name__ == "__main__":
    main()

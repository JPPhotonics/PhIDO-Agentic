"""Per-failure-class attribution for the E2 ablation pilot.

base_agentic (agentic floor, all features off) fails to INSTANTIATE 4 prompts that rigid builds
(TB032/036/070/102 — see the agentic-floor diagnosis). This script asks, per failing prompt,
WHICH feature recovers it: for each additive arm (base+X) does the prompt now instantiate, and
does the full system? Emits results/e2_ablation_attribution.md. Safe to run anytime — shows
whatever arms are present so far (missing arms shown as "—").

Run: PYTHONPATH=<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_e2_attribution.py
"""

from __future__ import annotations

import json
import time

import e2_funnel as F
from report import Reporter
from run_e2_testbench import ROOT

OUT = ROOT / "benchmark" / "results" / "e2_ablation.json"
FAIL = ["TB032", "TB036", "TB070", "TB102"]  # base_agentic instantiate-failures
# feature -> additive arm (base+X). routing dropped (treated as unimplemented, no routing arm).
ADD = {"kg": "base_plus_kg",
       "gate": "base_plus_gate", "critic": "base_plus_critic"}
COLS = ["rigid_baseline", "base_agentic"] + list(ADD.values()) + ["full"]
# mechanism note per failing prompt (from the diagnosis)
MECH = {
    "TB032": "internal _mmi1x2/_mmi2x2 blocks; delta_length '200 µm' (str); width None",
    "TB036": "delta_length '100um' (units-as-string) -> str/int build error",
    "TB070": "delta_length '30µm,50µm,30µm,50µm' (multi-value string in one scalar)",
    "TB102": "4x mrr_2x2 outputs routed into one 2-port straight (over-subscribed)",
}


def _load():
    for _ in range(8):
        try:
            return json.load(open(OUT))
        except Exception:  # noqa: BLE001 — file rewritten each run
            time.sleep(0.5)
    return json.load(open(OUT))


def _inst(d, arm, pid):
    """instantiate outcome (✓/✗) for arm/prompt, or None if arm/prompt absent."""
    reps = d.get(arm, {}).get(pid)
    if not reps:
        return None
    return bool(F.normalize(reps[0]["stages"])["instantiate"])


def _cell(v):
    return "—" if v is None else ("✓" if v else "✗")


def main():
    d = _load()
    rep = Reporter(OUT.with_suffix("").with_name("e2_ablation_attribution.md"),
                   "E2 attribution — which feature recovers each agentic-floor failure",
                   meta={"failing_prompts": ", ".join(FAIL),
                         "note": "instantiate outcome per arm; base_agentic fails all 4 by construction",
                         "raw_json": OUT.name})

    rep.h("instantiate outcome per arm (✓ builds / ✗ crashes GDS build / — not run yet)")
    rep.table(["prompt", "level", "mechanism"] + COLS,
              [[pid,
                (d.get("base_agentic", {}).get(pid, [{}])[0].get("level", "?")),
                MECH.get(pid, "")]
               + [_cell(_inst(d, arm, pid)) for arm in COLS]
               for pid in FAIL])

    rep.h("Attribution: feature(s) that flip instantiate ✗→✓ (additive over base_agentic)")
    rows = []
    for pid in FAIL:
        base_ok = _inst(d, "base_agentic", pid)
        fixers = [feat for feat, arm in ADD.items()
                  if _inst(d, arm, pid) is True and base_ok is False]
        pending = [feat for feat, arm in ADD.items() if _inst(d, arm, pid) is None]
        full_ok = _inst(d, "full", pid)
        rows.append([pid,
                     ", ".join(fixers) or ("(none yet)" if not pending else ""),
                     ("?" if full_ok is None else _cell(full_ok)),
                     ", ".join(pending) or "—"])
    rep.table(["prompt", "fixed by (additive)", "full builds?", "arms still pending"], rows)

    # readiness: all additive arms + full have all 4 failing prompts
    ready = all(_inst(d, arm, pid) is not None
                for arm in list(ADD.values()) + ["full"] for pid in FAIL)
    rep.line(f"\n_status: {'COMPLETE — all additive arms + full have the 4 prompts' if ready else 'PARTIAL — some arms still running'}_")
    rep.save()
    print(f"READY={ready}", flush=True)
    return ready


if __name__ == "__main__":
    main()

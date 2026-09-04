"""Harder E2 prompt set — designed to make the topology gate's value VISIBLE end-to-end.

The default E2 prompts (`Testbench_modified.csv`) all describe small, valid topologies the
interpreter rarely gets wrong, so the topology gate has nothing to catch and the gate-on/off
ablation comes out neutral. These prompts instead target the architectures the gate actually
has rules for (`architecture_rules.lp`) at scales/configurations where single-shot generation
is PRONE to drift — wrong component counts, a missing required block — i.e. exactly the
template-conformance errors the gate detects and the repair loop can fix.

Each prompt requests a VALID, well-defined topology (the gold). The gate only fires if
GENERATION drifts from that gold; the gate-on arm then re-extracts with the Clingo feedback
(`pipeline_orchestrator` single-shot retry loop, ~line 1541) while the gate-off arm cannot.
Measuring the *final* design intent's rule-conformance for both arms isolates the gate's
repair value — see `run_hard_e2.py`.

Gold fields per prompt:
  architecture  — expected `DesignIntent.architecture_type` (drives which .lp rules apply)
  n_value       — expected scaling parameter
  gold_counts   — {clingo_component_type: exact expected count} the valid design must have
  buildable     — True if the architecture maps to DemoPDK cells (so DRC can also move);
                  exotic meshes (benes/clements/reck/qpsk) are intent-level-only.
"""

from __future__ import annotations

HARD_PROMPTS: list[dict] = [
    # ── buildable: splitter trees (mmi1x2 cascades) — gate rule: tree_wrong_splitters ──
    {
        "id": "h_tree8",
        "level": "hard",
        "buildable": True,
        "architecture": "splitter_tree",
        "n_value": 8,
        "gold_counts": {"splitter": 7},
        "prompt": "Lay out a 1x8 optical power splitter tree built entirely from cascaded 1x2 MMI splitters.",
    },
    {
        "id": "h_tree16",
        "level": "hard",
        "buildable": True,
        "architecture": "splitter_tree",
        "n_value": 16,
        "gold_counts": {"splitter": 15},
        "prompt": "Design a 1x16 power splitter tree composed of 1x2 MMI splitters in a balanced binary tree.",
    },
    {
        "id": "h_tree32",
        "level": "hard",
        "buildable": True,
        "architecture": "splitter_tree",
        "n_value": 32,
        "gold_counts": {"splitter": 31},
        "prompt": "Build a 1x32 splitter tree using only 1x2 MMI power splitters, fully balanced.",
    },
    # ── buildable: WDM demux (ring resonators) — gate rule: wdm_insufficient_rings ──
    {
        "id": "h_wdm4",
        "level": "hard",
        "buildable": True,
        "architecture": "wdm_demux",
        "n_value": 4,
        "gold_counts": {"ring_resonator": 4},
        "prompt": "Design a 4-channel WDM demultiplexer using add-drop ring resonators, one ring per channel.",
    },
    {
        "id": "h_wdm8",
        "level": "hard",
        "buildable": True,
        "architecture": "wdm_demux",
        "n_value": 8,
        "gold_counts": {"ring_resonator": 8},
        "prompt": "Lay out an 8-channel WDM demultiplexer built from cascaded add-drop ring resonators, one ring per wavelength channel.",
    },
    # ── buildable: MZI templates — gate rule: mzi_wrong_splitter/combiner_count ──
    {
        "id": "h_mzi",
        "level": "hard",
        "buildable": True,
        "architecture": "mzi",
        "n_value": None,
        "gold_counts": {"splitter": 1, "combiner": 1},
        "prompt": "Design a balanced Mach-Zehnder interferometer: an input 1x2 splitter feeding two equal-length arms that recombine in a 2x1 combiner.",
    },
    # ── intent-level (exotic, likely not in DemoPDK): QPSK — many required blocks ──
    {
        "id": "h_qpsk",
        "level": "hard",
        "buildable": False,
        "architecture": "qpsk",
        "n_value": None,
        "gold_counts": {"splitter": 1, "combiner": 1, "mzm": 2, "phase_shifter": 1},
        "prompt": "Design an optical QPSK modulator: a splitter feeding two child Mach-Zehnder modulators (I and Q), one with a 90-degree phase shifter, recombined by a combiner.",
    },
    # ── intent-level: Benes / Clements / Reck meshes — exact switch/MZI counts ──
    {
        "id": "h_benes4",
        "level": "hard",
        "buildable": False,
        "architecture": "benes",
        "n_value": 4,
        "gold_counts": {"mzm": 6},
        "prompt": "Design a 4x4 Benes rearrangeable switch network built from 2x2 Mach-Zehnder switch elements.",
    },
    {
        "id": "h_clements4",
        "level": "hard",
        "buildable": False,
        "architecture": "clements",
        "n_value": 4,
        "gold_counts": {"mzm": 6},
        "prompt": "Design a 4x4 universal unitary photonic processor as a Clements rectangular mesh of 2x2 MZI blocks.",
    },
    {
        "id": "h_reck4",
        "level": "hard",
        "buildable": False,
        "architecture": "reck",
        "n_value": 4,
        "gold_counts": {"mzm": 6},
        "prompt": "Design a 4x4 Reck triangular mesh of 2x2 Mach-Zehnder interferometer blocks for arbitrary unitary transformations.",
    },
    {
        "id": "h_clements8",
        "level": "hard",
        "buildable": False,
        "architecture": "clements",
        "n_value": 8,
        "gold_counts": {"mzm": 28},
        "prompt": "Design an 8x8 Clements rectangular mesh of 2x2 MZI blocks implementing an arbitrary 8-mode unitary.",
    },
]


def prompts(buildable_only: bool = False) -> list[dict]:
    return [p for p in HARD_PROMPTS if (p["buildable"] or not buildable_only)]

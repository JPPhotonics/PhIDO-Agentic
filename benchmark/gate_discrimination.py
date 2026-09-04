"""B4 — topology-gate DISCRIMINATION eval, aligned to the gate's ACTUAL rule vocabulary.

The powered ablation (N=20×3) showed the topology gate is ~outcome-neutral on the natural
test prompts because those prompts almost never contain the errors the gate checks. This
module measures the gate where it can actually be measured: a labeled valid/invalid set built
in the gate's own vocabulary — architecture-template conformance, NOT generic graph errors.

`architecture_rules.lp` reasons over `architecture/1`, `n_value/1`, `component/3`,
`connection/2`, `requirement_trace/2`. Its error classes are template-conformance checks:
mzi (1 splitter + 1 combiner + connected path), splitter_tree (N power-of-2, N-1 splitters),
wdm_demux (>=N rings), qpsk (1 spl + 1 comb + 2 mzm + phase shifter), and requirement
traceability. So the invalid cases here are constructed by VIOLATING exactly those rules — the
label is known by construction, and a violation maps to a named error class. We run the REAL
`validate_topology` and score with `gate_eval.formal_gate_eval`:

  * per-error-class catch rate (does the gate fire on the class it should?),
  * false-reject rate on the matched valid seeds,
  * coverage gap = classes the gate never catches (rule not encoded / mis-encoded),
  * code-precision = when the gate fires, does it fire the *expected* error code?

This isolates gate QUALITY from the DRC-stage noise (flip-rate 0.3-0.65) that confounded the
end-to-end ablation. Unlike `inject_topology.py` (generic graph-error taxonomy the .lp does NOT
check), every class here corresponds to a rule in `architecture_rules.lp`.

Run: PYTHONPATH=<wt> <wt>/.venv/bin/python <wt>/benchmark/gate_discrimination.py
"""

from __future__ import annotations

import pathlib

from gate_eval import formal_gate_eval
from report import Reporter

from mcp_servers.clingo_validator import validate_topology
from mcp_servers.models import (
    ComponentIntent,
    Connection,
    DesignIntent,
    RequirementManifest,
    RequirementTrace,
    UserRequirement,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- DesignIntent construction helpers ------------------------------------
def _comp(cid: str, ctype: str, role: str | None = None) -> ComponentIntent:
    return ComponentIntent(
        id=cid, description=f"{ctype} {cid}", role=role, component_type=ctype
    )


def _conn(a: str, b: str) -> Connection:
    return Connection(from_component=a, to_component=b, description=f"{a}->{b}")


def _di(
    title: str, arch: str | None, comps, conns, n=None, manifest=None, traces=None
) -> DesignIntent:
    return DesignIntent(
        title=title,
        brief_summary=title,
        components=comps,
        connections=conns,
        architecture_type=arch,
        n_value=n,
        requirement_manifest=manifest,
        requirement_traces=traces or [],
    )


# --- valid seeds, one (or more) per architecture the .lp covers -----------
def _valid_mzi() -> DesignIntent:
    return _di(
        "MZI",
        "mzi",
        [_comp("c1", "splitter"), _comp("c2", "combiner")],
        [_conn("c1", "c2")],
    )


def _valid_tree4() -> DesignIntent:
    # 1x4 splitter tree: power-of-2, N-1 = 3 splitters
    return _di(
        "1x4 splitter tree",
        "splitter_tree",
        [_comp(f"c{i}", "splitter") for i in range(1, 4)],
        [_conn("c1", "c2"), _conn("c1", "c3")],
        n=4,
    )


def _valid_wdm4() -> DesignIntent:
    # 4-channel demux: >= 4 rings
    return _di(
        "4ch WDM demux",
        "wdm_demux",
        [_comp(f"r{i}", "ring_resonator") for i in range(1, 5)],
        [_conn("r1", "r2"), _conn("r2", "r3"), _conn("r3", "r4")],
        n=4,
    )


def _valid_qpsk() -> DesignIntent:
    return _di(
        "QPSK modulator",
        "qpsk",
        [
            _comp("spl", "splitter"),
            _comp("comb", "combiner"),
            _comp("m1", "mzm"),
            _comp("m2", "mzm"),
            _comp("ps", "phase_shifter"),
        ],
        [
            _conn("spl", "m1"),
            _conn("spl", "m2"),
            _conn("m1", "comb"),
            _conn("m2", "comb"),
        ],
    )


def _mzm_mesh(arch: str, title: str, k: int) -> DesignIntent:
    # benes/clements/reck 4x4 all expect 6 switching units (mzm) for N=4
    return _di(
        title,
        arch,
        [_comp(f"s{i}", "mzm") for i in range(1, k + 1)],
        [_conn(f"s{i}", f"s{i + 1}") for i in range(1, k)],
        n=4,
    )


def _valid_reqs() -> DesignIntent:
    # requirement-traceability seed: all reqs traced 'direct'
    reqs = [
        UserRequirement(
            id=f"r{i}",
            category="functional",
            description=f"req {i}",
            source_span=f"req {i}",
            priority="explicit",
        )
        for i in (1, 2)
    ]
    traces = [
        RequirementTrace(
            requirement_id=f"r{i}", satisfied_by=["c1"], satisfaction_type="direct"
        )
        for i in (1, 2)
    ]
    return _di(
        "Traced design",
        "mzi",
        [_comp("c1", "splitter"), _comp("c2", "combiner")],
        [_conn("c1", "c2")],
        manifest=RequirementManifest(requirements=reqs),
        traces=traces,
    )


# --- invalid cases: each VIOLATES one named .lp rule (label by construction)
# (id, error_class, expected_clingo_code, DesignIntent)
def invalid_cases() -> list[dict]:
    # (error_class, expected_clingo_code, DesignIntent) — class == code by construction here
    cases: list[tuple[str, str, DesignIntent]] = [
        # MZI template
        (
            "mzi_wrong_splitter_count",
            "mzi_wrong_splitter_count",
            _di(
                "MZI 2 splitters",
                "mzi",
                [
                    _comp("c1", "splitter"),
                    _comp("c1b", "splitter"),
                    _comp("c2", "combiner"),
                ],
                [_conn("c1", "c2")],
            ),
        ),
        (
            "mzi_wrong_combiner_count",
            "mzi_wrong_combiner_count",
            _di("MZI no combiner", "mzi", [_comp("c1", "splitter")], []),
        ),
        (
            "mzi_disconnected",
            "mzi_disconnected",
            _di(
                "MZI disconnected",
                "mzi",
                [_comp("c1", "splitter"), _comp("c2", "combiner")],
                [],
            ),
        ),  # no path c1->c2
        # splitter tree template
        (
            "tree_not_power_of_2",
            "tree_not_power_of_2",
            _di(
                "1x3 tree",
                "splitter_tree",
                [_comp(f"c{i}", "splitter") for i in range(1, 3)],
                [_conn("c1", "c2")],
                n=3,
            ),
        ),
        (
            "tree_wrong_splitters",
            "tree_wrong_splitters",
            _di(
                "1x4 tree, 2 splitters",
                "splitter_tree",
                [_comp(f"c{i}", "splitter") for i in range(1, 3)],
                [_conn("c1", "c2")],
                n=4,
            ),
        ),
        # benes / clements / reck meshes (4x4 -> expect 6 switches; give 4)
        (
            "benes_wrong_switch_count",
            "benes_wrong_switch_count",
            _mzm_mesh("benes", "Benes 4x4, 4 switches", 4),
        ),
        (
            "clements_wrong_mzi_count",
            "clements_wrong_mzi_count",
            _mzm_mesh("clements", "Clements 4x4, 4 MZIs", 4),
        ),
        (
            "reck_wrong_mzi_count",
            "reck_wrong_mzi_count",
            _mzm_mesh("reck", "Reck 4x4, 4 MZIs", 4),
        ),
        # wdm demux template
        (
            "wdm_insufficient_rings",
            "wdm_insufficient_rings",
            _di(
                "4ch demux, 2 rings",
                "wdm_demux",
                [_comp("r1", "ring_resonator"), _comp("r2", "ring_resonator")],
                [_conn("r1", "r2")],
                n=4,
            ),
        ),
        # qpsk template
        (
            "qpsk_wrong_splitter_count",
            "qpsk_wrong_splitter_count",
            _di(
                "QPSK 2 splitters",
                "qpsk",
                [
                    _comp("spl", "splitter"),
                    _comp("spl2", "splitter"),
                    _comp("comb", "combiner"),
                    _comp("m1", "mzm"),
                    _comp("m2", "mzm"),
                    _comp("ps", "phase_shifter"),
                ],
                [
                    _conn("spl", "m1"),
                    _conn("spl", "m2"),
                    _conn("m1", "comb"),
                    _conn("m2", "comb"),
                ],
            ),
        ),
        (
            "qpsk_wrong_combiner_count",
            "qpsk_wrong_combiner_count",
            _di(
                "QPSK no combiner",
                "qpsk",
                [
                    _comp("spl", "splitter"),
                    _comp("m1", "mzm"),
                    _comp("m2", "mzm"),
                    _comp("ps", "phase_shifter"),
                ],
                [_conn("spl", "m1"), _conn("spl", "m2")],
            ),
        ),
        (
            "qpsk_wrong_mzm_count",
            "qpsk_wrong_mzm_count",
            _di(
                "QPSK 1 mzm",
                "qpsk",
                [
                    _comp("spl", "splitter"),
                    _comp("comb", "combiner"),
                    _comp("m1", "mzm"),
                    _comp("ps", "phase_shifter"),
                ],
                [_conn("spl", "m1"), _conn("m1", "comb")],
            ),
        ),
        (
            "qpsk_missing_phase_shifter",
            "qpsk_missing_phase_shifter",
            _di(
                "QPSK no phase shifter",
                "qpsk",
                [
                    _comp("spl", "splitter"),
                    _comp("comb", "combiner"),
                    _comp("m1", "mzm"),
                    _comp("m2", "mzm"),
                ],
                [
                    _conn("spl", "m1"),
                    _conn("spl", "m2"),
                    _conn("m1", "comb"),
                    _conn("m2", "comb"),
                ],
            ),
        ),
        # requirement traceability
        (
            "unsatisfied_requirement",
            "unsatisfied_requirement",
            _di(
                "Untraced req",
                "mzi",
                [_comp("c1", "splitter"), _comp("c2", "combiner")],
                [_conn("c1", "c2")],
                manifest=RequirementManifest(
                    requirements=[
                        UserRequirement(
                            id="r1",
                            category="functional",
                            description="req",
                            source_span="req",
                            priority="explicit",
                        )
                    ]
                ),
                traces=[],
            ),
        ),  # requirement declared but NO trace at all
        (
            "unaddressed_requirement",
            "unaddressed_requirement",
            _di(
                "Unaddressed req",
                "mzi",
                [_comp("c1", "splitter"), _comp("c2", "combiner")],
                [_conn("c1", "c2")],
                manifest=RequirementManifest(
                    requirements=[
                        UserRequirement(
                            id="r1",
                            category="functional",
                            description="req",
                            source_span="req",
                            priority="explicit",
                        )
                    ]
                ),
                traces=[
                    RequirementTrace(
                        requirement_id="r1",
                        satisfied_by=[],
                        satisfaction_type="unaddressed",
                    )
                ],
            ),
        ),
    ]
    return [
        {
            "id": cls,
            "label": "invalid",
            "error_class": cls,
            "expected_code": code,
            "di": di,
        }
        for cls, code, di in cases
    ]


def valid_cases() -> list[dict]:
    builders = {
        "mzi": _valid_mzi,
        "splitter_tree": _valid_tree4,
        "wdm_demux": _valid_wdm4,
        "qpsk": _valid_qpsk,
        "requirements": _valid_reqs,
        "benes": lambda: _mzm_mesh("benes", "Benes 4x4", 6),
        "clements": lambda: _mzm_mesh("clements", "Clements 4x4", 6),
        "reck": lambda: _mzm_mesh("reck", "Reck 4x4", 6),
    }
    return [
        {"id": f"valid-{k}", "label": "valid", "di": fn()} for k, fn in builders.items()
    ]


def _run_gate(di: DesignIntent) -> tuple[bool, list[str]]:
    """Returns (rejected, fired_codes)."""
    fb = validate_topology(di)
    codes = [f.context.get("clingo_error_code") for f in fb]
    return (len(fb) > 0, codes)


def build_records() -> list[dict]:
    records = []
    for case in valid_cases() + invalid_cases():
        rejected, codes = _run_gate(case["di"])
        rec = {
            "id": case["id"],
            "label": case["label"],
            "rejected": rejected,
            "fired_codes": codes,
        }
        if "error_class" in case:
            rec["error_class"] = case["error_class"]
            rec["expected_code"] = case["expected_code"]
            rec["code_match"] = case["expected_code"] in codes
        records.append(rec)
    return records


def main() -> None:
    records = build_records()
    report = formal_gate_eval(records)

    out_md = ROOT / "benchmark" / "results" / "gate_discrimination.md"
    rep = Reporter(
        out_md,
        "Topology-gate discrimination (rule-aligned, no DRC)",
        meta={
            "valid_seeds": report["n_valid"],
            "invalid_cases": report["n_invalid"],
            "error_classes": report["n_error_classes"],
        },
    )

    rep.h("Summary")
    rep.line(
        f"- false-reject rate: **{report['false_reject_rate']:.2f}** "
        f"(CI95 {tuple(round(x, 2) for x in report['false_reject_ci95'])})"
    )
    rep.line(
        f"- overall catch rate: **{report['catch_rate']:.2f}** "
        f"(CI95 {tuple(round(x, 2) for x in report['catch_rate_ci95'])})"
    )
    rep.line(
        f"- coverage gap (classes never caught): {report['coverage_gap'] or '(none)'}"
    )
    rep.line(
        f"- invalid accepted (missed): {report['soundness']['n_invalid_accepted']} "
        f"→ {report['soundness']['accepted_classes'] or '(none)'}"
    )

    rep.h("Per-error-class catch")
    inv = {r["id"]: r for r in records if r["label"] == "invalid"}
    rows = []
    for cls, st in report["per_error_class"].items():
        rec = next(r for r in inv.values() if r["error_class"] == cls)
        if not rec["rejected"]:
            verdict = "NOT CAUGHT"
        else:
            verdict = (
                "code ✓"
                if rec["code_match"]
                else f"code ✗ (fired {rec['fired_codes']})"
            )
        rows.append([cls, f"{st['catch_rate']:.2f}", verdict])
    rep.table(["error class", "catch rate", "code verdict"], rows)

    fr = [r["id"] for r in records if r["label"] == "valid" and r["rejected"]]
    if fr:
        rep.line(f"\n**[!] valid seeds the gate WRONGLY rejected:** {fr}")
    rep.line(f"\n_{report['soundness']['note']}_")
    rep.save()


if __name__ == "__main__":
    main()

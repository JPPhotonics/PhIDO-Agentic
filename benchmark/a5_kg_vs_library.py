"""A5 — PDK ingestion quality: KG PDK_Cells vs the DesignLibrary (enumerable gold). READ-ONLY.

Per BENCHMARK_ARCHITECTURE.md §5: the DesignLibrary is machine-readable and fully enumerable,
so it is near-free ground truth for PDK ingestion. A5 measures whether the KG faithfully
captured the PDK along several axes, strongest-evidence first:

  1. NODE precision/recall — did every library module become a PDK_Cell, and are there
     spurious cells? Gold = the set of DesignLibrary/*.py module stems (enumerable, strong).
  2. PER-EDGE-TYPE precision/recall against DOCSTRING gold — the upgrade. For PERFORMS_FUNCTION
     the gold is each module's docstring NodeLabels (minus the structural skip-labels the
     ingestion itself drops); for IMPLEMENTS we report edge PRESENCE plus an exploratory,
     heavily-caveated name-level proxy (docstrings name no single abstract Component). Names
     are matched normalization-aware (the KG vocabulary is fragmented), and every unmatched
     gold label is listed so vocabulary-mismatch is separable from genuine ingestion gaps.
  3. PORTS / PARAMETERS — measured only insofar as the KG actually represents them. The
     ingestion stores ports/parameters as PDK_Cell *node properties* (JSON strings), NOT as
     graph edges/nodes (see pdk_ingestion_agent._cell_to_props). We therefore report property
     coverage + value agreement, and explicitly state that no port/param GRAPH structure
     exists to score as edges (an ingestion-design finding, not a harness limitation).
  4. STRUCTURAL edge-type coverage — does each cell carry >=1 of each expected edge type
     (kept from the original A5; complements the per-function correctness above).

A5 correctness BOUNDS the downstream retrieval/E2 ceiling: if PDK edges are missing or wrong,
grounding is capped. ALL Cypher here is read-only; the harness never writes to the shared KB.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/a5_kg_vs_library.py
"""

from __future__ import annotations

import ast
import json
import pathlib

import yaml
from a5_scoring import (
    ACTIVE_PASSIVE,
    PERFORMS_SKIP_LABELS,
    cell_set_prf,
    normalize_set,
    performs_function_gold,
    unmatched_gold,
)
from kb_access import KB
from metrics import prf, set_prf
from report import Reporter
from stats import wilson

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "PhotonicsAI" / "KnowledgeBase" / "DesignLibrary"
OUT_MD = ROOT / "benchmark" / "results" / "a5_kg_vs_library.md"

# edge types a fully-ingested PDK_Cell is expected to carry; the first two are "required"
REQUIRED = ["IMPLEMENTS", "PERFORMS_FUNCTION"]
OPTIONAL = ["EXHIBITS", "FABRICATED_WITH", "COMPOSED_OF"]


# --------------------------------------------------------------------------- gold from docstrings
def gold_modules() -> set[str]:
    return {p.stem for p in LIB.glob("*.py") if p.stem != "__init__"}


def _parse_docstring_metadata(path: pathlib.Path) -> dict:
    """Parse one module's YAML front-matter docstring exactly as the ingestion does.

    Mirrors pdk_ingestion_agent.docstring_parser.parse_component_file: take the module-level
    docstring, split on the '---' front-matter separator, YAML-safe_load the remainder, and
    read NodeLabels / ports / Args. We parse inline (not by importing the ingestion package)
    because that package's __init__ eagerly imports the full agent + llm_api stack, which the
    benchmark must not pull in. The extraction logic here is identical, so gold still matches
    the ingestion's interpretation of each docstring.
    """
    raw = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    parts = raw.split("---", 1)
    meta: dict = {}
    if len(parts) > 1:
        try:
            meta = yaml.safe_load(parts[1].strip()) or {}
        except yaml.YAMLError:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def docstring_gold() -> dict[str, dict]:
    """module_name -> parsed docstring gold (labels, ports, param keys, function gold)."""
    gold: dict[str, dict] = {}
    for p in sorted(LIB.glob("*.py")):
        if p.stem == "__init__":
            continue
        meta = _parse_docstring_metadata(p)
        labels = meta.get("NodeLabels") or []
        if not isinstance(labels, list):
            labels = [labels]
        raw_args = meta.get("Args", {})
        if isinstance(raw_args, dict):
            param_keys = sorted(
                {
                    _norm_param_key(k)
                    for k in raw_args.keys()
                    if _norm_param_key(k) != "raw"
                }
            )
        else:
            param_keys = []
        gold[p.stem] = {
            "labels": [str(x) for x in labels],
            "ports": str(meta.get("ports", "unknown")),
            "param_keys": param_keys,
            "func_gold": performs_function_gold([str(x) for x in labels]),
        }
    return gold


# --------------------------------------------------------------------------- KG (read-only)
def kg_cells(kb: KB) -> dict[str, dict]:
    """module_name -> structural edge-type out-counts for every PDK_Cell."""
    rows = kb.q(
        "MATCH (p:PDK_Cell) "
        "OPTIONAL MATCH (p)-[e]->() "
        "RETURN p.module_name AS m, type(e) AS rel, count(e) AS n"
    )
    cells: dict[str, dict] = {}
    for r in rows:
        c = cells.setdefault(r["m"], {})
        if r["rel"]:
            c[r["rel"]] = c.get(r["rel"], 0) + r["n"]
    return cells


def kg_edge_targets(kb: KB, edge_type: str, target_label: str) -> dict[str, set[str]]:
    """module_name -> set of target `name` values for a typed edge (e.g. PERFORMS_FUNCTION)."""
    rows = kb.q(
        f"MATCH (p:PDK_Cell)-[:{edge_type}]->(t:{target_label}) "
        "RETURN p.module_name AS m, coalesce(t.name, t.display_name) AS name"
    )
    out: dict[str, set[str]] = {}
    for r in rows:
        if r["name"]:
            out.setdefault(r["m"], set()).add(r["name"])
    return out


def kg_node_props(kb: KB) -> dict[str, dict]:
    """module_name -> selected PDK_Cell node properties (ports/params representation probe)."""
    rows = kb.q(
        "MATCH (p:PDK_Cell) RETURN p.module_name AS m, p.ports AS ports, "
        "p.parameters AS parameters, p.port_details AS port_details, "
        "p.labels_list AS labels_list, p.numeric_specs AS numeric_specs, "
        "keys(p) AS keys"
    )
    return {r["m"]: r for r in rows}


def _norm_param_key(k: str) -> str:
    """Canonicalize a parameter key for symmetric gold/KG comparison.

    The docstring `Args:` block uses `-name:` syntax (no space after the dash), so YAML parses
    each entry as a mapping key that retains a leading hyphen (e.g. '-length'). Both the gold
    and the KG-stored parameters JSON inherit those dash-keys (the ingestion stores the same
    parse), so we strip the leading hyphen + whitespace and lowercase on BOTH sides to compare
    the actual parameter names robustly.
    """
    return str(k).lstrip("- ").strip().lower()


def _parse_json_keys(blob) -> list[str]:
    """Best-effort: normalized dict keys of a JSON-string node property (params)."""
    if not blob:
        return []
    try:
        d = json.loads(blob) if isinstance(blob, str) else blob
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(d, dict):
        return []
    return sorted(_norm_param_key(k) for k in d.keys() if _norm_param_key(k) != "raw")


def _parse_json_len(blob) -> int:
    """Best-effort length of a JSON-string list node property (port_details/numeric_specs)."""
    if not blob:
        return 0
    try:
        d = json.loads(blob) if isinstance(blob, str) else blob
        return len(d) if isinstance(d, (list, dict)) else 0
    except (json.JSONDecodeError, TypeError):
        return 0


# --------------------------------------------------------------------------- per-edge-type scoring
def score_edge_type(
    matched: list[str],
    gold_by_cell: dict[str, set[str]],
    pred_by_cell: dict[str, set[str]],
) -> dict:
    """Micro + macro P/R/F1 over cells that HAVE a non-empty normalized gold set.

    micro = pool tp/fp/fn across cells then prf; macro = mean of per-cell P and R.
    Also collects every unmatched gold label (recall miss) for the honesty section.
    """
    tp = fp = fn = 0
    macro_p: list[float] = []
    macro_r: list[float] = []
    per_cell: list[dict] = []
    unmatched: dict[str, list[str]] = {}
    scored: list[str] = []

    for m in matched:
        g = gold_by_cell.get(m, set())
        if not g:
            continue  # no asserted gold for this edge type on this cell -> not scored
        scored.append(m)
        p_set = pred_by_cell.get(m, set())
        ctp, cfp, cfn = cell_set_prf(p_set, g)
        tp += ctp
        fp += cfp
        fn += cfn
        cp, cr, cf = prf(ctp, cfp, cfn)
        macro_p.append(cp)
        macro_r.append(cr)
        miss = sorted(unmatched_gold(p_set, g))
        if miss:
            unmatched[m] = miss
        per_cell.append(
            {
                "module": m,
                "gold": sorted(g),
                "pred": sorted(p_set),
                "tp": ctp,
                "fp": cfp,
                "fn": cfn,
                "P": cp,
                "R": cr,
                "F1": cf,
            }
        )

    micro_p, micro_r, micro_f = prf(tp, fp, fn)
    n = len(scored)
    return {
        "n_cells": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_P": micro_p,
        "micro_R": micro_r,
        "micro_F1": micro_f,
        "macro_P": (sum(macro_p) / n) if n else 0.0,
        "macro_R": (sum(macro_r) / n) if n else 0.0,
        "per_cell": per_cell,
        "unmatched": unmatched,
        "scored_cells": scored,
    }


def _ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}]" if n else "—"


# --------------------------------------------------------------------------- main
def main() -> None:
    gold_mods = gold_modules()
    gold = docstring_gold()

    kb = KB()
    cells = kg_cells(kb)
    pf_targets = kg_edge_targets(kb, "PERFORMS_FUNCTION", "Design_Function")
    impl_targets = kg_edge_targets(kb, "IMPLEMENTS", "Component")
    node_props = kg_node_props(kb)
    kb.close()

    kg = set(cells)
    p, r, f1 = set_prf(kg, gold_mods)
    missing = sorted(gold_mods - kg)  # in library, absent from KG (recall miss)
    extra = sorted(kg - gold_mods)  # in KG, not a library module (precision miss)
    matched = sorted(gold_mods & kg)

    # ---- per-edge-type docstring-grounded scoring -------------------------------------
    # PERFORMS_FUNCTION: gold = normalized NodeLabels minus structural skip-labels.
    pf_gold = {m: gold[m]["func_gold"] for m in matched if m in gold}
    pf_pred = {m: normalize_set(pf_targets.get(m, set())) for m in matched}
    pf = score_edge_type(matched, pf_gold, pf_pred)

    # IMPLEMENTS: docstrings name no single abstract Component, so a clean name-level gold is
    # not derivable. We score edge PRESENCE (strong) and dump the raw KG IMPLEMENTS target
    # names per cell for inspection; no name-level P/R is asserted as a primary number.
    impl_present = sum(1 for m in matched if impl_targets.get(m))

    # ---- ports / params representation probe ------------------------------------------
    # The KG stores ports/params as node PROPERTIES (JSON), not as edges/nodes. Evidence:
    ports_prop = sum(
        1
        for m in matched
        if (node_props.get(m, {}).get("ports") not in (None, "", "unknown"))
    )
    params_prop = sum(
        1 for m in matched if _parse_json_keys(node_props.get(m, {}).get("parameters"))
    )
    portdet_prop = sum(
        1 for m in matched if _parse_json_len(node_props.get(m, {}).get("port_details"))
    )

    # ports VALUE agreement (docstring `ports` string vs node `ports` property) where both present
    ports_gold_cells = [
        m for m in matched if gold.get(m, {}).get("ports") not in (None, "", "unknown")
    ]
    ports_agree = sum(
        1
        for m in ports_gold_cells
        if str(node_props.get(m, {}).get("ports", "")).strip().lower()
        == str(gold[m]["ports"]).strip().lower()
    )

    # param-KEY agreement (docstring Args keys vs node parameters JSON keys), micro P/R
    pk_tp = pk_fp = pk_fn = 0
    pk_scored = 0
    for m in matched:
        g_keys = set(gold.get(m, {}).get("param_keys", []))
        if not g_keys:
            continue
        pk_scored += 1
        k_keys = set(_parse_json_keys(node_props.get(m, {}).get("parameters")))
        ctp = len(g_keys & k_keys)
        pk_tp += ctp
        pk_fp += len(k_keys) - ctp
        pk_fn += len(g_keys) - ctp
    pk_p, pk_r, pk_f = prf(pk_tp, pk_fp, pk_fn)

    # ---- structural edge-type coverage (kept) -----------------------------------------
    cov = {
        et: sum(1 for m in matched if cells[m].get(et)) for et in REQUIRED + OPTIONAL
    }
    missing_required = {
        et: [m for m in matched if not cells[m].get(et)] for et in REQUIRED
    }

    # =================================================================== report
    rep = Reporter(
        OUT_MD,
        "A5 — PDK ingestion: KG PDK_Cells vs DesignLibrary",
        meta={
            "gold_modules": len(gold_mods),
            "kg_pdk_cells": len(kg),
            "node_precision": round(p, 3),
            "node_recall": round(r, 3),
            "node_f1": round(f1, 3),
            "pf_micro_F1": round(pf["micro_F1"], 3),
            "pf_cells_scored": pf["n_cells"],
        },
    )

    # ---- 1. node P/R
    rep.h("1. Node precision / recall (gold = DesignLibrary module set)")
    rep.table(
        ["metric", "value"],
        [
            ["precision", f"{p:.3f}"],
            ["recall", f"{r:.3f}"],
            ["F1", f"{f1:.3f}"],
            ["gold modules", len(gold_mods)],
            ["KG PDK_Cells", len(kg)],
            ["matched", len(matched)],
        ],
    )
    rep.line("")
    rep.line(f"- missing (in library, NOT in KG): {missing or '(none)'}")
    rep.line(f"- extra (in KG, NOT a library module): {extra or '(none)'}")

    # ---- 2. PERFORMS_FUNCTION docstring-grounded P/R
    rep.h("2. PERFORMS_FUNCTION precision / recall (gold = docstring NodeLabels)")
    rep.line(
        f"Gold = each module's NodeLabels minus the structural skip-labels the ingestion "
        f"itself drops ({sorted(PERFORMS_SKIP_LABELS)}); names matched "
        f"normalization-aware. Cells with no asserted function gold are not scored."
    )
    rep.line("")
    rep.table(
        ["view", "P", "R", "F1", "CI (Wilson, on recall-tp/gold)"],
        [
            [
                "micro",
                f"{pf['micro_P']:.3f}",
                f"{pf['micro_R']:.3f}",
                f"{pf['micro_F1']:.3f}",
                _ci(pf["tp"], pf["tp"] + pf["fn"]),
            ],
            ["macro", f"{pf['macro_P']:.3f}", f"{pf['macro_R']:.3f}", "—", "—"],
        ],
    )
    rep.line("")
    rep.line(
        f"- cells scored: **{pf['n_cells']}** of {len(matched)} matched "
        f"(rest assert no function label)"
    )
    rep.line(f"- pooled counts: tp={pf['tp']}, fp={pf['fp']}, fn={pf['fn']}")
    rep.line("")
    rep.line("Per-cell PERFORMS_FUNCTION (gold vs KG-predicted, normalized):")
    rep.table(
        ["module", "gold", "KG pred", "tp", "fp", "fn"],
        [
            [
                c["module"],
                ", ".join(c["gold"]) or "—",
                ", ".join(c["pred"]) or "—",
                c["tp"],
                c["fp"],
                c["fn"],
            ]
            for c in pf["per_cell"]
        ],
    )
    if pf["unmatched"]:
        rep.line("")
        rep.line(
            "**Unmatched gold labels (recall misses) — each is EITHER a genuine "
            "ingestion gap OR a KG vocabulary-normalization mismatch:**"
        )
        for m, miss in sorted(pf["unmatched"].items()):
            rep.line(f"- `{m}`: {miss}")

    # ---- 3. IMPLEMENTS
    rep.h("3. IMPLEMENTS (edge presence + exploratory name proxy)")
    rep.line(
        "Docstrings name no single abstract Component, so a clean name-level IMPLEMENTS gold "
        "is NOT derivable. We report the strong signal — edge PRESENCE — plus the raw KG "
        "IMPLEMENTS targets per cell for inspection. Treat any name-level reading as exploratory."
    )
    rep.line("")
    rep.table(
        ["metric", "value", "CI (Wilson)"],
        [
            [
                "cells with >=1 IMPLEMENTS edge",
                f"{impl_present}/{len(matched)}",
                _ci(impl_present, len(matched)),
            ]
        ],
    )
    rep.line("")
    rep.line("Per-cell IMPLEMENTS targets (KG, raw names):")
    rep.table(
        ["module", "IMPLEMENTS -> Component(name)"],
        [[m, ", ".join(sorted(impl_targets.get(m, set()))) or "—"] for m in matched],
    )

    # ---- 4. ports / params representation
    rep.h("4. Ports / parameters — representation in the KG")
    rep.line(
        "**Finding:** the ingestion stores ports and parameters as PDK_Cell *node properties* "
        "(`ports` string, `parameters`/`port_details` JSON strings), NOT as graph "
        "edges or separate nodes (pdk_ingestion_agent._cell_to_props). There is therefore "
        "**no port/param graph structure to score as edges** — only property coverage and "
        "value agreement against the docstring are measurable."
    )
    rep.line("")
    rep.table(
        ["property", "cells populated", "coverage", "CI (Wilson)"],
        [
            [
                "ports (string)",
                f"{ports_prop}/{len(matched)}",
                f"{ports_prop / len(matched):.2f}" if matched else "—",
                _ci(ports_prop, len(matched)),
            ],
            [
                "parameters (JSON dict)",
                f"{params_prop}/{len(matched)}",
                f"{params_prop / len(matched):.2f}" if matched else "—",
                _ci(params_prop, len(matched)),
            ],
            [
                "port_details (JSON list)",
                f"{portdet_prop}/{len(matched)}",
                f"{portdet_prop / len(matched):.2f}" if matched else "—",
                _ci(portdet_prop, len(matched)),
            ],
        ],
    )
    rep.line("")
    rep.line("Value agreement vs docstring (where docstring asserts the field):")
    rep.table(
        ["check", "agree", "rate", "CI (Wilson)"],
        [
            [
                "ports string == docstring ports",
                f"{ports_agree}/{len(ports_gold_cells)}",
                f"{ports_agree / len(ports_gold_cells):.2f}"
                if ports_gold_cells
                else "—",
                _ci(ports_agree, len(ports_gold_cells)),
            ],
            [
                "param-key set (micro P / R / F1)",
                f"P={pk_p:.2f} R={pk_r:.2f} F1={pk_f:.2f}",
                f"{pk_scored} cells",
                _ci(pk_tp, pk_tp + pk_fn),
            ],
        ],
    )

    # ---- 5. structural edge coverage (kept)
    rep.h("5. Per-cell structural edge-type coverage (>=1 such edge)")
    rep.table(
        ["edge type", "cells with >=1", "coverage", "CI (Wilson)", "required?"],
        [
            [
                et,
                f"{cov[et]}/{len(matched)}",
                f"{cov[et] / len(matched):.2f}" if matched else "—",
                _ci(cov[et], len(matched)),
                "required" if et in REQUIRED else "optional",
            ]
            for et in REQUIRED + OPTIONAL
        ],
    )
    for et in REQUIRED:
        if missing_required[et]:
            rep.line("")
            rep.line(f"- cells missing **{et}**: {missing_required[et]}")

    # ---- caveats
    rep.h("Caveats (read before citing any number)")
    rep.line(
        "1. **Function-name P/R conflates true ingestion error with KG vocabulary mismatch.** "
        "Recall misses are listed per-cell in §2; a name there may be a genuine missing edge OR "
        "a synonym the conservative normalizer (lowercase, drop parenthetical acronyms, collapse "
        "punctuation; NO synonym map) did not unify. Quantify the vocabulary share by inspecting "
        "that list before attributing a shortfall to ingestion."
    )
    rep.line(
        "2. **Gold derives from the PDK authors' own docstrings**, parsed with the SAME parser the "
        "ingestion uses. That is a fair, machine-readable reference, but it is author-asserted "
        "metadata, not an independent oracle — it bounds 'did ingestion preserve what authors wrote', "
        "not absolute physical correctness."
    )
    rep.line(
        "3. **Not measurable as graph structure:** ports and parameters (stored only as node "
        "properties), and a clean name-level IMPLEMENTS gold (no single abstract-Component field in "
        "the docstrings). These are reported as property coverage / presence, explicitly labelled."
    )
    rep.line(
        "4. **PERFORMS_FUNCTION gold excludes** the structural skip-labels "
        f"{sorted(ACTIVE_PASSIVE | {'1x1', '1x2', '2x2', '1x4', '2x4'})} because the ingestion's "
        "resolve_performs_function() drops them before searching Design_Functions; scoring them "
        "would manufacture false recall misses."
    )
    rep.save()


if __name__ == "__main__":
    main()

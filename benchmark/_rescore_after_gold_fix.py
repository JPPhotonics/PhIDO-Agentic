#!/usr/bin/env python3
"""Rescore stored E2 runs against a REVISED gold, in place, with a validation gate.

Why this exists: `benchmark/results/` is gitignored, so a rescore performed ad hoc is not
recoverable from git history. This script is the reproducible record of the 2026-07-30
rescore that followed the crossbar planarity fix to `b3_gold_v2.json` (L4_04 + L4_09: row
links o3->o2, column links o4->o1). Committing this file makes that step repeatable even
though its output artifact is untracked.

Method. For every stored run of the affected prompts it (1) re-scores against the OLD gold
and asserts the result reproduces the edgeF1/ged already in the results file, which proves
the scoring path here is the same one that produced them, then (2) re-scores against the
NEW gold and writes the updated values back. Scoring uses the project's own modules
(`b3_eval._entry_to_topology`, `topology_eval.score_topology`), never a reimplementation.
Component-F1 is untouched by a port-convention change and is left alone.

Usage (from the repo root, with the venv python):
    .venv/bin/python benchmark/_rescore_after_gold_fix.py \
        --old benchmark/b3_gold_v2.json.pre_planar_crossbar_fix.bak \
        --new benchmark/b3_gold_v2.json \
        --results benchmark/results/_ablation_correctness_gpt54_v2.json \
        --prompts L4_04 L4_09 [--dry-run]

A backup is written next to the results file unless --no-backup is passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from b3_eval import _entry_to_topology  # noqa: E402
from topology_eval import Topology, score_topology  # noqa: E402


def load_gold(path: Path) -> dict:
    return {e["id"]: e for e in json.loads(Path(path).read_text())["gold"]}


def score(run: dict, gold_entry: dict) -> tuple[float | None, float]:
    pred = Topology(
        nodes=dict(run["pred_node_map"]),
        edges=[frozenset(e) for e in run["pred_edges"]],
        external={},
    )
    s = score_topology(pred, _entry_to_topology(gold_entry))
    ef = s["typed_edge_prf"]["f1"]
    return (None if ef is None else round(ef, 3)), s["approx_ged"]["ged"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="gold BEFORE the fix (validation gate)")
    ap.add_argument("--new", required=True, help="gold AFTER the fix (authoritative)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    gold_old, gold_new = load_gold(args.old), load_gold(args.new)
    rpath = Path(args.results)
    res = json.loads(rpath.read_text())

    changed, checked = [], 0
    for pid in args.prompts:
        for arm in res:
            for rep, run in enumerate(res[arm].get(pid, []), start=1):
                if run.get("status") != "scored":
                    continue
                old_ef, old_ged = score(run, gold_old[pid])
                assert old_ef == run["edgeF1"] and old_ged == run["ged"], (
                    f"VALIDATION FAILED on {arm}::{pid}::rep{rep}: recomputed "
                    f"({old_ef}, {old_ged}) != stored ({run['edgeF1']}, {run['ged']}). "
                    "The scoring path does not reproduce the stored values; aborting "
                    "rather than overwriting them."
                )
                checked += 1
                new_ef, new_ged = score(run, gold_new[pid])
                if (new_ef, new_ged) != (old_ef, old_ged):
                    changed.append((f"{arm}::{pid}::rep{rep}", old_ef, new_ef,
                                    old_ged, new_ged))
                run["edgeF1"], run["ged"] = new_ef, new_ged

    print(f"validation gate passed on all {checked} scored runs")
    print(f"{len(changed)} run(s) changed:")
    for key, oe, ne, og, ng in changed:
        print(f"  {key:38s} edgeF1 {oe} -> {ne}   ged {og} -> {ng}")

    if args.dry_run:
        print("\n--dry-run: results file NOT written")
        return
    if not args.no_backup:
        bak = rpath.with_suffix(rpath.suffix + ".pre_planar_gold_fix.bak")
        if not bak.exists():
            shutil.copy2(rpath, bak)
            print(f"backup -> {bak}")
    rpath.write_text(json.dumps(res, indent=1))
    print(f"wrote {rpath}")


if __name__ == "__main__":
    main()

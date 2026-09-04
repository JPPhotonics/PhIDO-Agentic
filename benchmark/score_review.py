"""Aggregate the blind 3-axis human review into per-arm / per-level scores + auto-metric agreement.

Consumes the labels exported by ``_ablation_review.html`` (Export button) and the ablation results
JSON (for the arm/level/auto-metric join, since the UI grades blind). Emits a Markdown report.

Axes → scalar:  A1/A2 {correct:1, minor/partial:.5, wrong:0} ;  A3 {yes:1, partial:.5, no:0}.
Pipeline-fails (no circuit) that were graded count as their given labels; if ungraded they are
listed separately (they cannot be silently dropped — that would flatter reliability).

Usage:
    python benchmark/score_review.py LABELS.json \
        [--results benchmark/results/_ablation_correctness_gpt54_v2.json] \
        [--out benchmark/results/_ablation_review_report.md]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ["A1", "A2", "A3"]
SCALAR = {"correct": 1.0, "yes": 1.0, "minor": 0.5, "partial": 0.5, "wrong": 0.0, "no": 0.0}


def parse_rid(rid: str):
    arm, pid, rep = rid.split("::")
    return arm, pid, int(rep.replace("rep", ""))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def spearman(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        return round(float(rho), 3), round(float(p), 4), len(pairs)
    except Exception:  # noqa: BLE001
        return None


def build(labels_path: Path, results_path: Path, out_path: Path):
    labels = json.loads(labels_path.read_text())
    ann = labels.get("annotations", labels)
    results = json.loads(results_path.read_text())

    def rget(arm, pid, rep):
        try:
            return results[arm][pid][rep - 1]
        except (KeyError, IndexError):
            return {}

    rows = []
    for rid, a in ann.items():
        axes = a.get("axes", {})
        if not all(axes.get(x) for x in AXES):
            continue  # incomplete; excluded from scored aggregates, counted below
        arm, pid, rep = parse_rid(rid)
        rec = rget(arm, pid, rep)
        rows.append({
            "arm": arm, "pid": pid, "rep": rep,
            "level": rec.get("level_paper") or rec.get("level"),
            "A1": SCALAR[axes["A1"]], "A2": SCALAR[axes["A2"]], "A3": SCALAR[axes["A3"]],
            "A1raw": axes["A1"], "A2raw": axes["A2"], "A3raw": axes["A3"],
            "tags": a.get("tags", {}),
            "compF1": rec.get("compF1"), "edgeF1": rec.get("edgeF1"), "ged": rec.get("ged"),
            "status": rec.get("status"),
        })

    n_total = len({parse_rid(r)[0] + r for r in ann}) if ann else 0
    graded = len(rows)
    incomplete = len(ann) - graded

    L = ["# E2 ablation — blind 3-axis human review", "",
         f"**Labels:** `{labels_path.name}` · **Results join:** `{results_path.name}`",
         f"**Complete runs scored:** {graded}  ·  incomplete/skipped: {incomplete}", "",
         "Axis scalars: A1/A2 correct=1, minor/partial=0.5, wrong=0; A3 yes=1, partial=0.5, no=0.",
         "A1 = component fidelity, A2 = connectivity fidelity, A3 = intent satisfaction.", ""]

    # ---- per-arm means (all levels) ----
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    L += ["## Per-arm human scores (all levels)", "",
          "| arm | n | A1 | A2 | A3 | intent=Yes |"]
    L.append("|---|---|---|---|---|---|")
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        yes = round(sum(x["A3raw"] == "yes" for x in rs) / len(rs), 3)
        L.append(f"| {arm} | {len(rs)} | {mean([x['A1'] for x in rs])} | "
                 f"{mean([x['A2'] for x in rs])} | {mean([x['A3'] for x in rs])} | {yes} |")

    # ---- per-arm x level ----
    for axis in AXES:
        L += ["", f"### {axis} by arm × level", "", "| arm | L3 | L4 |", "|---|---|---|"]
        for arm in sorted(by_arm):
            def lvl(n):
                return mean([x[axis] for x in by_arm[arm] if x["level"] == n])
            L.append(f"| {arm} | {lvl(3)} | {lvl(4)} |")

    # ---- human vs automatic agreement ----
    L += ["", "## Human ↔ automatic-metric agreement (Spearman ρ)", "",
          "Tests whether the F1/GED metrics track the human judgment. Divergence at a given axis is "
          "where the automatic metric misleads.", "", "| pair | ρ | p | n |", "|---|---|---|---|"]
    for hax, auto, sign in [("A1", "compF1", 1), ("A2", "edgeF1", 1), ("A3", "edgeF1", 1),
                            ("A3", "ged", -1)]:
        sp = spearman([r[hax] for r in rows], [(sign * r[auto]) if r[auto] is not None else None
                                               for r in rows])
        L.append(f"| {hax} vs {auto} | {sp[0] if sp else 'n/a'} | {sp[1] if sp else '—'} | "
                 f"{sp[2] if sp else 0} |")

    # ---- compF1 degeneracy witness: compF1=1.0 but human says wiring/intent wrong ----
    deg = [r for r in rows if r["compF1"] == 1.0 and (r["A2raw"] == "wrong" or r["A3raw"] == "no")]
    L += ["", "## compF1 degeneracy witnesses",
          f"Runs where **compF1 = 1.0** yet human graded connectivity=Wrong or intent=No: "
          f"**{len(deg)}**.", ""]
    if deg:
        L += ["| pid | arm | edgeF1 | A2 | A3 |", "|---|---|---|---|---|"]
        for r in sorted(deg, key=lambda x: (x["level"] or 0, x["pid"])):
            L.append(f"| {r['pid']} | {r['arm']} | {r['edgeF1']} | {r['A2raw']} | {r['A3raw']} |")

    # ---- error-nature tag frequency ----
    tagfreq = defaultdict(int)
    for r in rows:
        for ax, tags in r["tags"].items():
            for t in tags or []:
                tagfreq[f"{ax}:{t}"] += 1
    if tagfreq:
        L += ["", "## Error-nature tag frequency", "", "| tag | count |", "|---|---|"]
        for k, v in sorted(tagfreq.items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v} |")

    out_path.write_text("\n".join(L) + "\n")
    print(f"Wrote {out_path}  ({graded} scored, {incomplete} incomplete, {len(deg)} degeneracy witnesses)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels")
    ap.add_argument("--results", default="benchmark/results/_ablation_correctness_gpt54_v2.json")
    ap.add_argument("--out", default="benchmark/results/_ablation_review_report.md")
    args = ap.parse_args()
    build(Path(args.labels), ROOT / args.results, ROOT / args.out)


if __name__ == "__main__":
    main()

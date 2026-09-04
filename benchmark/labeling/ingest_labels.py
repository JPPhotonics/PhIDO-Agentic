#!/usr/bin/env python3
"""Analyse returned faithfulness labels.

Reads the annotator-returned ``labels_<id>.json`` files and the private
``bundle_meta_<id>.json`` files, joins them by triple id, and reports the metrics the
benchmark plan specifies for the small clustered expert set:

* per-paper (per-document) faithfulness rate — **always descriptive**;
* pooled rate two ways: naive Wilson interval (reference) and the honest
  **cluster-level t-interval** (clusters = documents, df = #docs − 1) — exploratory,
  only meaningful with enough clusters;
* the design effect / ICC, so you can see how much clustering inflates uncertainty;
* if a judge-label file is given: **expert↔judge agreement + Cohen's κ**, overall and on
  the **out-of-ontology** subset (the load-bearing validation number);
* cross-annotator agreement on the shared overlap set;
* timing sanity (median seconds/item, flags rushed annotators).

No headline rests on the pooled CI — see KB_BENCHMARK_PLAN.md §5c.

Usage::

    python ingest_labels.py --dir bundles [--judge judge_labels.json] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Two-sided 95% Student-t critical values; falls back to z=1.96 for large df.
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    return 1.96 if df > 30 else _T95[min(_T95, key=lambda k: abs(k - df))]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (centre - half, centre + half)


def icc_oneway(groups: list[list[int]]) -> tuple[float, float]:
    """One-way random-effects ICC(1) and mean cluster size, from 0/1 group data."""
    groups = [g for g in groups if g]
    k = len(groups)
    if k < 2:
        return (float("nan"), float("nan"))
    n_i = [len(g) for g in groups]
    total_n = sum(n_i)
    grand = sum(sum(g) for g in groups) / total_n
    msb = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    within = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
    msw = within / (total_n - k) if total_n > k else 0.0
    m0 = (total_n - sum(n * n for n in n_i) / total_n) / (k - 1)
    denom = msb + (m0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    return (max(0.0, min(1.0, icc)), m0)


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's κ for two raters over binary judgments."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _to01(label: Any) -> int | None:
    """Map a label to 1 (supported), 0 (not), or None (skip/missing)."""
    if label in (1, "1", "supported", "support", "faithful", "yes", True):
        return 1
    if label in (0, "0", "not_supported", "unfaithful", "no", False):
        return 0
    return None  # skip / unsure / unlabelled


def load_dir(d: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return (labels_by_id, meta_by_id) merged across all annotators in the dir."""
    labels: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for mf in d.glob("bundle_meta_*.json"):
        m = json.loads(mf.read_text())
        for tid, info in m.get("items", {}).items():
            meta[tid] = {**info, "annotator_id": m["annotator_id"]}
    for lf in d.glob("labels_*.json"):
        payload = json.loads(lf.read_text())
        aid = payload.get("annotator_id", lf.stem.replace("labels_", ""))
        for row in payload.get("labels", []):
            tid = row["id"]
            # keep one row per (annotator, triple) so overlap items survive per annotator
            labels[f"{aid}::{tid}"] = {**row, "annotator_id": aid, "triple_id": tid}
    return labels, meta


def analyse(d: Path, judge_path: Path | None) -> dict[str, Any]:
    labels, meta = load_dir(d)
    report: dict[str, Any] = {}

    # ---- per-document faithfulness rate (descriptive) ----
    by_doc: dict[str, list[int]] = defaultdict(list)
    by_annot_time: dict[str, list[float]] = defaultdict(list)
    skips = 0
    for row in labels.values():
        v = _to01(row.get("label"))
        if row.get("time_ms"):
            by_annot_time[row["annotator_id"]].append(row["time_ms"] / 1000.0)
        if v is None:
            skips += 1
            continue
        doc = row.get("source_document") or meta.get(row["triple_id"], {}).get("source_document", "?")
        by_doc[doc].append(v)

    per_doc = {doc: {"n": len(g), "rate": statistics.mean(g)} for doc, g in by_doc.items() if g}
    report["per_document"] = per_doc
    report["total_judged"] = sum(len(g) for g in by_doc.values())
    report["total_skipped"] = skips

    # ---- pooled: naive Wilson vs honest cluster t-interval ----
    all01 = [v for g in by_doc.values() for v in g]
    if all01:
        k1 = sum(all01)
        lo, hi = wilson(k1, len(all01))
        report["pooled_naive_wilson"] = {"rate": statistics.mean(all01),
                                         "ci95": [clamp01(lo), clamp01(hi)],
                                         "note": "ignores clustering -> too narrow; reference only"}
    cluster_means = [statistics.mean(g) for g in by_doc.values() if g]
    kcl = len(cluster_means)
    if kcl >= 2:
        m = statistics.mean(cluster_means)
        sd = statistics.stdev(cluster_means)
        half = t95(kcl - 1) * sd / math.sqrt(kcl)
        icc, m0 = icc_oneway(list(by_doc.values()))
        deff = 1 + (m0 - 1) * icc if not math.isnan(m0) else float("nan")
        report["pooled_cluster_t"] = {
            "rate": m, "ci95": [clamp01(m - half), clamp01(m + half)], "n_clusters": kcl, "df": kcl - 1,
            "icc": icc, "mean_cluster_size": m0, "design_effect": deff,
            "effective_n": (len(all01) / deff) if deff and not math.isnan(deff) else None,
            "note": "EXPLORATORY: clusters=documents; fragile below ~10 clusters",
        }
    else:
        report["pooled_cluster_t"] = {"note": f"only {kcl} cluster(s); no clustered CI possible"}

    # ---- expert <-> judge agreement (the validation number) ----
    if judge_path and judge_path.exists():
        judge_raw = json.loads(judge_path.read_text())
        judge = {tid: _to01(lab) for tid, lab in judge_raw.items()}
        pairs_all, pairs_ooo = [], []
        for row in labels.values():
            e = _to01(row.get("label"))
            j = judge.get(row["triple_id"])
            if e is None or j is None:
                continue
            pairs_all.append((e, j))
            if meta.get(row["triple_id"], {}).get("is_out_of_ontology"):
                pairs_ooo.append((e, j))
        report["judge_validation"] = {
            "overall": {"n": len(pairs_all),
                        "agreement": (sum(1 for a, b in pairs_all if a == b) / len(pairs_all)) if pairs_all else None,
                        "cohen_kappa": cohen_kappa(pairs_all)},
            "out_of_ontology": {"n": len(pairs_ooo),
                                "agreement": (sum(1 for a, b in pairs_ooo if a == b) / len(pairs_ooo)) if pairs_ooo else None,
                                "cohen_kappa": cohen_kappa(pairs_ooo),
                                "note": "primary: validates judge where the ontology cannot"},
        }

    # ---- cross-annotator agreement on the shared overlap set ----
    overlap_by_triple: dict[str, list[int]] = defaultdict(list)
    for row in labels.values():
        if meta.get(row["triple_id"], {}).get("is_overlap"):
            v = _to01(row.get("label"))
            if v is not None:
                overlap_by_triple[row["triple_id"]].append(v)
    multi = {t: vs for t, vs in overlap_by_triple.items() if len(vs) >= 2}
    if multi:
        agree = sum(1 for vs in multi.values() if len(set(vs)) == 1) / len(multi)
        report["overlap_agreement"] = {"n_items": len(multi), "unanimous_fraction": agree,
                                       "note": "human-human consistency spot-check"}
    else:
        report["overlap_agreement"] = {"note": "no overlap items labelled by >=2 annotators"}

    # ---- timing sanity ----
    report["timing"] = {
        aid: {"n": len(ts), "median_s": round(statistics.median(ts), 1),
              "rushed": statistics.median(ts) < 2.0}
        for aid, ts in by_annot_time.items() if ts
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True, help="dir with bundle_meta_*.json + returned labels_*.json")
    ap.add_argument("--judge", type=Path, default=None, help="optional judge_labels.json: {triple_id: label}")
    ap.add_argument("--out", type=Path, default=None, help="write the report JSON here")
    args = ap.parse_args()

    report = analyse(args.dir, args.judge)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[ok] report written to {args.out}")


if __name__ == "__main__":
    main()

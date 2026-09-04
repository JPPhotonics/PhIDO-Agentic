"""L1 — reference-free KG health: link-prediction recovery vs a corrupted control. READ-ONLY.

Pulls the KG's typed triples and runs `lp_measure.health_report`: de-leak (Akrami), then LP
recovery (MRR + Hit@k) on the clean graph vs a 70%-corrupted control. The meaningful number is
the SEPARATION (clean MRR − corrupted MRR): a structurally-consistent graph recovers held-out
edges better than a randomly-rewired one; a near-zero gap means the graph carries little
recoverable regularity.

SCOPE (state honestly): this measures internal consistency / redundancy / structural regularity,
NOT correctness or faithfulness — a graph can be self-consistent and still wrong. Secondary L1
sanity check only. The default predictor is relation-conditioned tail frequency (a weak baseline),
so absolute MRR is not meaningful on its own; report the clean−corrupted gap.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_l1_health.py
"""

from __future__ import annotations

import pathlib

from kb_access import KB
from lp_measure import health_report
from report import Reporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "benchmark" / "results" / "l1_health.md"
# generic escape-hatch / provenance edges add noise without structural signal
EXCLUDE_RELS = {"EXTRACTED_FROM"}


def kb_triples(kb: KB) -> list[tuple]:
    rows = kb.q(
        "MATCH (a)-[e]->(b) "
        "WITH coalesce(a.name, a.module_name) AS h, type(e) AS r, coalesce(b.name, b.module_name) AS t "
        "WHERE h IS NOT NULL AND t IS NOT NULL "
        "RETURN h, r, t"
    )
    return [(x["h"], x["r"], x["t"]) for x in rows if x["r"] not in EXCLUDE_RELS]


def main() -> None:
    kb = KB()
    triples = kb_triples(kb)
    kb.close()

    rep_data = health_report(triples)
    dl, clean, corr = (
        rep_data["de_leak"],
        rep_data["clean"],
        rep_data["corrupted_control"],
    )
    sep = rep_data["separation_mrr"]

    rep = Reporter(
        OUT_MD,
        "L1 — reference-free KG health (link-prediction recovery)",
        meta={
            "triples_raw": len(triples),
            "triples_after_deleak": dl.get("n_after"),
            "separation_mrr": round(sep, 4),
        },
    )

    rep.h("Verdict")
    healthy = sep > 0.02
    rep.line(
        f"- **clean MRR {clean['mrr']:.3f}  vs  corrupted-control MRR {corr['mrr']:.3f}  "
        f"→ separation {sep:+.3f}**"
    )
    rep.line(
        f"- {'PASS' if healthy else 'WEAK'}: the graph recovers held-out edges "
        f"{'better than' if healthy else 'no better than'} a randomly-rewired control "
        f"→ {'has' if healthy else 'little'} recoverable structural regularity."
    )

    rep.h("De-leak (Akrami — removes trivial symmetric/duplicate inflation)")
    rep.table(
        ["metric", "value"],
        [
            ["triples before", dl.get("n_before")],
            ["triples after", dl.get("n_after")],
            ["removed exact duplicates", dl.get("removed_exact_duplicates")],
            ["removed symmetric", dl.get("removed_symmetric")],
        ],
    )

    rep.h("LP recovery: clean vs corrupted control")
    ks = [k for k in clean if k.startswith("hit@")]
    rep.table(
        ["graph", "n_test", "MRR"] + ks,
        [
            ["clean", clean["n_test"], round(clean["mrr"], 3)]
            + [round(clean[k], 3) for k in ks],
            ["corrupted (70%)", corr["n_test"], round(corr["mrr"], 3)]
            + [round(corr[k], 3) for k in ks],
        ],
    )

    rep.line("")
    rep.line(
        "_Measures internal consistency/redundancy, NOT correctness or faithfulness. The "
        "predictor is a weak frequency baseline → read the clean−corrupted gap, not absolute MRR._"
    )
    rep.save()


if __name__ == "__main__":
    main()

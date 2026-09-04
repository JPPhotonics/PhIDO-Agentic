"""A4 — SEA (Schema Evolution Agent) convergence / stability. READ-ONLY diagnostic.

Per BENCHMARK_ARCHITECTURE.md §5, A4's PRIMARY metric is schema convergence/stability and is
"fully automatable, no labels required": type-growth curve, churn, and threshold sensitivity. This
harness reconstructs those from the KB state SEA already produced — `RelationshipObservation` nodes
(each carries proposed_type, source_document, confidence, cluster_id) and `SchemaRelationType`
nodes (seed vs promoted) — WITHOUT re-running SEA (which mutates the graph + calls the LLM).

The SEA promotion gate (sea_agent.py): a cluster promotes to a new type only if it spans
≥ PROMOTION_THRESHOLD (3) distinct source documents AND mean confidence ≥ MIN_CONFIDENCE_MEAN
(0.70) [+ directional consistency + LLM validation]. We compute the statistical part per cluster
to explain the observed promotion count, and sweep PROMOTION_THRESHOLD to show sensitivity.

The write-heavy convergence EXPERIMENT (re-run evolve() k times / in multiple ingestion orders to
measure determinism + order-invariance) is gated behind `run_convergence_experiment()` — it must
NOT touch the shared KB and is not invoked by this diagnostic.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/a4_sea_convergence.py
"""

from __future__ import annotations

import pathlib

from kb_access import KB
from report import Reporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "benchmark" / "results" / "a4_sea_convergence.md"

# SEA gate constants (sea_agent.py) — mirrored here for the read-only what-if; source of truth is the agent
PROMOTION_THRESHOLD = 3
MIN_CONFIDENCE_MEAN = 0.70


def cluster_stats(kb: KB) -> list[dict]:
    """Per-cluster: distinct source docs, mean confidence, dominant proposed_type, size."""
    rows = kb.q(
        "MATCH (o:RelationshipObservation) WHERE o.cluster_id IS NOT NULL "
        "RETURN o.cluster_id AS cid, o.proposed_type AS t, o.source_document AS d, o.confidence AS c"
    )
    by: dict[str, dict] = {}
    for r in rows:
        c = by.setdefault(r["cid"], {"docs": set(), "types": {}, "confs": [], "n": 0})
        c["docs"].add(r["d"])
        c["types"][r["t"]] = c["types"].get(r["t"], 0) + 1
        c["confs"].append(r["c"] if r["c"] is not None else 0.0)
        c["n"] += 1
    out = []
    for cid, c in by.items():
        dominant = max(c["types"], key=c["types"].get)
        out.append(
            {
                "cid": cid,
                "n_obs": c["n"],
                "n_docs": len(c["docs"]),
                "mean_conf": sum(c["confs"]) / len(c["confs"]),
                "dominant_type": dominant,
                "n_types_in_cluster": len(c["types"]),
            }
        )
    return out


def type_growth(kb: KB) -> list[dict]:
    """Cumulative distinct proposed_type as documents are added in observation-time order."""
    rows = kb.q(
        "MATCH (o:RelationshipObservation) "
        "RETURN o.source_document AS d, min(o.observed_at) AS first, "
        "collect(DISTINCT o.proposed_type) AS types ORDER BY first"
    )
    seen: set[str] = set()
    curve = []
    for i, r in enumerate(rows, 1):
        seen |= set(r["types"])
        curve.append(
            {
                "doc_index": i,
                "document": (r["d"] or "")[:32],
                "new_types_this_doc": len(set(r["types"]) - (seen - set(r["types"]))),
                "cumulative_distinct_types": len(seen),
            }
        )
    return curve


def main() -> None:
    kb = KB()
    n_obs = kb.scalar("MATCH (o:RelationshipObservation) RETURN count(*)") or 0
    n_proposed = (
        kb.scalar(
            "MATCH (o:RelationshipObservation) RETURN count(DISTINCT o.proposed_type)"
        )
        or 0
    )
    n_docs = (
        kb.scalar(
            "MATCH (o:RelationshipObservation) RETURN count(DISTINCT o.source_document)"
        )
        or 0
    )
    seed = (
        kb.scalar("MATCH (s:SchemaRelationType) WHERE s.is_seed RETURN count(*)") or 0
    )
    promoted_rows = kb.q(
        "MATCH (s:SchemaRelationType) WHERE NOT s.is_seed RETURN s.name AS n"
    )
    promoted = [r["n"] for r in promoted_rows]

    clusters = cluster_stats(kb)
    curve = type_growth(kb)

    # per proposed_type doc-spread (churn): how many distinct docs each candidate type appears in
    spread_rows = kb.q(
        "MATCH (o:RelationshipObservation) "
        "RETURN o.proposed_type AS t, count(DISTINCT o.source_document) AS docs ORDER BY docs DESC"
    )
    singletons = [r["t"] for r in spread_rows if r["docs"] == 1]
    kb.close()

    # statistical promotion gate (pre-LLM): clusters meeting docs>=PROMOTION_THRESHOLD and mean_conf>=MIN
    passes = [
        c
        for c in clusters
        if c["n_docs"] >= PROMOTION_THRESHOLD and c["mean_conf"] >= MIN_CONFIDENCE_MEAN
    ]
    sweep = {
        t: sum(
            1
            for c in clusters
            if c["n_docs"] >= t and c["mean_conf"] >= MIN_CONFIDENCE_MEAN
        )
        for t in (1, 2, 3, 4, 5)
    }

    rep = Reporter(
        OUT_MD,
        "A4 — SEA schema convergence / stability (read-only diagnostic)",
        meta={
            "observations": n_obs,
            "distinct_proposed_types": n_proposed,
            "documents": n_docs,
            "seed_types": seed,
            "promoted_types": len(promoted),
            "clusters": len(clusters),
        },
    )

    rep.h("Convergence verdict")
    rep.line(
        f"- **{n_proposed} distinct relationship types proposed** across {n_docs} documents → "
        f"**{len(promoted)} promoted** (schema kept {seed} seed types)."
    )
    rep.line(f"- Promoted: {promoted or '(none)'}")
    verdict = (
        "CONVERGED — proposals collapse onto the seed vocabulary; no schema drift"
        if not promoted
        else f"{len(promoted)} new type(s) admitted"
    )
    rep.line(f"- **Verdict: {verdict}.**")

    rep.h("Why: statistical promotion gate (docs ≥ 3 AND mean-confidence ≥ 0.70)")
    rep.line(
        f"- clusters meeting the statistical gate (pre-LLM): **{len(passes)}** of {len(clusters)}"
    )
    if passes:
        rep.table(
            ["dominant_type", "n_docs", "mean_conf", "n_obs"],
            [
                [c["dominant_type"], c["n_docs"], round(c["mean_conf"], 2), c["n_obs"]]
                for c in passes
            ],
        )
    else:
        rep.line(
            "- → no cluster spans ≥3 documents at ≥0.70 confidence, so 0 promotions is the "
            "**correct** gate behavior (not a missed promotion)."
        )

    rep.h(
        "Promotion-threshold sensitivity (clusters passing at each PROMOTION_THRESHOLD)"
    )
    rep.table(
        ["threshold (distinct docs)", "clusters passing"],
        [[t, sweep[t]] for t in (1, 2, 3, 4, 5)],
    )

    rep.h("Churn — proposed types confined to a single document")
    rep.line(
        f"- {len(singletons)}/{n_proposed} proposed types appear in exactly ONE document "
        f"(transient, below any multi-doc promotion bar)."
    )
    rep.line(f"- examples: {singletons[:12]}")

    rep.h("Type-growth curve (cumulative distinct proposed types over documents)")
    rep.table(
        ["doc #", "document", "cumulative distinct types"],
        [
            [c["doc_index"], c["document"], c["cumulative_distinct_types"]]
            for c in curve
        ],
    )
    rep.save()


def run_convergence_experiment(*_a, **_k):
    """GATED, write+LLM-heavy: re-run SEA.evolve() k times / in multiple ingestion orders on a
    SANDBOX KB to measure determinism + order-invariance (BENCHMARK_ARCHITECTURE A4 secondary,
    pilot-gated). Intentionally NOT implemented against the shared KB to avoid mutating another
    session's graph. To run: stand up an isolated Neo4j, ingest a fixed paper subset in k orders,
    call evolve() each time, and diff the resulting SchemaRelationType sets + recategorization logs."""
    raise NotImplementedError(
        "A4 convergence experiment is write/LLM-heavy and must run on an isolated sandbox KB — "
        "not invoked by the read-only diagnostic. See docstring for the procedure."
    )


if __name__ == "__main__":
    main()

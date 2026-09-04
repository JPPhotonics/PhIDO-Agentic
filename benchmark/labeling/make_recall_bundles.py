#!/usr/bin/env python3
"""Build RECALL labeling bundles from DIA's *rejected/demoted* relationships.

Companion to ``make_labeling_bundles.py`` (which builds PRECISION bundles from COMMITTED
edges). This script samples the **discarded** edges — ``:ReviewItem`` nodes — for each
annotator's paper and renders them with the *identical* question and format as the
precision bundle, so the two can be interleaved/blinded at send time.

What this measures (and what it does NOT):
  * It estimates **gate recall** = of the relationships DIA PROPOSED then REJECTED, how many
    the expert judges valid (= DIA false negatives). It is **conditional on having been
    proposed** and is an UPPER BOUND on system recall — it is blind to relationships that
    were never proposed at all (the R_extract term). See A2_FAITHFULNESS_FINDINGS.md.
  * The reject pool is 100% RELATED_TO (DIA only gates its global-inference proposals), so
    this is RELATED_TO-inference-gate recall specifically.

Design choices baked in:
  * The ReviewItem ``description`` begins with "Rejected by semantic verifier: …"; that
    verdict is SUPPRESSED (reasoning="") so it can't anchor the annotator.
  * Sampling prioritizes ``generic_related_to_demoted`` (the A+B demotion audit) then fills
    with ``semantic_verification_rejected``; ``reason`` is recorded per item in the meta.
  * A companion ``recall_judge_labels.json`` marks every edge "not_supported" (DIA's verdict)
    -> ingest_labels.py --judge; an expert "valid" verdict is a DIA false negative.

Usage:  CUDA_VISIBLE_DEVICES="" python make_recall_bundles.py bundles_config.json
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path
from typing import Any

from make_labeling_bundles import (  # reuse the precision-bundle machinery
    HERE,
    DEFAULT_INSTRUCTIONS,
    load_relation_definitions,
    render_html,
)

# Three discarded-edge populations, in audit-priority order:
#   schema_nonconformant_typed_demoted — TYPED edges demoted because their endpoint labels violated
#     the schema signature (the new (B2) gate). High-value: the DIA relation reasoning is often
#     CORRECT and only the entity TYPE is wrong, so an expert "valid" here flags an upstream
#     mistyping rather than a relation error. This is a typed-edge audit, NOT RELATED_TO-gate recall.
#   generic_related_to_demoted — the (B) A+B demotion audit (RELATED_TO).
#   semantic_verification_rejected — DIA's semantic verifier rejections (RELATED_TO).
REJECT_REASONS = (
    "schema_nonconformant_typed_demoted",
    "generic_related_to_demoted",
    "semantic_verification_rejected",
)


def _parse_quotes(raw: Any) -> list[str]:
    """ReviewItem evidence_quotes is stored as a stringified list; recover it safely."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if x]
    s = str(raw).strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            v = parse(s)
            if isinstance(v, list):
                return [x for x in v if x]
            if isinstance(v, str) and v:
                return [v]
        except Exception:
            continue
    return [s] if s else []


def fetch_rejected(session: Any, source_document: str) -> list[dict[str, Any]]:
    """All rejected/demoted ReviewItem edges for a document, newest payload fields parsed."""
    q = (
        "MATCH (n:ReviewItem) "
        "WHERE n.source_document = $doc AND n.reason IN $reasons "
        "RETURN elementId(n) AS id, n.payload_json AS payload, n.reason AS reason"
    )
    out: list[dict[str, Any]] = []
    for rec in session.run(q, doc=source_document, reasons=list(REJECT_REASONS)):
        try:
            p = json.loads(rec["payload"])
        except Exception:
            continue
        out.append(
            {
                "id": rec["id"],
                "reason": rec["reason"],
                "head": p.get("from_node"),
                "head_type": p.get("from_collection"),
                "relation": p.get("edge_collection"),
                "tail": p.get("to_node"),
                "tail_type": p.get("to_collection"),
                "evidence_quotes": _parse_quotes(p.get("evidence_quotes")),
                "confidence": p.get("confidence"),
                "reasoning": "",  # SUPPRESSED: payload description reveals the reject verdict
                "source_document": source_document,
            }
        )
    return out


def fetch_descriptions(session: Any, names: set[str]) -> dict[str, str]:
    """Best-effort head/tail descriptions by entity name, for parity with precision bundles."""
    if not names:
        return {}
    rows = session.run(
        "MATCH (n) WHERE n.name IN $names AND n.description IS NOT NULL "
        "RETURN n.name AS name, n.description AS d",
        names=list(names),
    )
    return {r["name"]: r["d"] for r in rows}


def select_rejected(triples: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    """Prioritize the targeted demotion audits, then fill with semantic rejections.

    Order: schema-nonconformant typed demotions (smallest, highest-value — audits the new (B2)
    gate) → generic RELATED_TO demotions (the (B) audit) → semantic-verifier rejections (fill).
    """
    schema_dem = [t for t in triples if t["reason"] == "schema_nonconformant_typed_demoted"]
    demoted = [t for t in triples if t["reason"] == "generic_related_to_demoted"]
    semrej = [t for t in triples if t["reason"] == "semantic_verification_rejected"]
    for bucket in (schema_dem, demoted, semrej):
        rng.shuffle(bucket)
    chosen = (schema_dem + demoted + semrej)[:cap]
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python make_recall_bundles.py config.json")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cap = int(cfg.get("cap", 40))
    rng = random.Random(cfg.get("seed", 0))
    out_dir = Path(cfg.get("out_dir", HERE / "bundles"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient  # type: ignore

    client = Neo4jClient()
    client.connect()
    judge_labels: dict[str, str] = {}

    with client.driver.session() as session:
        relation_defs = load_relation_definitions(session)
        for ann in cfg["annotators"]:
            aid, doc = ann["annotator_id"], ann["source_document"]
            pool = fetch_rejected(session, doc)
            chosen = select_rejected(pool, cap, rng)

            names = {t["head"] for t in chosen} | {t["tail"] for t in chosen}
            descs = fetch_descriptions(session, {n for n in names if n})
            for t in chosen:
                t["head_description"] = descs.get(t["head"], "")
                t["tail_description"] = descs.get(t["tail"], "")

            html = render_html(aid, chosen, relation_defs)
            (out_dir / f"labeler_{aid}_recall.html").write_text(html, encoding="utf-8")

            for t in chosen:
                judge_labels[t["id"]] = "not_supported"  # DIA rejected -> expert "valid" = false neg

            meta = {
                "annotator_id": aid,
                "source_document": doc,
                "arm": "recall",
                "n_available": len(pool),
                "n_shown": len(chosen),
                "items": {
                    t["id"]: {
                        "reason": t["reason"],
                        "relation": t["relation"],
                        "confidence": t.get("confidence"),
                        "source_document": doc,
                    }
                    for t in chosen
                },
            }
            (out_dir / f"bundle_meta_{aid}_recall.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            n_schema = sum(1 for t in chosen if t["reason"] == "schema_nonconformant_typed_demoted")
            n_dem = sum(1 for t in chosen if t["reason"] == "generic_related_to_demoted")
            n_sem = sum(1 for t in chosen if t["reason"] == "semantic_verification_rejected")
            print(f"[ok] {aid}: {len(chosen)} rejected ({doc}) "
                  f"[schema_typed={n_schema}, related_to_demoted={n_dem}, semantic_rejected={n_sem}, "
                  f"avail={len(pool)}] -> labeler_{aid}_recall.html")

    (HERE / "recall_judge_labels.json").write_text(json.dumps(judge_labels, indent=2), encoding="utf-8")
    client.close() if hasattr(client, "close") else None
    print(f"\n{len(judge_labels)} edges -> recall_judge_labels.json (all 'not_supported').")
    print("NOTE: pool is mostly RELATED_TO (gate recall) + a few schema-nonconformant TYPED demotions "
          "(typed-mistyping audit, label separately by reason); blind/interleave with precision at send.")


if __name__ == "__main__":
    main()

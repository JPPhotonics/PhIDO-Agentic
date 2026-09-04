#!/usr/bin/env python3
"""Merged BLIND labeling bundles: precision + recall interleaved into ONE file per annotator.

Per ``benchmark/A2_FAITHFULNESS_FINDINGS.md`` ("interleave discarded edges **blind** into the
existing bundles so experts can't anchor"), this combines each annotator's precision items
(committed / DIA-approved edges) and recall items (rejected / demoted ReviewItems) into a single
shuffled ``labeler_<aid>.html``. The annotator cannot tell which arm any item came from, which
removes the "this is a reject pool" anchoring bias of sending a separate ``_recall`` file.

**This is the canonical send-bundle generator.** It reuses the selection machinery of
``make_labeling_bundles`` (precision) and ``make_recall_bundles`` (recall) so the two arms are
sampled exactly as before, then merges them. The researcher-side ``bundle_meta_<aid>.json`` records,
per item: ``arm`` (precision|recall), ``judge`` (supported|not_supported = DIA's verdict),
``confidence``, ``is_out_of_ontology``, and the recall ``reason`` — so precision and gate-recall are
scored SEPARATELY after labels return (re-join on item id). ``merged_judge_labels.json`` is the flat
id->judge map.

Blindness contract: head / relation / tail + entity descriptions + evidence quotes are shown for
BOTH arms; the system ``reasoning`` field is SUPPRESSED for ALL items. (Recall reasoning would reveal
the reject verdict, so precision reasoning is hidden too — otherwise the mere presence of a reasoning
panel would distinguish the arms.) Confidence / ooo / reason never reach the HTML.

Caps: ``cap`` (precision, default 40) + ``recall_cap`` (default = ``cap``) → up to cap+recall_cap
items per annotator. Lower either in the config to reduce annotator burden.

Usage:  CUDA_VISIBLE_DEVICES="" python make_merged_bundles.py bundles_config.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from make_labeling_bundles import (
    HERE,
    fetch_triples,
    is_out_of_ontology,
    load_ontology_vocab,
    load_relation_definitions,
    render_html,
    select_for_annotator,
)
from make_recall_bundles import fetch_descriptions, fetch_rejected, select_rejected


def _select_precision(session, ann, cap, ooo_fraction, vocab, rng, overlap_triples):
    """Replicate make_labeling_bundles' per-annotator precision selection (supp + overlap)."""
    aid, doc = ann["annotator_id"], ann["source_document"]
    triples = fetch_triples(session, doc)
    for t in triples:
        t["source_document"] = doc
    chosen = select_for_annotator(triples, cap, ooo_fraction, vocab, rng)
    supp_docs = ann.get("supplementary_documents", [])
    if supp_docs and len(chosen) < cap:
        supp_pool: list[dict[str, Any]] = []
        for sd in supp_docs:
            st = fetch_triples(session, sd)
            for t in st:
                t["source_document"] = sd
            supp_pool += st
        chosen += select_for_annotator(supp_pool, cap - len(chosen), ooo_fraction, vocab, rng)
    chosen_ids = {t["id"] for t in chosen}
    merged = [t for t in overlap_triples if t["id"] not in chosen_ids] + chosen
    return merged, len(triples)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python make_merged_bundles.py config.json")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    cap = int(cfg.get("cap", 40))
    recall_cap = int(cfg.get("recall_cap", cap))
    ooo_fraction = float(cfg.get("ooo_fraction", 0.6))
    overlap_ids = set(cfg.get("overlap_ids", []))
    rng = random.Random(cfg.get("seed", 0))
    out_dir = Path(cfg.get("out_dir", HERE / "bundles"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient  # type: ignore

    vocab = load_ontology_vocab()
    client = Neo4jClient()
    client.connect()
    judge_labels: dict[str, str] = {}

    with client.driver.session() as session:
        relation_defs = load_relation_definitions(session)

        # Shared overlap set (IAA) — same fetch as make_labeling_bundles; empty when unused.
        overlap_triples: list[dict[str, Any]] = []
        if overlap_ids:
            q = (
                "MATCH (h)-[r]->(t) WHERE elementId(r) IN $ids "
                "RETURN elementId(r) AS id, type(r) AS relation, "
                "coalesce(h.name,h.title,elementId(h)) AS head, labels(h)[0] AS head_type, "
                "coalesce(h.description,'') AS head_description, "
                "coalesce(t.name,t.title,elementId(t)) AS tail, labels(t)[0] AS tail_type, "
                "coalesce(t.description,'') AS tail_description, "
                "coalesce(r.description,'') AS reasoning, "
                "r.evidence_quotes AS evidence_quotes, r.confidence AS confidence, "
                "r.source_document AS source_document"
            )
            for rec in session.run(q, ids=list(overlap_ids)):
                d = dict(rec)
                qs = d["evidence_quotes"]
                d["evidence_quotes"] = [qs] if isinstance(qs, str) else [x for x in (qs or []) if x]
                overlap_triples.append(d)

        for ann in cfg["annotators"]:
            aid, doc = ann["annotator_id"], ann["source_document"]

            # --- precision arm (committed / DIA-approved) ---
            prec, n_prec_avail = _select_precision(
                session, ann, cap, ooo_fraction, vocab, rng, overlap_triples
            )
            for t in prec:
                t["arm"], t["judge"] = "precision", "supported"
                t["reason"] = None

            # --- recall arm (rejected / demoted) ---
            pool = fetch_rejected(session, doc)
            rec_items = select_rejected(pool, recall_cap, rng)
            names = {t["head"] for t in rec_items} | {t["tail"] for t in rec_items}
            descs = fetch_descriptions(session, {n for n in names if n})
            for t in rec_items:
                t["head_description"] = descs.get(t["head"], "")
                t["tail_description"] = descs.get(t["tail"], "")
                t["arm"], t["judge"] = "recall", "not_supported"
                # reason already set by fetch_rejected

            # --- merge, suppress reasoning for blindness, shuffle ---
            seen: set[str] = set()
            merged: list[dict[str, Any]] = []
            for t in prec + rec_items:
                if t["id"] in seen:
                    continue
                seen.add(t["id"])
                t["reasoning"] = ""  # blind: hide system rationale for BOTH arms
                merged.append(t)
            rng.shuffle(merged)

            html = render_html(aid, merged, relation_defs)
            (out_dir / f"labeler_{aid}.html").write_text(html, encoding="utf-8")

            meta_items = {}
            for t in merged:
                judge_labels[t["id"]] = t["judge"]
                meta_items[t["id"]] = {
                    "arm": t["arm"],
                    "judge": t["judge"],
                    "confidence": t.get("confidence"),
                    "is_out_of_ontology": is_out_of_ontology(t, vocab),
                    "reason": t.get("reason"),
                    "relation": t["relation"],
                    "source_document": t.get("source_document", doc),
                }
            meta = {
                "annotator_id": aid,
                "source_document": doc,
                "supplementary_documents": ann.get("supplementary_documents", []),
                "bundle_kind": "merged_blind",
                "n_precision": sum(1 for t in merged if t["arm"] == "precision"),
                "n_recall": sum(1 for t in merged if t["arm"] == "recall"),
                "n_shown": len(merged),
                "n_precision_available": n_prec_avail,
                "n_recall_available": len(pool),
                "items": meta_items,
            }
            (out_dir / f"bundle_meta_{aid}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

            # Remove now-obsolete separate-arm files so only the merged bundle is sendable.
            for stale in (f"labeler_{aid}_recall.html", f"bundle_meta_{aid}_recall.json"):
                (out_dir / stale).unlink(missing_ok=True)

            print(f"[ok] {aid}: {len(merged)} items "
                  f"(precision={meta['n_precision']}, recall={meta['n_recall']}) "
                  f"[avail prec={n_prec_avail}, recall={len(pool)}] -> labeler_{aid}.html")

    (HERE / "merged_judge_labels.json").write_text(json.dumps(judge_labels, indent=2), encoding="utf-8")
    if hasattr(client, "close"):
        client.close()
    print(f"\n{len(judge_labels)} items -> merged_judge_labels.json (precision=supported, "
          "recall=not_supported).")
    print("Blind merged bundles in", out_dir, "— send each labeler_<id>.html; split arms via bundle_meta.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build self-contained HTML labeling bundles from the Neo4j knowledge graph.

One ``labeler_<annotator>.html`` is produced per annotator, with that annotator's
assigned triples embedded directly in the file (so it runs from ``file://`` with no
server, no install, fully offline/asynchronous). A private ``bundle_meta_<annotator>.json``
is also written **for you** — it holds ``confidence`` and the out-of-ontology flag keyed
by triple id, which are deliberately **excluded** from the annotator-facing HTML to avoid
anchoring bias. Re-join it during analysis (see ``ingest_labels.py``).

Run this on the ``graphRAG-implementation`` branch with a populated Neo4j KB. The
recruits' papers must already be ingested (``process_papers.py``); this script assumes
they are present and selects triples by the edge ``source_document`` property.

Config (JSON, passed as the only arg)::

    {
      "annotators": [
        {"annotator_id": "alice", "source_document": "Smith2021_MZM"},
        {"annotator_id": "bob",   "source_document": "Lee2020_RingMod"}
      ],
      "cap": 40,                      # max triples shown per annotator
      "ooo_fraction": 0.6,           # target share of out-of-ontology triples
      "overlap_ids": [],             # elementId(r) list everyone also labels (cross-annotator IAA)
      "seed": 0,
      "out_dir": "bundles"
    }

Usage::

    python make_labeling_bundles.py config.json
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any


def _clean_reasoning(text: str) -> str:
    """Strip the 'Global Inference:'/'Rejected…' prefix and the '(Confidence: …)'/'(novel type: …)'
    boilerplate from a DIA edge description, leaving the human-readable justification."""
    if not text:
        return ""
    t = re.sub(r"^(Global Inference|Rejected by semantic verifier)\s*:\s*", "", text).strip()
    t = re.sub(r"\s*\(novel type:[^)]*\)\s*", " ", t)
    t = re.sub(r"\s*\(Confidence:\s*[0-9.]+\)\s*$", "", t)
    return t.strip()

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "labeler_template.html"

DEFAULT_INSTRUCTIONS = (
    "For each relationship the system extracted from your paper, decide whether it is CORRECT, "
    "using your expertise with this work — you generally won't need to re-read the paper. The "
    "entity descriptions and any supporting evidence are shown for context. Many relationships were "
    "INFERRED by the system rather than stated in a single sentence, so judge correctness, not "
    "whether one sentence states it. Use 'Unsure' if you can't judge from what's shown; no penalty."
)

# Relationship types covered by the GenerativeOntology's closed vocabulary. Edges whose
# type is NOT here (e.g. RELATED_TO, inferred cross-doc links) are treated as
# out-of-ontology — exactly where the LLM-judge is uncalibrated, so we oversample them.
ONTOLOGY_RELATION_TYPES = {
    "PERFORMS_FUNCTION",
    "BASED_ON_PRINCIPLE",
    "HAS_PROPERTY",
    "USES_COMPONENT",
}


def load_ontology_vocab() -> set[str] | None:
    """Best-effort load of the ontology's closed concept vocabulary.

    Returns a lowercased set of allowed concept names, or ``None`` if the loader is
    unavailable (then out-of-ontology is decided by relation type alone).
    """
    try:
        from PhotonicsAI.KnowledgeBase.agents.ppc_agent.ontology_loader import (  # type: ignore
            load_ontology,
        )

        onto = load_ontology()
        vocab: set[str] = set()
        for attr in ("concepts", "primitives", "terms", "vocabulary"):
            v = getattr(onto, attr, None)
            if v:
                vocab |= {str(x).strip().lower() for x in v}
        return vocab or None
    except Exception as exc:  # noqa: BLE001 - defensive: loader shape may vary by branch
        print(f"[warn] ontology vocab unavailable ({exc}); using relation-type test only")
        return None


def fetch_triples(session: Any, source_document: str) -> list[dict[str, Any]]:
    """Return all evidence-bearing triples whose edge came from ``source_document``."""
    query = """
    MATCH (h)-[r]->(t)
    WHERE r.source_document = $doc
      AND r.evidence_quotes IS NOT NULL
    RETURN elementId(r)                       AS id,
           type(r)                            AS relation,
           coalesce(h.name, h.title, elementId(h)) AS head,
           labels(h)[0]                       AS head_type,
           coalesce(h.description, '')        AS head_description,
           coalesce(t.name, t.title, elementId(t)) AS tail,
           labels(t)[0]                       AS tail_type,
           coalesce(t.description, '')        AS tail_description,
           coalesce(r.description, '')        AS reasoning,
           r.evidence_quotes                  AS evidence_quotes,
           r.confidence                       AS confidence
    """
    out: list[dict[str, Any]] = []
    for rec in session.run(query, doc=source_document):
        d = dict(rec)
        quotes = d.get("evidence_quotes")
        if isinstance(quotes, str):
            quotes = [quotes]
        quotes = [q for q in (quotes or []) if q and str(q).strip()]
        if not quotes:  # no usable evidence -> can't judge faithfulness; report separately
            continue
        d["evidence_quotes"] = quotes
        out.append(d)
    return out


def load_relation_definitions(session: Any) -> dict[str, str]:
    """{relation_type: human-readable definition} from the seed SchemaRelationType nodes.
    Shown to annotators so they know what each relation means (e.g. RELATED_TO = 'generic… use
    sparingly'), which is essential context for judging faithfulness."""
    q = ("MATCH (rt:SchemaRelationType) WHERE rt.description IS NOT NULL "
         "RETURN rt.name AS n, rt.description AS d")
    return {rec["n"]: rec["d"] for rec in session.run(q) if rec["n"]}


def is_out_of_ontology(triple: dict[str, Any], vocab: set[str] | None) -> bool:
    if triple["relation"] not in ONTOLOGY_RELATION_TYPES:
        return True
    if vocab is not None:
        for end in (triple.get("head"), triple.get("tail")):
            if end and str(end).strip().lower() not in vocab:
                return True
    return False


def select_for_annotator(
    triples: list[dict[str, Any]],
    cap: int,
    ooo_fraction: float,
    vocab: set[str] | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Cap the triples, oversampling out-of-ontology ones toward ``ooo_fraction``."""
    ooo = [t for t in triples if is_out_of_ontology(t, vocab)]
    ino = [t for t in triples if not is_out_of_ontology(t, vocab)]
    rng.shuffle(ooo)
    rng.shuffle(ino)
    n_ooo = min(len(ooo), round(cap * ooo_fraction))
    chosen = ooo[:n_ooo]
    chosen += ino[: cap - len(chosen)]
    chosen += ooo[n_ooo : cap - len(chosen)]  # backfill if not enough in-ontology
    rng.shuffle(chosen)
    return chosen[:cap]


def render_html(annotator_id: str, items: list[dict[str, Any]], relation_defs: dict[str, str]) -> str:
    """Embed the annotator-facing bundle (no confidence / no ooo flag) into the template."""
    public_items = [
        {
            "id": t["id"],
            "head": t["head"],
            "head_type": t.get("head_type"),
            "head_description": t.get("head_description", ""),
            "relation": t["relation"],
            "tail": t["tail"],
            "tail_type": t.get("tail_type"),
            "tail_description": t.get("tail_description", ""),
            "reasoning": _clean_reasoning(t.get("reasoning", "")),  # system's justification (opt-in view)
            "source_document": t["source_document"],
            "shown_order": i,
            "evidence_quotes": t["evidence_quotes"],
        }
        for i, t in enumerate(items)
    ]
    used_rels = {t["relation"] for t in items}
    bundle = {
        "annotator_id": annotator_id,
        "schema_version": 2,
        "task": "relationship_validity",   # judge whether the relationship is correct, NOT quote-faithfulness
        "instructions": DEFAULT_INSTRUCTIONS,
        "relation_definitions": {r: relation_defs[r] for r in sorted(used_rels) if r in relation_defs},
        "items": public_items,
    }
    # Escape '</' so embedded data can never close the <script> tag early.
    payload = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    template = TEMPLATE.read_text(encoding="utf-8")
    start, end = template.index("/*__BUNDLE_START__*/"), template.index("/*__BUNDLE_END__*/")
    return (
        template[: start + len("/*__BUNDLE_START__*/")]
        + payload
        + template[end:]
    )


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python make_labeling_bundles.py config.json")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    cap = int(cfg.get("cap", 40))
    ooo_fraction = float(cfg.get("ooo_fraction", 0.6))
    overlap_ids = set(cfg.get("overlap_ids", []))
    rng = random.Random(cfg.get("seed", 0))
    out_dir = Path(cfg.get("out_dir", HERE / "bundles"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient  # type: ignore

    vocab = load_ontology_vocab()
    client = Neo4jClient()
    client.connect()

    # Pre-fetch overlap triples once (shared across all annotators).
    overlap_triples: list[dict[str, Any]] = []
    with client.driver.session() as session:
        relation_defs = load_relation_definitions(session)
        if overlap_ids:
            q = """
            MATCH (h)-[r]->(t) WHERE elementId(r) IN $ids
            RETURN elementId(r) AS id, type(r) AS relation,
                   coalesce(h.name,h.title,elementId(h)) AS head, labels(h)[0] AS head_type,
                   coalesce(h.description,'') AS head_description,
                   coalesce(t.name,t.title,elementId(t)) AS tail, labels(t)[0] AS tail_type,
                   coalesce(t.description,'') AS tail_description,
                   coalesce(r.description,'') AS reasoning,
                   r.evidence_quotes AS evidence_quotes, r.confidence AS confidence,
                   r.source_document AS source_document
            """
            for rec in session.run(q, ids=list(overlap_ids)):
                d = dict(rec)
                qs = d["evidence_quotes"]
                d["evidence_quotes"] = [qs] if isinstance(qs, str) else [x for x in (qs or []) if x]
                overlap_triples.append(d)

        for ann in cfg["annotators"]:
            aid, doc = ann["annotator_id"], ann["source_document"]
            triples = fetch_triples(session, doc)
            for t in triples:
                t["source_document"] = doc
            no_evidence = ann.get("_skip_count", 0)  # already filtered; informational

            # Optional: top up a thin bundle with triples from documents adjacent to the
            # annotator's expertise (e.g. Sharma's own #08 has only 29 committed triples, topped
            # up from her co-authored OPA paper #09). The annotator's OWN paper is kept in full;
            # supplementary docs fill only the remaining slots, still prioritizing ooo edges.
            supp_docs = ann.get("supplementary_documents", [])
            chosen = select_for_annotator(triples, cap, ooo_fraction, vocab, rng)
            if supp_docs and len(chosen) < cap:
                supp_pool: list[dict[str, Any]] = []
                for sd in supp_docs:
                    st = fetch_triples(session, sd)
                    for t in st:
                        t["source_document"] = sd
                    supp_pool += st
                topup = select_for_annotator(supp_pool, cap - len(chosen), ooo_fraction, vocab, rng)
                chosen = chosen + topup
                rng.shuffle(chosen)

            # Prepend the shared overlap set (deduped against the annotator's own).
            chosen_ids = {t["id"] for t in chosen}
            merged = [t for t in overlap_triples if t["id"] not in chosen_ids] + chosen

            html = render_html(aid, merged, relation_defs)
            (out_dir / f"labeler_{aid}.html").write_text(html, encoding="utf-8")

            meta = {
                "annotator_id": aid,
                "source_document": doc,
                "supplementary_documents": supp_docs,
                "n_available": len(triples),
                "n_shown": len(merged),
                "n_no_evidence_excluded": no_evidence,
                "items": {
                    t["id"]: {
                        "confidence": t.get("confidence"),
                        "is_out_of_ontology": is_out_of_ontology(t, vocab),
                        "is_overlap": t["id"] in overlap_ids,
                        "source_document": t.get("source_document", doc),
                    }
                    for t in merged
                },
            }
            (out_dir / f"bundle_meta_{aid}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"[ok] {aid}: {len(merged)} triples ({doc}) -> labeler_{aid}.html")

    client.close() if hasattr(client, "close") else None
    print(f"\nBundles written to {out_dir}. Send each labeler_<id>.html to its annotator.")


if __name__ == "__main__":
    main()

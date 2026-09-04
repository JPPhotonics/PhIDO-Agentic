"""Relationship-specific evidence-quote re-extraction for the labeling bundles.

The KG's edge `evidence_quotes` are ENTITY-level (a per-paper pool of quotes supporting the
entities), and relationships are largely LLM-INFERRED — so the attached quote often backs the
topic, not the specific head->relation->tail fact (vsa_agent.py:417). For a faithfulness
annotation that means the quote must instead be the sentence(s) that actually assert the relation.

This script, for each triple in the 5 bundles, reads the source paper text and asks an LLM to
return the VERBATIM sentence(s) that directly support that relation (or none). Quotes are
verified to be real substrings of the paper (no hallucinations). Edges with a real supporting
sentence get it; edges with none get an explicit "inferred, no direct support" note so the
annotator can judge them honestly. Original quotes are backed up (reversible); the KG edge gains
`evidence_status` ("supported_specific" | "inferred_no_direct_support").

Run (pilot):  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/labeling/reextract_quotes.py --paper 08_ --dry-run
Run (all):    CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/labeling/reextract_quotes.py
Then regenerate bundles: make_labeling_bundles.py benchmark/labeling/bundles_config.json
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
import subprocess
from pathlib import Path
from typing import List

from pydantic import BaseModel

import sys
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent.parent))
from PhotonicsAI.Photon import llm_api  # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient  # noqa: E402
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig  # noqa: E402

PAPERS = HERE.parent.parent / "papers"
BUNDLES = HERE / "bundles"
BACKUP = HERE / "quote_reextract_backup.json"
BATCH = 10
NO_SUPPORT_NOTE = "[No sentence in the source paper directly supports this; the relationship was inferred by the extraction system.]"

REL_PHRASE = {
    "PERFORMS_FUNCTION": "performs the function of", "BASED_ON_PRINCIPLE": "is based on the principle of",
    "HAS_PROPERTY": "has the property", "USES_COMPONENT": "uses the component", "RELATED_TO": "is related to",
}


class QuoteResult(BaseModel):
    id: str
    supported: bool
    quotes: List[str]


class QuoteResults(BaseModel):
    results: List[QuoteResult]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def is_real_quote(q: str, ntext: str, ntext_words: list[str]) -> bool:
    """Accept a quote as genuine if it's a normalized substring of the paper, or if a long
    CONTIGUOUS run of its words appears in the paper (tolerates pdftotext mangling / light LLM
    cleaning, while still rejecting hallucinated text that has no long run in the source)."""
    nq = _norm(q)
    if len(nq) < 15:
        return False
    if nq in ntext:
        return True
    qw = nq.split()
    if len(qw) < 4:
        return False
    block = difflib.SequenceMatcher(None, qw, ntext_words, autojunk=False).find_longest_match(
        0, len(qw), 0, len(ntext_words))
    return block.size >= max(6, int(0.6 * len(qw)))


def paper_text(stem: str) -> str:
    pdf = PAPERS / f"{stem}.pdf"
    raw = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    raw = re.sub(r"-\s*\n\s*", "", raw)        # de-hyphenate line breaks
    return re.sub(r"\s+", " ", raw).strip()    # collapse whitespace


SYS = ("You verify whether a paper supports extracted relationship statements. For each statement, "
       "return verbatim sentence(s) COPIED EXACTLY from the provided paper text that DIRECTLY support "
       "it — the text must actually assert the specific relationship, not merely mention the entities. "
       "If nothing in the text directly supports it, set supported=false and quotes=[]. Never invent text.")


def extract_batch(text: str, triples: list[dict]) -> dict:
    lines = []
    for t in triples:
        ph = REL_PHRASE.get(t["rel"], t["rel"].lower().replace("_", " "))
        lines.append(f'[id={t["id"]}] "{t["head"]}" {ph} "{t["tail"]}"  (head:{t["head_type"]}, tail:{t["tail_type"]})')
    prompt = (f"PAPER TEXT:\n{text}\n\nSTATEMENTS:\n" + "\n".join(lines) +
              '\n\nReturn JSON {"results":[{"id":...,"supported":...,"quotes":[verbatim...]}]}.')
    try:
        res = llm_api.callgoogle_pydantic(prompt, SYS, QuoteResults)
        return {r.id: r for r in (res.results if res and hasattr(res, "results") else [])}
    except Exception as e:
        print(f"  [warn] batch extraction failed: {e}")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=None, help="only this source_document prefix (e.g. 08_)")
    ap.add_argument("--dry-run", action="store_true", help="don't write to Neo4j; just report")
    args = ap.parse_args()

    # collect bundle triple ids grouped by source_document
    by_doc: dict[str, list[str]] = {}
    for mf in sorted(glob.glob(str(BUNDLES / "bundle_meta_*.json"))):
        m = json.loads(Path(mf).read_text())
        doc = m["source_document"]
        if args.paper and not doc.startswith(args.paper):
            continue
        by_doc.setdefault(doc, []).extend(m["items"].keys())

    client = Neo4jClient(config=Neo4jConfig()); client.connect()
    backup = json.loads(BACKUP.read_text()) if BACKUP.exists() else {}
    totals = {"supported": 0, "no_support": 0, "dropped_unverified": 0}

    for doc, ids in by_doc.items():
        ids = list(dict.fromkeys(ids))
        text = paper_text(doc)
        if len(text) < 500:
            print(f"[skip] {doc}: paper text too short ({len(text)} chars)"); continue
        # fetch triple details
        with client.driver.session() as s:
            rows = {r["id"]: r for r in s.run(
                "MATCH (h)-[r]->(t) WHERE elementId(r) IN $ids "
                "RETURN elementId(r) AS id, type(r) AS rel, coalesce(h.name,h.title) AS head, "
                "labels(h)[0] AS head_type, coalesce(t.name,t.title) AS tail, labels(t)[0] AS tail_type, "
                "r.evidence_quotes AS old", ids=ids)}
        triples = [dict(rows[i], id=i) for i in ids if i in rows]
        print(f"\n=== {doc}  ({len(triples)} triples, paper {len(text)} chars) ===")

        results = {}
        for b in range(0, len(triples), BATCH):
            results.update(extract_batch(text, triples[b:b + BATCH]))

        ntext = _norm(text)
        ntext_words = ntext.split()
        updates = []
        for t in triples:
            r = results.get(t["id"])
            verified = []
            if r and r.supported:
                for q in r.quotes:
                    if is_real_quote(q, ntext, ntext_words):
                        verified.append(q.strip())
                    else:
                        totals["dropped_unverified"] += 1
            if verified:
                totals["supported"] += 1
                updates.append((t["id"], verified, "supported_specific"))
            else:
                totals["no_support"] += 1
                updates.append((t["id"], [NO_SUPPORT_NOTE], "inferred_no_direct_support"))

        sup = sum(1 for _, _, st in updates if st == "supported_specific")
        print(f"  supported w/ verbatim quote: {sup}/{len(triples)}   no-direct-support: {len(triples)-sup}")
        for tid, q, st in updates[:3]:
            tr = next(x for x in triples if x["id"] == tid)
            print(f"   ({tr['head']}) -{tr['rel']}-> ({tr['tail']}) [{st}]")
            print(f"      {q[0][:150]!r}")

        if not args.dry_run:
            with client.driver.session() as s:
                for tid, q, st in updates:
                    if tid not in backup:
                        backup[tid] = rows[tid]["old"]
                    s.run("MATCH ()-[r]->() WHERE elementId(r)=$id "
                          "SET r.evidence_quotes=$q, r.evidence_status=$st, "
                          "r.evidence_quotes_reextracted=true", id=tid, q=q, st=st)

    if not args.dry_run:
        BACKUP.write_text(json.dumps(backup, indent=2))
        print(f"\n[backup] originals -> {BACKUP}")
    print(f"\nTOTALS: {totals}")
    print("(dry-run; no KG writes)" if args.dry_run else "KG updated. Now regenerate the bundles.")


if __name__ == "__main__":
    main()

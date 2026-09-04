"""KB quality fix: strip redundant parenthetical restatements in node names.

The acronym agent sometimes produced names like "free spectral range (free spectral range)"
or "quantum-confined Stark effect (quantum-confined stark effect)" — where the parenthetical
merely restates the term (case/hyphen/substring variant) rather than giving a genuine acronym
or module id. Those read badly in the triples shown to expert annotators. This collapses them
to the clean form, leaving genuine acronym/expansion pairs (e.g. "titanium nitride (TiN)",
"Germanium photodetector (PD)") untouched.

Idempotent (re-running finds nothing) and reversible (originals backed up to
fix_name_backup.json). Re-embeds each renamed node so retrieval stays consistent.

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python fix_redundant_parenthetical_names.py
"""
import json
import re
from pathlib import Path

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

BACKUP = Path("fix_name_backup.json")
_PAREN = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


def redundant_fix(name: str) -> str | None:
    """Return the cleaned name if the parenthetical is a redundant restatement, else None."""
    m = _PAREN.match(name)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    if not a or not b:
        return None
    na, nb = _norm(a), _norm(b)
    # Redundant when the two sides are the same term modulo case/hyphen/space, or one
    # fully contains the other (a longer phrase whose paren adds no new token).
    if na == nb or nb in na or na in nb:
        return a
    return None


def main() -> None:
    client = Neo4jClient(config=Neo4jConfig())
    client.connect()

    fixes = []
    with client.driver.session() as s:
        rows = s.run("MATCH (n) WHERE n.name IS NOT NULL AND n.name CONTAINS '(' "
                     "RETURN elementId(n) AS eid, labels(n)[0] AS lab, n.name AS nm")
        for r in rows:
            new = redundant_fix(r["nm"])
            if new and new != r["nm"]:
                fixes.append({"eid": r["eid"], "label": r["lab"], "old": r["nm"], "new": new})

    if not fixes:
        print("No redundant parenthetical names found — KB already clean.")
        return

    BACKUP.write_text(json.dumps(fixes, indent=2))
    print(f"Backed up {len(fixes)} originals -> {BACKUP}")

    for f in fixes:
        with client.driver.session() as s:
            s.run("MATCH (n) WHERE elementId(n) = $eid SET n.name = $new", eid=f["eid"], new=f["new"])
        client.update_entity_embedding(f["eid"], f["label"])   # re-embed from the cleaned name
        print(f"  [{f['label']:18s}] {f['old']!r} -> {f['new']!r}")

    print(f"\nFixed + re-embedded {len(fixes)} node names.")


if __name__ == "__main__":
    main()

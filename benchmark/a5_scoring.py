"""Pure (KB-free) scoring + normalization helpers for the A5 PDK-ingestion benchmark.

Split out from ``a5_kg_vs_library.py`` so the deterministic, hermetic parts (label
normalization, docstring-gold derivation, per-cell set scoring) can be unit-tested with
synthetic inputs and no Neo4j connection.

The KG ``Design_Function`` / ``Component`` vocabulary is known to be fragmented and
synonymous (e.g. "amplitude modulation (AM)" vs "amplitude modulator" vs "AM modulator").
Name-level precision/recall therefore MUST be normalization-aware, and any gold label that
finds no normalized match must be surfaced (not silently scored as an ingestion miss) so a
reader can separate genuine ingestion gaps from pure vocabulary mismatch.
"""

from __future__ import annotations

import re

# NodeLabels that the ingestion's resolve_performs_function() deliberately drops before
# building the Design_Function search query (relationship_resolver.py::resolve_performs_function).
# These are structural / port-shape / active-passive tags, not functions, so they must be
# EXCLUDED from the PERFORMS_FUNCTION gold (including them would manufacture false recall misses).
PERFORMS_SKIP_LABELS = frozenset(
    {"active", "passive", "1x1", "1x2", "2x2", "1x4", "2x4"}
)

# The two structural tags that describe a cell's active/passive nature rather than its ports.
ACTIVE_PASSIVE = frozenset({"active", "passive"})

# Parenthetical-acronym pattern, e.g. "amplitude modulation (AM)" -> base + "am".
_PAREN_ACRONYM = re.compile(r"\(([^)]*)\)")
_NONWORD = re.compile(r"[^a-z0-9]+")


def normalize_label(label: str) -> str:
    """Canonicalize a function/component/property name for normalization-aware matching.

    Lowercase, strip, drop parenthetical acronyms, collapse hyphens/underscores/whitespace
    to single spaces, and trim. This is intentionally conservative: it does NOT apply a
    synonym map (that would hide vocabulary fragmentation we want to measure). It only
    removes surface formatting differences.
    """
    s = (label or "").strip().lower()
    s = _PAREN_ACRONYM.sub(
        " ", s
    )  # "amplitude modulation (am)" -> "amplitude modulation"
    s = _NONWORD.sub(" ", s)  # hyphens, underscores, punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_set(labels) -> set[str]:
    """Normalize an iterable of labels to a set of canonical strings (drops empties)."""
    out: set[str] = set()
    for label in labels or []:
        n = normalize_label(str(label))
        if n:
            out.add(n)
    return out


def performs_function_gold(node_labels) -> set[str]:
    """Gold PERFORMS_FUNCTION target names from a cell's NodeLabels.

    Drops the structural/port skip-labels the ingestion itself drops, then normalizes.
    A cell with only structural labels (e.g. ['passive', '1x1']) yields an empty gold set —
    such cells are correctly NOT scored for PERFORMS_FUNCTION (no function is asserted).
    """
    kept = [
        lbl
        for lbl in (node_labels or [])
        if normalize_label(str(lbl))
        not in {normalize_label(s) for s in PERFORMS_SKIP_LABELS}
    ]
    return normalize_set(kept)


def cell_set_prf(retrieved, gold) -> tuple[int, int, int]:
    """True/false-positive/false-negative counts for one cell's normalized sets.

    Returns (tp, fp, fn). Aggregation (micro = sum the counts; macro = average per-cell P/R)
    is done by the caller so both views are available.
    """
    r, g = set(retrieved), set(gold)
    tp = len(r & g)
    return (tp, len(r) - tp, len(g) - tp)


def unmatched_gold(retrieved, gold) -> set[str]:
    """Gold (normalized) labels with NO normalized match in retrieved — the recall misses.

    Surfacing these is the honesty mechanism: a name appearing here is EITHER a genuine
    ingestion gap OR a vocabulary-normalization mismatch (the KG used a synonym our
    conservative normalizer did not unify). The report lists them so a reader can judge.
    """
    return set(gold) - set(retrieved)

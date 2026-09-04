"""Shared benchmark metrics (stdlib only).

Retrieval (E1/B2): ``pass_at_k``, ``mrr``, ``average_rank``, ``set_prf``,
``retrievable_coverage_at_k``. Classification (A1/A5/A2-types): ``prf``,
``confusion_matrix``. Topology (B3): ``edge_prf``, ``approx_ged``.

A *ranked result* is a list of candidate ids best-first. *Gold* is either a single id
(single-target retrieval) or a set of ids (compositional / set retrieval).
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Sequence

Id = Hashable


# --------------------------------------------------------------------------- retrieval
def pass_at_k(ranked: Sequence[Id], gold: Id | set[Id], k: int) -> bool:
    """True if (any) gold id appears within the top-k of ``ranked``."""
    top = set(ranked[:k])
    gold_set = gold if isinstance(gold, set) else {gold}
    return bool(top & gold_set)


def mean_pass_at_k(cases: Iterable[tuple[Sequence[Id], Id | set[Id]]], k: int) -> float:
    cases = list(cases)
    if not cases:
        return float("nan")
    return sum(pass_at_k(r, g, k) for r, g in cases) / len(cases)


def reciprocal_rank(ranked: Sequence[Id], gold: Id | set[Id]) -> float:
    """1/rank of the first gold hit (rank starts at 1); 0 if not found."""
    gold_set = gold if isinstance(gold, set) else {gold}
    for i, c in enumerate(ranked, start=1):
        if c in gold_set:
            return 1.0 / i
    return 0.0


def mrr(cases: Iterable[tuple[Sequence[Id], Id | set[Id]]]) -> float:
    cases = list(cases)
    if not cases:
        return float("nan")
    return sum(reciprocal_rank(r, g) for r, g in cases) / len(cases)


def average_rank(cases: Iterable[tuple[Sequence[Id], Id | set[Id]]], missing: int | None = None) -> float:
    """Mean rank of the first gold hit. Misses count as ``missing`` (default len+1)."""
    cases = list(cases)
    ranks: list[int] = []
    for ranked, gold in cases:
        gold_set = gold if isinstance(gold, set) else {gold}
        rank = next((i for i, c in enumerate(ranked, 1) if c in gold_set), None)
        ranks.append(rank if rank is not None else (missing if missing is not None else len(ranked) + 1))
    return sum(ranks) / len(ranks) if ranks else float("nan")


def set_prf(retrieved: Iterable[Id], gold: Iterable[Id]) -> tuple[float, float, float]:
    """Precision/recall/F1 for compositional (set) retrieval."""
    r, g = set(retrieved), set(gold)
    tp = len(r & g)
    return prf(tp, len(r) - tp, len(g) - tp)


def retrievable_coverage_at_k(
    cases: Iterable[tuple[Sequence[Id], set[Id]]], k: int
) -> float:
    """Mean fraction of each query's gold set found in the top-k.

    The benchmark's completeness ceiling: end-to-end success <= retrievable coverage of
    the required component set (KB_BENCHMARK_PLAN.md / lit-review C3).
    """
    cases = list(cases)
    if not cases:
        return float("nan")
    fracs = []
    for ranked, gold in cases:
        if not gold:
            continue
        fracs.append(len(set(ranked[:k]) & set(gold)) / len(gold))
    return sum(fracs) / len(fracs) if fracs else float("nan")


# ----------------------------------------------------------------------- classification
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from counts."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f)


def confusion_matrix(
    pairs: Iterable[tuple[Hashable, Hashable]], labels: Sequence[Hashable] | None = None
) -> dict[Hashable, dict[Hashable, int]]:
    """Confusion matrix as ``{true: {pred: count}}`` (e.g. A2 edge-type confusion)."""
    pairs = list(pairs)
    if labels is None:
        labels = sorted({x for pair in pairs for x in pair}, key=str)
    m = {t: {p: 0 for p in labels} for t in labels}
    for t, p in pairs:
        m.setdefault(t, {l: 0 for l in labels}).setdefault(p, 0)
        m[t][p] += 1
    return m


# --------------------------------------------------------------------------- topology
Edge = tuple  # (src, dst) or (src, dst, label)


def edge_prf(pred_edges: Iterable[Edge], gold_edges: Iterable[Edge]) -> tuple[float, float, float]:
    """Directed edge-set precision/recall/F1 for B3 topology correctness.

    Assumes node identity corresponds across pred/gold (e.g. shared instance names).
    Include the relation/port in the tuple to score typed edges.
    """
    return set_prf(set(pred_edges), set(gold_edges))


def approx_ged(
    pred_nodes: dict[Id, Hashable],
    pred_edges: Iterable[Edge],
    gold_nodes: dict[Id, Hashable],
    gold_edges: Iterable[Edge],
) -> dict[str, float]:
    """Approximate graph-edit-distance for small labeled circuit graphs.

    Exact GED is NP-hard; for benchmark topologies we use a tractable label-aware
    surrogate: node-label multiset symmetric difference + directed edge-set symmetric
    difference (node identity assumed shared). Reported raw and normalized by gold size.
    Pair with ``edge_prf`` as the primary correctness signal.
    """
    node_diff = sum((Counter(pred_nodes.values()) - Counter(gold_nodes.values())).values())
    node_diff += sum((Counter(gold_nodes.values()) - Counter(pred_nodes.values())).values())
    pe, ge = set(pred_edges), set(gold_edges)
    edge_diff = len(pe ^ ge)
    ged = node_diff + edge_diff
    denom = len(gold_nodes) + len(ge)
    return {"ged": float(ged), "node_diff": float(node_diff), "edge_diff": float(edge_diff),
            "normalized_ged": (ged / denom) if denom else 0.0}

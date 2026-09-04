"""LP-Measure — reference-free KG health via link-prediction recovery (secondary L1 check).

Hide a fraction of the graph's triples, then see whether the graph's *own* structure
predicts them back (Hit@k, MRR). A healthy, internally-consistent KG recovers its hidden
edges; a corrupted one doesn't — so this separates good from bad with **no gold standard
and no human labels**. It formally captures consistency/redundancy only (a caveat to state).

**Akrami de-leak first:** strip exact-duplicate and symmetric (a,r,b)/(b,r,a) triples, which
otherwise let trivial rules inflate the score (19-175% in the link-prediction literature).

A triple is ``(head, relation, tail)``. The default predictor ranks candidate tails by
relation-conditioned tail frequency — structural, no training labels.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable

Triple = tuple


def de_leak(triples: list[Triple]) -> dict:
    """Remove exact duplicates and one side of each symmetric (a,r,b)/(b,r,a) pair."""
    seen, kept, removed_dup, removed_sym = set(), [], 0, 0
    have = set(triples)
    for h, r, t in triples:
        if (h, r, t) in seen:
            removed_dup += 1
            continue
        if (t, r, h) in seen:          # its mirror already kept -> drop this one as leakage
            removed_sym += 1
            seen.add((h, r, t))
            continue
        seen.add((h, r, t))
        kept.append((h, r, t))
    return {"triples": kept, "removed_exact_duplicates": removed_dup,
            "removed_symmetric": removed_sym, "n_before": len(triples), "n_after": len(kept)}


def _default_predictor(train: list[Triple]):
    """Rank candidate tails by P(tail|relation), tie-broken by global tail frequency."""
    rel_tail = defaultdict(Counter)
    glob = Counter()
    for _, r, t in train:
        rel_tail[r][t] += 1
        glob[t] += 1

    def score(h, r, candidate_tail) -> tuple:
        return (rel_tail[r][candidate_tail], glob[candidate_tail])
    return score


def lp_measure(triples: list[Triple], fraction: float = 0.2, seed: int = 0,
               predictor_factory: Callable | None = None, ks=(1, 3, 10)) -> dict:
    """Hide `fraction` of triples, rank the true tail among all entities, report Hit@k + MRR."""
    triples = list(triples)
    rng = random.Random(seed)
    idx = list(range(len(triples)))
    rng.shuffle(idx)
    n_test = max(1, int(len(triples) * fraction))
    test_ids = set(idx[:n_test])
    train = [t for i, t in enumerate(triples) if i not in test_ids]
    test = [t for i, t in enumerate(triples) if i in test_ids]

    entities = sorted({e for h, _, t in triples for e in (h, t)})
    score = (predictor_factory or _default_predictor)(train)

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    for h, r, t in test:
        ranked = sorted(entities, key=lambda c: score(h, r, c), reverse=True)
        rank = ranked.index(t) + 1 if t in ranked else len(entities) + 1
        rr_sum += 1.0 / rank
        for k in ks:
            hits[k] += int(rank <= k)
    n = len(test)
    return {"n_test": n, "mrr": rr_sum / n, **{f"hit@{k}": hits[k] / n for k in ks},
            "n_entities": len(entities)}


def corrupt(triples: list[Triple], rate: float = 0.5, seed: int = 0) -> list[Triple]:
    """Randomly rewire a fraction of tails — the negative control for LP-Measure."""
    rng = random.Random(seed)
    entities = sorted({e for h, _, t in triples for e in (h, t)})
    out = []
    for h, r, t in triples:
        out.append((h, r, rng.choice(entities)) if rng.random() < rate else (h, r, t))
    return out


def health_report(triples: list[Triple], fraction: float = 0.2, seed: int = 0) -> dict:
    """De-leak, then LP-Measure the clean graph and a corrupted control for separation."""
    dl = de_leak(triples)
    clean = lp_measure(dl["triples"], fraction, seed)
    corrupted = lp_measure(corrupt(dl["triples"], 0.7, seed), fraction, seed)
    return {"de_leak": {k: v for k, v in dl.items() if k != "triples"},
            "clean": clean, "corrupted_control": corrupted,
            "separation_mrr": clean["mrr"] - corrupted["mrr"]}

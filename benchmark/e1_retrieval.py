"""E1 / B2 — retrieval A/B harness (the core extrinsic claim).

Three arms, same queries + same candidate descriptions, KG as the only treatment:
* ``lexical_arm`` — token-overlap ranking (stand-in for ``search_components``);
* ``llm_json_arm(llm_fn)`` — LLM-over-JSON selection (wraps ``llm_api.llm_search``);
* ``kg_arm(retriever)`` — KG-grounded; the retriever is supplied once the KB exists
  (also serves the enrichment ablation: pass a no-enrichment retriever as a separate arm).

Query set mixes **paraphrase** and **functional/indirect** queries (the latter exercise the
KG's PERFORMS_FUNCTION edges and are required for the KG advantage to be observable).
``run_scaling`` re-runs every arm across PDK sizes (50/131/262) to test whether the KG
advantage grows with distractor density. Metrics come from ``metrics.py``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

from metrics import mean_pass_at_k, mrr, retrievable_coverage_at_k, set_prf

Arm = Callable[[str, list[dict]], list[str]]   # (query, candidates) -> ranked candidate ids


@dataclass
class QueryCase:
    id: str
    query: str
    gold: set[str]                 # component id(s) a correct retrieval returns
    kind: str = "paraphrase"       # "paraphrase" | "functional"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


# ------------------------------------------------------------------------------- arms
def lexical_arm(query: str, candidates: list[dict]) -> list[str]:
    """Rank by token overlap of the query against each candidate's name+description."""
    q = _tokens(query)
    scored = []
    for c in candidates:
        text = _tokens(f"{c.get('name','')} {c.get('description','')} {c.get('function','')}")
        overlap = len(q & text)
        scored.append((overlap, c["id"]))
    scored.sort(key=lambda t: (-t[0], str(t[1])))    # deterministic tie-break by id
    return [cid for _, cid in scored]


def llm_json_arm(llm_fn: Callable[[str, list[dict]], list[str]]) -> Arm:
    """Wrap an LLM-over-JSON selector (mockable; production wires to llm_api.llm_search)."""
    return lambda query, candidates: list(llm_fn(query, candidates))


def kg_arm(retriever: Callable[[str, list[dict]], list[str]] | None = None) -> Arm:
    """KG-grounded arm. ``retriever`` is supplied once the KB exists; stub raises until then."""
    def _arm(query: str, candidates: list[dict]) -> list[str]:
        if retriever is None:
            raise NotImplementedError("KG arm needs a KB-backed retriever (pending KB population)")
        return list(retriever(query, candidates))
    return _arm


# ----------------------------------------------------------------------- query generation
def generate_queries(component: dict, llm_fn: Callable[[dict], list[tuple[str, str]]] | None = None) -> list[QueryCase]:
    """Paraphrase + functional queries for a component. Template fallback if no llm_fn."""
    cid = component["id"]
    if llm_fn is not None:
        return [QueryCase(f"{cid}-{i}", q, {cid}, kind) for i, (q, kind) in enumerate(llm_fn(component))]
    name, func = component.get("name", cid), component.get("function", "")
    cases = [QueryCase(f"{cid}-p", f"a {name}", {cid}, "paraphrase")]
    if func:
        cases.append(QueryCase(f"{cid}-f", f"a component that {func}", {cid}, "functional"))
    return cases


# --------------------------------------------------------------------------- scoring
def score_arm(arm: Arm, cases: list[QueryCase], candidates: list[dict], ks=(1, 3)) -> dict:
    ranked = {c.id: arm(c.query, candidates) for c in cases}
    pairs = [(ranked[c.id], c.gold) for c in cases]
    out = {"n_queries": len(cases), "mrr": mrr(pairs)}
    for k in ks:
        out[f"pass@{k}"] = mean_pass_at_k(pairs, k)
        out[f"coverage@{k}"] = retrievable_coverage_at_k([(ranked[c.id], c.gold) for c in cases], k)
        sp = [set_prf(ranked[c.id][:k], c.gold) for c in cases]
        out[f"set_precision@{k}"] = sum(p for p, _, _ in sp) / len(sp) if sp else float("nan")
        out[f"set_recall@{k}"] = sum(r for _, r, _ in sp) / len(sp) if sp else float("nan")
    # break out by query kind (paraphrase vs functional — where the KG should help most)
    out["by_kind"] = {}
    for kind in {c.kind for c in cases}:
        sub = [(ranked[c.id], c.gold) for c in cases if c.kind == kind]
        out["by_kind"][kind] = {"n": len(sub), "pass@1": mean_pass_at_k(sub, 1), "mrr": mrr(sub)}
    return out


def run_e1(arms: dict[str, Arm], cases: list[QueryCase], candidates: list[dict], ks=(1, 3)) -> dict:
    """Per-arm metrics on one candidate pool."""
    return {name: score_arm(arm, cases, candidates, ks) for name, arm in arms.items()}


# --------------------------------------------------------------------------- scaling
def sample_pool(candidates: list[dict], size: int, must_include: set[str], seed: int = 0) -> list[dict]:
    """A pool of `size` candidates that always contains the gold (`must_include`) components."""
    by_id = {c["id"]: c for c in candidates}
    keep = [by_id[i] for i in must_include if i in by_id]
    rest = [c for c in candidates if c["id"] not in must_include]
    random.Random(seed).shuffle(rest)
    return keep + rest[: max(0, size - len(keep))]


def run_scaling(arms: dict[str, Arm], cases: list[QueryCase], candidates: list[dict],
                sizes=(50, 131, 262), seed: int = 0, ks=(1, 3)) -> dict:
    """Re-run all arms at each PDK size; the KG advantage should grow with distractor density."""
    gold = {g for c in cases for g in c.gold}
    out = {}
    for size in sizes:
        pool = sample_pool(candidates, size, gold, seed)
        out[size] = {"pool_size": len(pool), "arms": run_e1(arms, cases, pool, ks)}
    return out

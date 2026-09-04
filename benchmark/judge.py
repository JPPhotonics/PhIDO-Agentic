"""LLM faithfulness judge (A2/A3 diagnostic) + ontology calibration.

Judges *faithfulness* — "does the evidence quote SUPPORT this triple?" (textual entailment,
NOT scientific truth, per the benchmark's scope boundary). Two bias controls from the lit
review:
* **order-swap** — ask with evidence-before-claim and claim-before-evidence; if the verdict
  flips, that's position bias (reported as ``order_consistent=False``);
* **multi-vote** — N calls per ordering, conservative majority aggregation.

``calibrate_judge`` validates the judge against ``GenerativeOntology`` cases (where the
ontology supplies ground truth) via agreement + Cohen's kappa — this is what licenses
trusting the judge on out-of-ontology edges. ``llm_fn(prompt, sys_prompt) -> str`` is
injectable (mock in tests; wires to ``llm_api.call_llm`` in production).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stats import cohen_kappa

SYS = ("You are checking whether a quoted passage SUPPORTS a stated relationship. Answer "
       "with exactly 'SUPPORTED' or 'NOT_SUPPORTED'. Judge only whether the quote backs the "
       "statement, not whether the statement is true in general.")


def _fmt_triple(triple) -> str:
    if isinstance(triple, dict):
        return f"{triple.get('head')} — {triple.get('relation')} — {triple.get('tail')}"
    return " — ".join(str(x) for x in triple)


def _prompt(triple, quotes: list[str], evidence_first: bool) -> str:
    claim = f"CLAIM: {_fmt_triple(triple)}"
    evidence = "EVIDENCE:\n" + "\n".join(f"- {q}" for q in quotes)
    parts = [evidence, claim] if evidence_first else [claim, evidence]
    return "\n\n".join(parts) + "\n\nDoes the EVIDENCE support the CLAIM? Answer SUPPORTED or NOT_SUPPORTED."


def _parse(resp: str) -> int:
    low = (resp or "").lower()
    if "not_supported" in low or "not supported" in low or "unsupported" in low:
        return 0
    return 1 if "support" in low else 0    # default conservative: not supported


@dataclass
class Verdict:
    label: str            # "supported" | "not_supported"
    n_supported: int
    n_total: int
    order_consistent: bool
    by_order: dict        # {"evidence_first": n_supported, "claim_first": n_supported}


def judge_faithfulness(triple, quotes: list[str], llm_fn: Callable[[str, str], str], votes: int = 3) -> Verdict:
    """Order-swapped, multi-vote faithfulness verdict for one triple."""
    by_order = {}
    for key, ev_first in (("evidence_first", True), ("claim_first", False)):
        p = _prompt(triple, quotes, ev_first)
        by_order[key] = sum(_parse(llm_fn(p, SYS)) for _ in range(votes))
    n_total = 2 * votes
    n_sup = sum(by_order.values())
    # majority within each ordering -> position-bias check
    ec_majority = by_order["evidence_first"] > votes / 2
    ce_majority = by_order["claim_first"] > votes / 2
    return Verdict(
        label="supported" if n_sup > n_total / 2 else "not_supported",
        n_supported=n_sup, n_total=n_total,
        order_consistent=(ec_majority == ce_majority), by_order=by_order,
    )


def calibrate_judge(cases: list[dict], llm_fn: Callable[[str, str], str], votes: int = 3) -> dict:
    """Validate the judge against ontology-gold cases.

    Each case: ``{"triple", "quotes", "gold": "supported"|"not_supported"}``. Returns
    agreement, Cohen's kappa, the order-bias rate, and the misclassified ids.
    """
    pairs, inconsistent, wrong = [], 0, []
    for i, c in enumerate(cases):
        v = judge_faithfulness(c["triple"], c.get("quotes", []), llm_fn, votes)
        gold = 1 if c["gold"] in (1, "supported", "support", True) else 0
        pred = 1 if v.label == "supported" else 0
        pairs.append((gold, pred))
        inconsistent += (not v.order_consistent)
        if gold != pred:
            wrong.append(c.get("id", i))
    n = len(pairs)
    return {
        "n": n,
        "agreement": (sum(1 for g, p in pairs if g == p) / n) if n else float("nan"),
        "cohen_kappa": cohen_kappa(pairs),
        "order_bias_rate": (inconsistent / n) if n else float("nan"),
        "misclassified": wrong,
    }


def make_llm_judge(model: str = "gpt-4o") -> Callable[[str, str], str]:
    """Production llm_fn that routes through llm_api.call_llm (lazy import)."""
    from llm_api import call_llm
    return lambda prompt, sys_prompt: call_llm(prompt, sys_prompt, model)

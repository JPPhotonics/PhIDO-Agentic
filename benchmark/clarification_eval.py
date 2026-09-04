"""B1 — interpreter / clarification evaluation + user simulator.

Three pieces:
* a curated **clear/ambiguous** query set (seed; extend with ambiguous Testbench variants);
* **should-clarify P/R** — does the interpreter ask when (and only when) it should?
* **intent correctness** — field-level match of the extracted DesignIntent vs gold;
* a **user simulator** that answers clarifying questions from a *hidden* gold spec, so the
  E2 clarification on/off ablation can run end-to-end without a human in the loop.

LLM calls go through an injectable ``llm_fn`` (mockable in tests; wires to
``llm_api.call_llm`` in production).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from metrics import prf


@dataclass
class LabeledQuery:
    id: str
    prompt: str
    ambiguous: bool                       # gold: should the interpreter clarify?
    gold_intent: dict = field(default_factory=dict)   # fields a correct intent must contain
    hidden_spec: dict = field(default_factory=dict)   # full spec the user "really wants" (for the simulator)


SEED_QUERIES: list[LabeledQuery] = [
    # --- clear: fully specified, interpreter should NOT clarify ---
    LabeledQuery("clr-dc", "Design a 50/50 directional coupler for C-band with 500nm waveguides, "
                 "2um coupling length, 200nm gap.", False,
                 {"component": "directional_coupler", "ratio": "50/50", "band": "C"}),
    LabeledQuery("clr-mzi", "Two cascaded MZI modulators, 10 GHz bandwidth, 2x2 ports.", False,
                 {"component": "mzi", "n_stages": 2, "bandwidth_GHz": 10, "ports": "2x2"}),
    LabeledQuery("clr-mmi", "Design a 1x2 MMI power splitter for C-band with 500nm waveguides.", False,
                 {"component": "mmi", "ports": "1x2", "band": "C"}),
    # --- ambiguous: key info missing, interpreter SHOULD clarify ---
    LabeledQuery("amb-split", "Design a power splitter.", True,
                 {"component": "splitter"},
                 {"component": "mmi", "ports": "1x4", "band": "C", "ratio": "equal"}),
    LabeledQuery("amb-mod", "I need a modulator.", True,
                 {"component": "modulator"},
                 {"component": "mzi", "bandwidth_GHz": 25, "band": "C", "ports": "1x1"}),
    LabeledQuery("amb-ring", "Make a ring resonator filter.", True,
                 {"component": "ring"},
                 {"component": "ring", "radius_um": 10, "fsr_nm": 8, "band": "C"}),
]


def should_clarify_scores(decisions: list[tuple[bool, bool]]) -> dict:
    """P/R for the 'should clarify' decision. Positive class = ambiguous (ought to clarify).

    ``decisions`` = list of (gold_ambiguous, system_clarified).
    """
    tp = sum(1 for amb, clr in decisions if amb and clr)
    fp = sum(1 for amb, clr in decisions if not amb and clr)   # over-asking on clear prompts
    fn = sum(1 for amb, clr in decisions if amb and not clr)   # missed an ambiguous prompt
    tn = sum(1 for amb, clr in decisions if not amb and not clr)
    p, r, f = prf(tp, fp, fn)
    acc = (tp + tn) / len(decisions) if decisions else float("nan")
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "n": len(decisions)}


def evaluate_clarifier(queries: list[LabeledQuery], decide_fn: Callable[[str], bool]) -> dict:
    """Run a clarify-decider (prompt -> should_clarify?) over the set and score it."""
    return should_clarify_scores([(q.ambiguous, bool(decide_fn(q.prompt))) for q in queries])


def intent_match(pred: dict, gold: dict) -> dict:
    """Field-level correctness of an extracted intent vs gold (over gold's keys)."""
    if not gold:
        return {"accuracy": float("nan"), "matched": 0, "total": 0, "per_field": {}}
    per_field = {k: (str(pred.get(k)).strip().lower() == str(v).strip().lower()) for k, v in gold.items()}
    matched = sum(per_field.values())
    return {"accuracy": matched / len(gold), "matched": matched, "total": len(gold), "per_field": per_field}


class UserSimulator:
    """Answers an interpreter's clarifying questions from a hidden gold spec.

    Enables the B1 clarification on/off ablation and interactive E2 runs with no human.
    ``llm_fn(prompt, sys_prompt) -> str`` is injectable; defaults to ``llm_api.call_llm``.
    """

    SYS = ("You are a photonics engineer with a fixed design in mind. Answer the question "
           "concisely and only from the spec provided. If the spec does not cover it, say "
           "'no preference'.")

    def __init__(self, hidden_spec: dict, llm_fn: Callable[[str, str], str] | None = None):
        self.hidden_spec = hidden_spec
        self._llm_fn = llm_fn

    def _call(self, prompt: str, sys_prompt: str) -> str:
        if self._llm_fn is not None:
            return self._llm_fn(prompt, sys_prompt)
        from llm_api import call_llm  # lazy: only needed in production
        return call_llm(prompt, sys_prompt, "gpt-4o")

    def answer(self, question: str) -> str:
        prompt = f"Hidden design spec: {self.hidden_spec}\n\nEngineer asks: {question}\n\nYour answer:"
        return self._call(prompt, self.SYS).strip()

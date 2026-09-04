"""B1 — should-the-agent-clarify: run the real interpreter clarifier over the labeled set.

Wires `clarification_eval.evaluate_clarifier` to the live pipeline: for each labeled prompt, run
`explore_and_ask` and read its final `{"type":"clarification","request":{...}}` event — the agent
"clarifies" when its ClarificationRequest is `ready_to_proceed=False` (or carries a critical
question). Scored as a binary decision (positive class = ambiguous → ought to clarify): P/R/F1 +
accuracy + confusion.

CAVEATS (state in the thesis): the gold set (`SEED_QUERIES`) is small (N=6, 3 clear / 3 ambiguous)
and single-author-labeled — underpowered and needs a rubric + a second labeler (κ) to be fully
defensible. The clarifier decision is LLM-stochastic; `K_REPEATS` runs each prompt k times and
takes the majority to damp run-to-run noise. This is a working, honestly-scoped result, not a
powered one.

Run: PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv/bin/python <wt>/benchmark/run_b1_clarify.py
"""

from __future__ import annotations

import os
import pathlib

from clarification_eval import SEED_QUERIES, should_clarify_scores
from report import Reporter
from run_pilot_e2 import ROOT  # noqa: F401  (ensures .env + sys.path are set up)

MODEL = os.getenv("PILOT_MODEL", "o3-mini")
K = int(os.getenv("K_REPEATS", "3"))
OUT_MD = pathlib.Path(ROOT) / "benchmark" / "results" / "b1_clarify.md"


def _clarifies_once(prompt: str) -> bool:
    """Run the live interpreter; True iff it would ask the user (not ready_to_proceed / critical Q)."""
    from mcp_servers.interpreter_agent import explore_and_ask

    clarify = False
    for ev in explore_and_ask(prompt, MODEL, 15, 0):
        if ev.get("type") == "clarification":
            req = ev.get("request") or {}
            rtp = req.get("ready_to_proceed")
            qs = req.get("questions") or []
            crit = any(
                isinstance(q, dict) and q.get("priority") == "critical" for q in qs
            )
            clarify = (rtp is False) or crit
    return clarify


def main() -> None:
    decisions, rows = [], []
    for q in SEED_QUERIES:
        votes = [_clarifies_once(q.prompt) for _ in range(K)]
        maj = sum(votes) > len(votes) / 2
        decisions.append((q.ambiguous, maj))
        rows.append(
            [
                q.id,
                "ambiguous" if q.ambiguous else "clear",
                "clarified" if maj else "proceeded",
                "✓" if (q.ambiguous == maj) else "✗",
                f"{sum(votes)}/{K}",
            ]
        )
        print(
            f"[b1] {q.id:10} gold={'amb' if q.ambiguous else 'clr'} "
            f"clarified={maj} votes={sum(votes)}/{K}",
            flush=True,
        )

    s = should_clarify_scores(decisions)
    rep = Reporter(
        OUT_MD,
        "B1 — should-the-agent-clarify (live interpreter)",
        meta={
            "model": MODEL,
            "K_repeats": K,
            "n_prompts": s["n"],
            "accuracy": round(s["accuracy"], 3),
            "f1": round(s["f1"], 3),
        },
    )
    rep.h("Decision scores (positive class = ambiguous → ought to clarify)")
    rep.table(
        ["metric", "value"],
        [
            ["precision", f"{s['precision']:.3f}"],
            ["recall", f"{s['recall']:.3f}"],
            ["F1", f"{s['f1']:.3f}"],
            ["accuracy", f"{s['accuracy']:.3f}"],
            [
                "confusion (tp/fp/fn/tn)",
                f"{s['confusion']['tp']}/{s['confusion']['fp']}/{s['confusion']['fn']}/{s['confusion']['tn']}",
            ],
        ],
    )
    rep.h("Per-prompt (majority over K repeats)")
    rep.table(["id", "gold", "system", "correct", "clarify-votes"], rows)
    rep.line("")
    rep.line(
        "_N=6, single-author labels → underpowered; needs a rubric + second labeler (κ) and a "
        "larger set to be fully defensible. fp = over-asking on clear prompts; fn = missed ambiguity._"
    )
    rep.save()


if __name__ == "__main__":
    main()

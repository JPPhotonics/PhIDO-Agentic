"""Unified gate-evaluation template (BENCHMARK_ARCHITECTURE.md §6.6).

Every "gate" in the pipeline — VSA architecture-integrity (A2), DIA semantic-verification
(A3), Clingo topology gate (B4), AR parameter gate (B5) — is scored by one reusable
pattern:

* **catch rate** — true-positive rate on a labeled *invalid* set (gate should reject);
* **false-reject rate** — false-positive rate on a labeled *valid* set (gate should pass);
* **funnel-ablation Δ** — change in a downstream pass rate (e.g. DRC-clean) with the gate
  on vs off.

**Formal-gate extension (B4/B5):** a correct ASP/AR gate is sound+complete w.r.t. its
encoded rules, so its failure mode is *rule incompleteness*, not stochastic error. We add
per-error-class catch rate, a coverage-gap list (classes never caught), and a soundness
check (any *encoded*-but-accepted invalid).

A *decision record* is a dict::

    {"id": ..., "label": "valid"|"invalid", "rejected": bool, "error_class": optional}

``rejected`` is the gate's verdict (True = gate flagged it invalid).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from stats import paired_mcnemar, wilson


def evaluate_gate(records: Sequence[dict]) -> dict:
    """Catch rate + false-reject rate (with Wilson CIs) over labeled valid/invalid cases."""
    invalid = [r for r in records if r["label"] == "invalid"]
    valid = [r for r in records if r["label"] == "valid"]
    n_caught = sum(1 for r in invalid if r.get("rejected"))
    n_false_rej = sum(1 for r in valid if r.get("rejected"))
    out: dict = {
        "n_invalid": len(invalid),
        "n_valid": len(valid),
        "catch_rate": (n_caught / len(invalid)) if invalid else float("nan"),
        "catch_rate_ci95": wilson(n_caught, len(invalid)) if invalid else None,
        "false_reject_rate": (n_false_rej / len(valid)) if valid else float("nan"),
        "false_reject_ci95": wilson(n_false_rej, len(valid)) if valid else None,
    }
    return out


def formal_gate_eval(records: Sequence[dict]) -> dict:
    """evaluate_gate + per-error-class catch, coverage gap, and a soundness summary.

    ``coverage_gap`` = error classes the gate never catches (likely *not encoded* in the
    rules). ``soundness`` here = the count of invalids the gate accepted, broken out so you
    can argue whether failures are un-encoded classes (expected) vs. encoded-but-missed
    (a real soundness violation — investigate).
    """
    base = evaluate_gate(records)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["label"] == "invalid":
            by_class[r.get("error_class", "_unspecified")].append(r)

    per_class = {}
    coverage_gap = []
    for cls, rs in sorted(by_class.items()):
        caught = sum(1 for r in rs if r.get("rejected"))
        rate = caught / len(rs)
        per_class[cls] = {"n": len(rs), "catch_rate": rate, "catch_ci95": wilson(caught, len(rs))}
        if rate == 0.0:
            coverage_gap.append(cls)

    accepted_invalid = [r for r in records if r["label"] == "invalid" and not r.get("rejected")]
    base.update({
        "per_error_class": per_class,
        "coverage_gap": coverage_gap,
        "n_error_classes": len(by_class),
        "soundness": {
            "n_invalid_accepted": len(accepted_invalid),
            "accepted_classes": sorted({r.get("error_class", "_unspecified") for r in accepted_invalid}),
            "note": "un-encoded classes are expected gaps; encoded-but-accepted = soundness violation",
        },
    })
    return base


def funnel_ablation_delta(
    downstream_with_gate: Sequence[int],
    downstream_without_gate: Sequence[int],
    stage: str = "DRC-clean",
) -> dict:
    """Δ in a downstream binary outcome (e.g. DRC-clean) for gate-on vs gate-off.

    Paired over the same prompts; significance via McNemar. Topology/parameter errors
    manifest upstream of DRC, so the gate's value shows here, not just in its own catch rate.
    """
    if len(downstream_with_gate) != len(downstream_without_gate):
        raise ValueError("on/off outcome vectors must be paired (same prompts)")
    on = sum(downstream_with_gate) / len(downstream_with_gate) if downstream_with_gate else float("nan")
    off = sum(downstream_without_gate) / len(downstream_without_gate) if downstream_without_gate else float("nan")
    return {
        "stage": stage,
        "rate_gate_on": on,
        "rate_gate_off": off,
        "delta": on - off,
        "mcnemar": paired_mcnemar(downstream_with_gate, downstream_without_gate),
        "n": len(downstream_with_gate),
    }

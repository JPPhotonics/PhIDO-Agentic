"""Post-hoc significance tests for the E2 9-arm correctness ablation.

Reads results/_ablation_correctness_gpt54_v2.json (per arm -> per prompt -> list of reps).
No LLM calls: pure arithmetic over stored per-rep outcomes.

For each agentic arm vs baseline (paired on prompt id, n=24):
  - paired Wilcoxon signed-rank on prompt-averaged compF1, edgeF1, ged
  - McNemar (exact binomial) on binarized "exactly correct" (compF1==1.0),
    prompt-level correct = majority of scored reps perfect (>50%)
Holm-Bonferroni correction applied across the 8 arm comparisons, per metric family.
"""

import json
import math
from pathlib import Path

from scipy.stats import wilcoxon

RESULTS = Path(__file__).parent / "results" / "_ablation_correctness_gpt54_v2.json"
BASELINE = "baseline"


def scored_reps(reps):
    return [r for r in reps if r.get("status") == "scored"]


def prompt_mean(reps, metric):
    """Mean of a metric over scored reps; None if no scored reps."""
    s = scored_reps(reps)
    if not s:
        return None
    return sum(r[metric] for r in s) / len(s)


def prompt_correct(reps):
    """Majority of scored reps exactly correct (compF1==1.0). None if no scored reps."""
    s = scored_reps(reps)
    if not s:
        return None
    perfect = sum(1 for r in s if r["compF1"] == 1.0)
    return perfect / len(s) > 0.5


def paired_metric(arm_data, metric):
    """Return (baseline_vals, arm_vals) paired over prompts with both scored."""
    b, a = [], []
    for pid in sorted(base_data):
        bm = prompt_mean(base_data[pid], metric)
        am = prompt_mean(arm_data.get(pid, []), metric)
        if bm is not None and am is not None:
            b.append(bm)
            a.append(am)
    return b, a


def wilcoxon_test(b, a):
    diffs = [ai - bi for ai, bi in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return {"n": len(b), "n_nonzero": 0, "stat": None, "p": 1.0,
                "median_delta": 0.0}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    sd = sorted(diffs)
    n = len(sd)
    median = sd[n // 2] if n % 2 else (sd[n // 2 - 1] + sd[n // 2]) / 2
    return {"n": len(b), "n_nonzero": len(nonzero), "stat": float(stat),
            "p": float(p), "median_delta": median}


def mcnemar_exact(arm_data):
    """Discordant-pair exact binomial (two-sided) on prompt-level 'exactly correct'."""
    b01 = b10 = concordant = 0  # b01: baseline wrong, arm right; b10: baseline right, arm wrong
    for pid in sorted(base_data):
        bc = prompt_correct(base_data[pid])
        ac = prompt_correct(arm_data.get(pid, []))
        if bc is None or ac is None:
            continue
        if bc == ac:
            concordant += 1
        elif not bc and ac:
            b01 += 1
        elif bc and not ac:
            b10 += 1
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        # two-sided exact binomial at q=0.5
        p = min(1.0, 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1)))
    return {"arm_gains": b01, "baseline_gains": b10, "concordant": concordant,
            "n_discordant": n, "p": p}


def holm(pairs):
    """pairs: list of (label, p). Returns dict label -> (p_raw, p_holm, reject@0.05)."""
    m = len(pairs)
    ordered = sorted(pairs, key=lambda x: x[1])
    out = {}
    running_max = 0.0
    for i, (label, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        out[label] = (p, running_max, running_max < 0.05)
    return out


d = json.loads(RESULTS.read_text())
base_data = d[BASELINE]
arms = [a for a in d if a != BASELINE]

# Collect raw p-values per metric family
wilcox_raw = {m: [] for m in ("compF1", "edgeF1", "ged")}
mcnemar_raw = []
detail = {}

for arm in arms:
    detail[arm] = {}
    for m in ("compF1", "edgeF1", "ged"):
        b, a = paired_metric(d[arm], m)
        res = wilcoxon_test(b, a)
        detail[arm][m] = res
        wilcox_raw[m].append((arm, res["p"]))
    mc = mcnemar_exact(d[arm])
    detail[arm]["mcnemar"] = mc
    mcnemar_raw.append((arm, mc["p"]))

wilcox_holm = {m: holm(wilcox_raw[m]) for m in wilcox_raw}
mcnemar_holm = holm(mcnemar_raw)


def star(reject):
    return " *" if reject else ""


print("=" * 78)
print("E2 9-arm ablation — significance vs baseline (paired on prompt, n=24)")
print("Holm-Bonferroni across 8 arm comparisons, per metric family")
print("=" * 78)

for m, better in (("compF1", "higher"), ("edgeF1", "higher"), ("ged", "lower")):
    print(f"\n### Wilcoxon signed-rank: {m} ({better} is better)")
    print(f"{'arm':<20}{'n':>4}{'med Δ':>10}{'p_raw':>10}{'p_holm':>10}  sig")
    for arm in arms:
        r = detail[arm][m]
        _, p_holm, rej = wilcox_holm[m][arm]
        print(f"{arm:<20}{r['n']:>4}{r['median_delta']:>10.3f}"
              f"{r['p']:>10.4f}{p_holm:>10.4f}{star(rej)}")

print("\n### McNemar exact (prompt-level 'exactly correct', compF1==1.0, majority of reps)")
print(f"{'arm':<20}{'arm+':>6}{'base+':>7}{'conc':>6}{'p_raw':>10}{'p_holm':>10}  sig")
for arm in arms:
    mc = detail[arm]["mcnemar"]
    _, p_holm, rej = mcnemar_holm[arm]
    print(f"{arm:<20}{mc['arm_gains']:>6}{mc['baseline_gains']:>7}"
          f"{mc['concordant']:>6}{mc['p']:>10.4f}{p_holm:>10.4f}{star(rej)}")

print("\narm+  = prompts where arm correct but baseline wrong (discordant, favors arm)")
print("base+ = prompts where baseline correct but arm wrong (discordant, favors baseline)")
print("* = reject H0 at Holm-adjusted alpha=0.05")

# machine-readable dump
out = {"wilcoxon": {}, "mcnemar": {}}
for arm in arms:
    out["wilcoxon"][arm] = {
        m: {**detail[arm][m], "p_holm": wilcox_holm[m][arm][1],
            "reject": wilcox_holm[m][arm][2]} for m in ("compF1", "edgeF1", "ged")}
    mc = detail[arm]["mcnemar"]
    out["mcnemar"][arm] = {**mc, "p_holm": mcnemar_holm[arm][1],
                           "reject": mcnemar_holm[arm][2]}
dump = RESULTS.parent / "_ablation_significance.json"
dump.write_text(json.dumps(out, indent=2))
print(f"\nwrote {dump}")

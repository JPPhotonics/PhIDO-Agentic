"""Shared benchmark statistics (stdlib only).

Paired comparisons (E1/E2 arms on the same prompts): ``mcnemar`` / ``paired_mcnemar``,
``bootstrap_ci``. Proportion estimation: ``wilson``. Clustered estimation (expert anchor):
``cluster_t_interval``, ``icc_oneway``, ``design_effect``. Agreement: ``cohen_kappa``.

Per KB_BENCHMARK_PLAN.md §6.3: McNemar for funnel/DRC pass rates, bootstrap CIs for
pass@k; no fixed seed in the pipeline -> report k-repeat variance.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Sequence

# Two-sided 95% Student-t critical values; falls back toward z=1.96 for large df.
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    return 1.96 if df > 30 else _T95[min(_T95, key=lambda k: abs(k - df))]


# ------------------------------------------------------------------- proportion (Wilson)
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (clamped to [0,1])."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (clamp01(centre - half), clamp01(centre + half))


# ----------------------------------------------------------------------- McNemar (paired)
def mcnemar(b: int, c: int, exact_max: int = 25) -> dict[str, float]:
    """McNemar's test on the two discordant counts.

    ``b`` = cases where arm A succeeds and arm B fails; ``c`` = the reverse. Uses the
    exact binomial p-value when ``b+c`` is small, else the chi-square approximation with
    continuity correction (df=1 survival via erfc).
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "method": "none"}
    if n <= exact_max:
        # exact: two-sided binomial against p=0.5
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return {"b": b, "c": c, "n_discordant": n, "p_value": min(1.0, 2 * tail), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n  # chi-square, df=1, continuity-corrected
    p = math.erfc(math.sqrt(stat / 2))
    return {"b": b, "c": c, "n_discordant": n, "statistic": stat, "p_value": p, "method": "chi2_cc"}


def paired_mcnemar(outcomes_a: Sequence[int], outcomes_b: Sequence[int]) -> dict[str, float]:
    """McNemar from two aligned 0/1 outcome vectors (same items, two arms)."""
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("outcome vectors must be the same length (paired)")
    b = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if a and not bb)
    c = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if bb and not a)
    return mcnemar(b, c)


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ci(
    data: Sequence,
    statistic: Callable[[Sequence], float],
    n_boot: int = 2000,
    conf: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for an arbitrary statistic (e.g. pass@k over queries)."""
    data = list(data)
    if not data:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(data)
    boots = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    lo_i = int((1 - conf) / 2 * n_boot)
    hi_i = int((1 + conf) / 2 * n_boot) - 1
    return {"point": statistic(data), "lo": boots[lo_i], "hi": boots[max(lo_i, hi_i)], "n_boot": n_boot}


# ------------------------------------------------------------------- clustered estimation
def icc_oneway(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    """One-way random-effects ICC(1) and mean cluster size, from grouped 0/1 (or real) data."""
    groups = [list(g) for g in groups if g]
    k = len(groups)
    if k < 2:
        return (float("nan"), float("nan"))
    n_i = [len(g) for g in groups]
    total = sum(n_i)
    grand = sum(sum(g) for g in groups) / total
    msb = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    within = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
    msw = within / (total - k) if total > k else 0.0
    m0 = (total - sum(n * n for n in n_i) / total) / (k - 1)
    denom = msb + (m0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    return (clamp01(icc), m0)


def design_effect(mean_cluster_size: float, icc: float) -> float:
    return 1 + (mean_cluster_size - 1) * icc


def cluster_t_interval(groups: Sequence[Sequence[float]], conf: float = 0.95) -> dict[str, float]:
    """Cluster-level t-interval (clusters = e.g. documents). Honest for few clusters.

    With few clusters this interval is wide and fragile (df = #clusters - 1) — that is the
    point. Report it as exploratory below ~10 clusters.
    """
    means = [statistics.mean(g) for g in groups if g]
    k = len(means)
    if k < 2:
        return {"note": f"only {k} cluster(s); no clustered CI", "n_clusters": k}
    m = statistics.mean(means)
    sd = statistics.stdev(means)
    half = t95(k - 1) * sd / math.sqrt(k)
    icc, m0 = icc_oneway(groups)
    deff = design_effect(m0, icc) if not math.isnan(m0) else float("nan")
    n_total = sum(len(g) for g in groups)
    return {"rate": m, "lo": clamp01(m - half), "hi": clamp01(m + half), "n_clusters": k,
            "df": k - 1, "icc": icc, "mean_cluster_size": m0, "design_effect": deff,
            "effective_n": (n_total / deff) if deff and not math.isnan(deff) else None}


# --------------------------------------------------------------------------- agreement
def cohen_kappa(pairs: Sequence[tuple[int, int]]) -> float:
    """Cohen's kappa for two raters over binary judgments."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

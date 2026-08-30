"""Confidence intervals and exact tests for benchmark proportions.

All estimators here are chosen for validity at small n and near boundary
success rates — the regime where agent benchmarks actually live. Wald
intervals (the default in most eval scripts) are wrong exactly there.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist, mean

__all__ = [
    "Interval",
    "z_two_sided",
    "wilson_interval",
    "wilson_interval_roots",
    "newcombe_diff_interval",
    "two_proportion_test",
    "mcnemar_exact",
    "binom_cdf",
    "cluster_bootstrap_ci",
]


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    alpha: float
    method: str

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def half(self) -> float:
        return self.width / 2

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    def excludes(self, value: float) -> bool:
        return value < self.low or value > self.high

    def __str__(self) -> str:
        return f"[{self.low:+.4f}, {self.high:+.4f}] ({self.method}, {1 - self.alpha:.0%})"


def z_two_sided(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return NormalDist().inv_cdf(1 - alpha / 2)


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    Valid at n as small as 1 and for p near 0 or 1, where the Wald interval
    collapses (e.g. 0/50 -> Wald [-0.0, 0.0], Wilson [0.0, 0.072]).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n], got {successes}/{n}")
    z = z_two_sided(alpha)
    z2 = z * z
    p = successes / n
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return Interval(max(0.0, center - half), min(1.0, center + half), alpha, "wilson")


def wilson_interval_roots(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson interval via direct quadratic roots.

    Independent derivation of the same interval, used as a cross-check in the
    test suite: (p_hat - p)^2 * n = z^2 * p * (1 - p).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n], got {successes}/{n}")
    z = z_two_sided(alpha)
    z2 = z * z
    p = successes / n
    a = n + z2
    b = -(2 * n * p + z2)
    c = n * p * p
    disc = b * b - 4 * a * c
    if disc < 0:  # cannot happen; guard for float safety
        disc = 0.0
    r = math.sqrt(disc)
    return Interval(max(0.0, (-b - r) / (2 * a)), min(1.0, (-b + r) / (2 * a)), alpha, "wilson")


def newcombe_diff_interval(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> Interval:
    """Newcombe hybrid-score CI for p1 - p2 (unpaired).

    Composes the two Wilson intervals; keeps near-boundary validity where the
    Wald CI for a difference overstates certainty. Newcombe, Stat. Med. 17
    (1998), method 10.
    """
    for p, n in ((p1, n1), (p2, n2)):
        if not 0 <= p <= 1:
            raise ValueError(f"proportion out of [0,1]: {p}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
    s1, s2 = round(p1 * n1), round(p2 * n2)
    l1, u1 = wilson_interval(s1, n1, alpha).low, wilson_interval(s1, n1, alpha).high
    l2, u2 = wilson_interval(s2, n2, alpha).low, wilson_interval(s2, n2, alpha).high
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return Interval(max(-1.0, lower), min(1.0, upper), alpha, "newcombe")


@dataclass(frozen=True)
class TwoPropResult:
    diff: float
    ci: Interval
    z: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.ci.alpha


def two_proportion_test(s1: int, n1: int, s2: int, n2: int, alpha: float = 0.05) -> TwoPropResult:
    """Pooled z-test for two proportions + Newcombe CI.

    The z-test decides significance (p-value), the Newcombe interval reports
    the effect with honest width. Reporting both is intentional: with small n
    they can disagree, and that disagreement is signal, not bug.
    """
    for s, n in ((s1, n1), (s2, n2)):
        if not 0 <= s <= n:
            raise ValueError(f"successes must be in [0, n], got {s}/{n}")
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        z = 0.0 if p1 == p2 else math.copysign(math.inf, p1 - p2)
    else:
        z = (p1 - p2) / se
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))
    return TwoPropResult(p1 - p2, newcombe_diff_interval(p1, n1, p2, n2, alpha), z, min(1.0, p_value))


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact binomial CDF, stable via log-gamma."""
    if not 0 <= k or k > n:
        raise ValueError(f"k must be in [0, n], got {k}/{n}")
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    log_p = math.log(p)
    log_q = math.log(1 - p)
    terms = []
    for i in range(k + 1):
        logpmf = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * log_p + (n - i) * log_q
        )
        terms.append(math.exp(logpmf))
    total = math.fsum(terms)
    if total > 1.0 - 1e-12:
        return 1.0
    return min(1.0, total)


@dataclass(frozen=True)
class McNemarResult:
    b: int
    c: int
    diff: float
    ci: Interval
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.ci.alpha


def mcnemar_exact(b: int, c: int, alpha: float = 0.05) -> McNemarResult:
    """Exact McNemar test for paired binary outcomes + CI for the diff.

    b = tasks A solved and B failed, c = tasks A failed and B solved. The test
    conditions on discordant pairs: under H0, b ~ Binomial(b + c, 1/2). This
    is the correct instrument for benchmark A/B on a fixed task set; running
    two independent z-tests instead discards the pairing and burns power.

    CI for d = (b - c)/N follows the paired-proportion interval
    var(d) = (b + c - (b - c)^2 / N) / N^2.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b, c must be >= 0, got {b}, {c}")
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, Interval(0.0, 0.0, alpha, "mcnemar-degenerate"), 1.0)
    x = min(b, c)
    p_value = min(1.0, 2 * binom_cdf(x, n, 0.5))
    d = (b - c) / n
    var_d = max(0.0, (n - (b - c) ** 2 / n) / (n * n))
    half = z_two_sided(alpha) * math.sqrt(var_d)
    return McNemarResult(b, c, d, Interval(d - half, d + half, alpha, "mcnemar-wald"), p_value)


def cluster_bootstrap_ci(
    clusters: list[list[float]],
    stat=None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI that resamples clusters (tasks), not runs.

    Benchmark observations are not iid: runs on the same task share task
    difficulty. A run-level bootstrap pretends every run is exchangeable and
    understates variance when task effects dominate — the classic
    pseudoreplication failure. Resampling task clusters is the fix.

    Deterministic given `seed`.
    """
    if B < 2:
        raise ValueError(f"B must be >= 2, got {B}")
    if not clusters:
        raise ValueError("no clusters given")
    if stat is None:
        stat = mean
    rng = random.Random(seed)
    k = len(clusters)
    stats = []
    for _ in range(B):
        sample = [clusters[rng.randrange(k)] for _ in range(k)]
        flat = [v for cl in sample for v in cl]
        stats.append(stat(flat))
    stats.sort()
    lo = stats[max(0, math.floor(alpha / 2 * B) - 1)]
    hi = stats[min(B - 1, math.ceil((1 - alpha / 2) * B) - 1)]
    return Interval(lo, hi, alpha, "cluster-bootstrap")

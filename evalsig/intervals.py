"""Confidence intervals and exact tests for benchmark proportions.

All estimators here are chosen for validity at small n and near boundary
success rates — the regime where agent benchmarks actually live. Wald
intervals (the default in most eval scripts) are wrong exactly there.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from statistics import NormalDist, StatisticsError, mean

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
    """Exact binomial CDF via exact rational arithmetic.

    p is converted to its exact binary rational, the pmf recurrence is
    carried out in exact fractions, and the sum is rounded to a double once.
    The result therefore depends on no libm (no lgamma/log/exp/fsum), is the
    correctly-rounded true value, and is bit-identical across platforms and
    across the reference Python and every ported implementation.
    """
    if not 0 <= k or k > n:
        raise ValueError(f"k must be in [0, n], got {k}/{n}")
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    pf = Fraction(p)
    if pf == Fraction(1, 2):
        s = sum(math.comb(n, i) for i in range(k + 1))
        r = Fraction(s, 1 << n)
    else:
        qf = 1 - pf
        pmf = qf**n
        total = pmf
        for i in range(1, k + 1):
            pmf = pmf * (n - i + 1) * pf / (i * qf)
            total += pmf
        r = total
    if r > 1 - Fraction(1, 10**12):
        return 1.0
    return float(min(r, Fraction(1)))


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


def _exact_scaled_ints(values: list) -> tuple[int, list[int]] | None:
    """Exact dyadic representation of finite floats under a common scale.

    Returns `(scale, ints)` with `values[i] == ints[i] / scale` exactly, or
    `None` when any value is not a finite float (the caller then falls back
    to the generic path). Every finite float is m * 2^-e, so a common scale
    2^e_max makes all values integers and integer sums exact.
    """
    ratios = []
    e_max = 0
    for v in values:
        if type(v) is not float or not math.isfinite(v):
            return None
        m, d = v.as_integer_ratio()
        e = d.bit_length() - 1
        if e > e_max:
            e_max = e
        ratios.append((m, e))
    scale = 1 << e_max
    ints = [m << (e_max - e) for m, e in ratios]
    return scale, ints


def _percentile_interval(stats: list, B: int, alpha: float, method: str) -> Interval:
    stats.sort()
    lo = stats[max(0, math.floor(alpha / 2 * B) - 1)]
    hi = stats[min(B - 1, math.ceil((1 - alpha / 2) * B) - 1)]
    return Interval(lo, hi, alpha, method)


def _exact_cluster_sums(clusters: list) -> tuple[int, list[int], list[int]] | None:
    """Exact per-cluster sums as integers on a common power-of-two scale."""
    flat = [v for cl in clusters for v in cl]
    rep = _exact_scaled_ints(flat)
    if rep is None:
        return None
    scale, ints = rep
    sums = []
    counts = []
    pos = 0
    for cl in clusters:
        n = len(cl)
        sums.append(sum(ints[pos:pos + n]))
        counts.append(n)
        pos += n
    return scale, sums, counts


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

    Deterministic given `seed`. The default statistic (the pooled mean) takes
    an exact-arithmetic fast path: per-cluster sums are precomputed as
    integers on a common power-of-two scale and each resample is one integer
    sum plus one correctly-rounded division — bit-identical to the generic
    `statistics.mean` path, ~10x faster.
    """
    if B < 2:
        raise ValueError(f"B must be >= 2, got {B}")
    if not clusters:
        raise ValueError("no clusters given")
    rng = random.Random(seed)
    k = len(clusters)
    if stat is None:
        rep = _exact_cluster_sums(clusters)
        if rep is not None:
            scale, sums, counts = rep
            randrange = rng.randrange
            stats = []
            append = stats.append
            for _ in range(B):
                total = 0
                cnt = 0
                for _ in range(k):
                    i = randrange(k)
                    total += sums[i]
                    cnt += counts[i]
                if cnt == 0:
                    raise StatisticsError("mean requires at least one data point")
                append(total / (scale * cnt))
            return _percentile_interval(stats, B, alpha, "cluster-bootstrap")
        stat = mean
    stats = []
    for _ in range(B):
        sample = [clusters[rng.randrange(k)] for _ in range(k)]
        flat = [v for cl in sample for v in cl]
        stats.append(stat(flat))
    return _percentile_interval(stats, B, alpha, "cluster-bootstrap")

"""Power analysis: how many reruns before a decision means anything.

Every comparison tool can tell you A beat B. This module tells you whether
you ran enough trials for that sentence to carry information, and — the part
that saves the most money — how many more you need if not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from .intervals import z_two_sided

__all__ = ["PowerPlan", "n_two_proportions", "n_paired", "n_for_ci_width"]


@dataclass(frozen=True)
class PowerPlan:
    n_per_version: int
    alpha: float
    power: float
    method: str
    note: str = ""


def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> PowerPlan:
    """Minimum n per version to detect p1 vs p2 (unpaired z-test).

    Normal approximation with pooled-variance form:
        n = (z_{a/2} sqrt(2 p_bar q_bar) + z_b sqrt(p1 q1 + p2 q2))^2 / delta^2.
    """
    _validate(p1, p2, alpha, power)
    delta = abs(p1 - p2)
    if delta == 0:
        raise ValueError("p1 == p2: no effect to detect, n is undefined")
    z_a = z_two_sided(alpha)
    z_b = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (
        z_a * math.sqrt(2 * pbar * (1 - pbar))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n = math.ceil(num / (delta * delta))
    return PowerPlan(max(1, n), alpha, power, "two-proportion z")


def n_paired(
    expected_discordance: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerPlan:
    """Minimum total N (paired, fixed task set) for McNemar.

    delta = |p1 - p2| is the win-rate difference; expected_discordance is the
    fraction of tasks you expect to flip between versions (b + c)/N — from a
    pilot run, or 0.3 as a common first guess for agent benchmarks.

    N = (z_{a/2} + z_b)^2 * psi / delta^2 with psi = discordance, derived from
    the paired variance of d = (b - c)/N.
    """
    _validate(0.5, 0.5, alpha, power)
    if not 0 < expected_discordance <= 1:
        raise ValueError(f"discordance must be in (0, 1], got {expected_discordance}")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    z_a = z_two_sided(alpha)
    z_b = NormalDist().inv_cdf(power)
    n = math.ceil((z_a + z_b) ** 2 * expected_discordance / (delta * delta))
    return PowerPlan(
        max(1, n), alpha, power, "mcnemar",
        note="paired on the same task set; multiply-unpaired designs need far more",
    )


def n_for_ci_width(width: float, p: float = 0.5, alpha: float = 0.05) -> PowerPlan:
    """Minimum n for a CI half-width <= width/2 at success rate p.

    Wald-based planning formula n = z^2 p(1-p) / (width/2)^2; conservative
    because it ignores Wilson's shrinkage, which is the right direction for
    planning.
    """
    if width <= 0 or width >= 1:
        raise ValueError(f"width must be in (0, 1), got {width}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    z = z_two_sided(alpha)
    half = width / 2
    n = math.ceil(z * z * p * (1 - p) / (half * half))
    return PowerPlan(max(1, n), alpha, 0.0, "ci-width")


def _validate(p1, p2, alpha, power):
    for p in (p1, p2):
        if not 0 <= p <= 1:
            raise ValueError(f"proportion out of [0,1]: {p}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0,1), got {power}")

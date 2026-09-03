"""Sequential stopping: end the eval when the number, not the budget, says so.

The statistical hazard of "run until it looks significant" is real and mostly
invisible: peeking at a CI after every batch and stopping on the first hit
inflates the false-positive rate far above the nominal alpha. The honest
fixes are (a) a pre-registered look schedule with alpha correction, and
(b) stopping on precision (CI width), which does not inflate type-I error at
all. This module implements both, conservatively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

from .intervals import wilson_interval, z_two_sided

__all__ = ["LookSchedule", "SequentialProportion", "SequentialAB", "Verdict"]

Verdict = str  # "CONFIRMED" | "NOT-DIFFERENT" | "PRECISION-REACHED" | "CONTINUE" | "BUDGET-EXHAUSTED"


@dataclass(frozen=True)
class LookSchedule:
    """Pre-registered doubling look schedule with Bonferroni alpha spending.

    Checks happen only at n = n0, 2*n0, 4*n0, ... up to max_looks. Each look
    tests at alpha/K (Bonferroni over all K looks), which keeps the family
    type-I error <= alpha under the registered schedule. Max-looks defaults
    to 10, i.e. up to 512*n0 samples — plenty for benchmark noise.
    """

    n0: int = 25
    max_looks: int = 10

    def ns(self) -> list:
        return [self.n0 * (2 ** k) for k in range(self.max_looks)]

    def n_max(self) -> int:
        return self.n0 * (2 ** (self.max_looks - 1))

    def look_for(self, n: int) -> int:
        """Index of the largest scheduled look <= n, or -1 before the first."""
        last = -1
        for i, t in enumerate(self.ns()):
            if t <= n:
                last = i
            else:
                break
        return last


def _boundary_ok(ok: bool, stopped_early: bool) -> str:
    return "boundary valid (registered schedule, Bonferroni)" if ok else ""


@dataclass
class SequentialProportion:
    """Track one proportion; stop on significance vs a reference or on width.

    Width-based stopping ("PRECISION-REACHED") is the estimator-friendly exit:
    it ends the run once the CI is tight enough to be useful, with no effect
    on error rates. Significance checking only fires at scheduled looks.
    """

    reference: float = 0.5
    alpha: float = 0.05
    target_half_width: float = 0.05
    schedule: LookSchedule = field(default_factory=LookSchedule)
    successes: int = 0
    n: int = 0
    looks_used: int = 0
    verdict: Verdict = "CONTINUE"

    def update(self, successes: int, n: int) -> Verdict:
        if n < self.n or successes < 0 or successes > n:
            raise ValueError(f"non-monotone update: ({successes},{n}) after ({self.successes},{self.n})")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.successes, self.n = successes, n
        if self.verdict != "CONTINUE":
            return self.verdict
        k = self.schedule.max_looks
        alpha_look = self.alpha / k
        p_hat = successes / n
        # width exit: plain-alpha Wilson width, any n (no error-rate cost)
        if wilson_interval(successes, n, self.alpha).half <= self.target_half_width:
            self.verdict = "PRECISION-REACHED"
            return self.verdict
        look = self.schedule.look_for(n)
        if look >= 0 and look > self.looks_used - 1:
            self.looks_used = look + 1
            # z-test vs reference at Bonferroni-corrected alpha, scheduled looks only
            if n * p_hat * (1 - p_hat) > 0:
                se = math.sqrt(p_hat * (1 - p_hat) / n)
                z = (p_hat - self.reference) / se
                p_val = 2 * (1 - NormalDist().cdf(abs(z)))
                if p_val < alpha_look:
                    self.verdict = "CONFIRMED"
                    return self.verdict
        if n >= self.schedule.n_max():
            self.verdict = "BUDGET-EXHAUSTED"
        return self.verdict

    def ci(self) -> object:
        """Plain-alpha Wilson CI at the current n (reporting, not testing)."""
        return wilson_interval(self.successes, self.n, self.alpha)

    def report(self) -> dict:
        p = self.successes / self.n if self.n else float("nan")
        return {
            "p_hat": p,
            "n": self.n,
            "verdict": self.verdict,
            "ci": str(self.ci()),
            "looks_used": self.looks_used,
            "test_alpha_per_look": self.alpha / self.schedule.max_looks,
        }


@dataclass
class SequentialAB:
    """Sequential A/B on two run streams of binary outcomes.

    Same discipline as SequentialProportion: scheduled looks with Bonferroni
    alpha for the significance exit, any-n width exit for precision. Use the
    paired McNemar machinery in `decide` for final analysis; this class is
    the online gate that tells the runner when to stop spending.
    """

    alpha: float = 0.05
    target_half_width: float = 0.05
    schedule: LookSchedule = field(default_factory=LookSchedule)
    s1: int = 0
    n1: int = 0
    s2: int = 0
    n2: int = 0
    looks_used: int = 0
    verdict: Verdict = "CONTINUE"

    def update(self, s1: int, n1: int, s2: int, n2: int) -> Verdict:
        for s, n in ((s1, n1), (s2, n2)):
            if not 0 <= s <= n:
                raise ValueError(f"successes out of range: {s}/{n}")
        self.s1, self.n1, self.s2, self.n2 = s1, n1, s2, n2
        if self.verdict != "CONTINUE":
            return self.verdict
        k = self.schedule.max_looks
        alpha_look = self.alpha / k
        d = s1 / n1 - s2 / n2 if n1 and n2 else 0.0
        if n1 == n2 and n1 > 0:
            half = z_two_sided(self.alpha) * math.sqrt(
                (s1 / n1) * (1 - s1 / n1) / n1 + (s2 / n2) * (1 - s2 / n2) / n2
            )
            if half <= self.target_half_width:
                self.verdict = "PRECISION-REACHED"
                return self.verdict
        look = self.schedule.look_for(min(n1, n2))
        if look >= 0 and look > self.looks_used - 1 and min(n1, n2) > 0:
            self.looks_used = look + 1
            p1, p2 = s1 / n1, s2 / n2
            pooled = (s1 + s2) / (n1 + n2)
            se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
            if se > 0:
                z = (p1 - p2) / se
                p_val = 2 * (1 - NormalDist().cdf(abs(z)))
                if p_val < alpha_look:
                    self.verdict = "CONFIRMED"
                    return self.verdict
        if min(n1, n2) >= self.schedule.n_max():
            self.verdict = "BUDGET-EXHAUSTED"
        return self.verdict

    def report(self) -> dict:
        p1 = self.s1 / self.n1 if self.n1 else float("nan")
        p2 = self.s2 / self.n2 if self.n2 else float("nan")
        return {
            "p1": p1,
            "p2": p2,
            "diff": p1 - p2,
            "n1": self.n1,
            "n2": self.n2,
            "verdict": self.verdict,
            "looks_used": self.looks_used,
            "test_alpha_per_look": self.alpha / self.schedule.max_looks,
        }

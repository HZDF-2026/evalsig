"""Variance attribution: where does run-to-run noise actually come from?

One-way random-effects decomposition (method of moments). Given runs grouped
by a suspected noise source (seed, engine version, environment instance),
splits total variance into between-group and within-group components and
reports the ICC — the fraction of noise attributable to the grouping factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

__all__ = ["VarianceComponents", "variance_components", "variance_attribution"]


@dataclass(frozen=True)
class VarianceComponents:
    between: float
    within: float
    total: float
    icc: float
    n_groups: int
    n_obs: int
    group_sizes: tuple

    def summary(self) -> str:
        return (
            f"between={self.between:.4g} within={self.within:.4g} "
            f"icc={self.icc:.3f} (groups={self.n_groups}, obs={self.n_obs})"
        )


def variance_components(groups: dict) -> VarianceComponents:
    """Method-of-moments one-way random effects: groups -> outcome lists.

    sigma^2_between = max(0, (MSB - MSW) / n0), with the unbalanced-design
    effective group size
        n0 = (N - sum(n_i^2) / N) / (a - 1).
    Negative variance estimates are truncated at zero (standard practice; the
    truncation makes the estimator biased conservative, which is the safe
    direction for a noise audit).
    """
    cleaned = {k: [float(v) for v in vals] for k, vals in groups.items() if len(vals) > 0}
    if not cleaned:
        raise ValueError("no non-empty groups")
    sizes = [len(v) for v in cleaned.values()]
    a = len(cleaned)
    N = sum(sizes)
    if a < 2:
        return VarianceComponents(0.0, _m_within(cleaned, N, a), 0.0, 0.0, a, N, tuple(sizes))
    grand = mean([v for vals in cleaned.values() for v in vals])
    ssb = sum(len(vals) * (mean(vals) - grand) ** 2 for vals in cleaned.values())
    msb = ssb / (a - 1)
    msw = _m_within(cleaned, N, a)
    n0 = (N - sum(n * n for n in sizes) / N) / (a - 1)
    if n0 <= 0:
        n0 = 1.0
    between = max(0.0, (msb - msw) / n0)
    within = max(0.0, msw)
    total = between + within
    icc = between / total if total > 0 else 0.0
    return VarianceComponents(between, within, total, icc, a, N, tuple(sizes))


def _m_within(groups: dict, N: int, a: int) -> float:
    if N <= a:
        return 0.0
    ssw = 0.0
    for vals in groups.values():
        m = mean(vals)
        ssw += sum((v - m) ** 2 for v in vals)
    return ssw / (N - a)


def variance_attribution(runs: list, factor: str) -> dict:
    """Split runs by metadata `factor` and decompose outcome variance.

    `runs` items are dicts with a numeric outcome key (success/score/...) and
    arbitrary string metadata keys. Returns per-group means plus components.
    """
    groups: dict = {}
    for r in runs:
        key = r.get(factor)
        if key is None:
            continue
        key = str(key)
        val = _outcome(r)
        groups.setdefault(key, []).append(val)
    comps = variance_components(groups)
    return {
        "factor": factor,
        "components": comps,
        "group_means": {k: mean(v) for k, v in sorted(groups.items())},
    }


def _outcome(run: dict) -> float:
    for k in ("success", "resolved", "passed", "score", "value", "outcome"):
        if k in run:
            v = run[k]
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            return float(v)
    raise ValueError(
        f"run has no outcome field (looked for success/resolved/passed/score/value): {sorted(run)}"
    )

"""The decision layer: statistically sound comparisons and candidate elimination.

This is the API prompt/harness optimizers (DSPy-style pipelines, agent tuning
loops) should call instead of comparing single noisy numbers. Every verdict
comes with an interval, a p-value, and — when inconclusive — the number of
additional runs that would make it conclusive.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist, mean, stdev

from .intervals import Interval, cluster_bootstrap_ci, mcnemar_exact, two_proportion_test
from .power import n_paired, n_two_proportions
from .variance import _outcome

__all__ = ["Run", "load_runs", "Comparison", "compare", "holm", "decide", "DecisionReport"]


@dataclass(frozen=True)
class Run:
    task: str
    outcome: float
    factors: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Run":
        return cls(str(d.get("task", d.get("id", ""))), _outcome(d), {
            k: v for k, v in d.items() if k not in ("task", "id")
        })

    def flat(self) -> dict:
        """Flattened dict: task, outcome, and all factor fields at top level."""
        out = {"task": self.task, "outcome": self.outcome}
        out.update(self.factors)
        return out


def load_runs(path: str) -> list:
    """JSON array of run dicts; see README for the schema."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of runs")
    return [Run.from_dict(d) for d in data]


@dataclass
class Comparison:
    name_a: str
    name_b: str
    estimate: float
    ci: Interval
    p_value: float
    method: str
    paired: bool
    verdict: str
    additional_needed: int = 0

    def __str__(self) -> str:
        if self.additional_needed > 10000:
            extra = (
                f"; resolving an effect this small needs ~{self.additional_needed} runs"
                " — too small to chase, treat as no difference"
            )
        elif self.additional_needed:
            extra = f", ~{self.additional_needed} more runs to resolve"
        else:
            extra = ""
        return (
            f"{self.name_a} vs {self.name_b}: {self.estimate:+.4f} CI {self.ci} "
            f"p={self.p_value:.4g} [{self.method}{'/paired' if self.paired else ''}] "
            f"-> {self.verdict}{extra}"
        )


def _is_binary(runs: list) -> bool:
    return all(r.outcome in (0.0, 1.0) for r in runs)


def _by_task(runs: list) -> dict:
    g = {}
    for r in runs:
        g.setdefault(r.task, []).append(r.outcome)
    return g


def compare(
    runs_a: list,
    runs_b: list,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> Comparison:
    """Compare two run sets on the same tasks (paired) or independently.

    Pairing is auto-detected from task-id overlap — using it when available is
    not an option but the whole point: a fixed task set makes the paired
    design dramatically more powerful than two independent samples.
    """
    if not runs_a or not runs_b:
        raise ValueError("both run sets must be non-empty")
    ta, tb = _by_task(runs_a), _by_task(runs_b)
    overlap = set(ta) & set(tb)
    paired = bool(overlap) and len(overlap) >= max(0.5 * min(len(ta), len(tb)), 1)
    binary = _is_binary(runs_a) and _is_binary(runs_b)
    est_a = mean([r.outcome for r in runs_a])
    est_b = mean([r.outcome for r in runs_b])
    est = est_a - est_b

    if paired:
        diffs = [mean(ta[t]) - mean(tb[t]) for t in overlap]
        ci = cluster_bootstrap_ci([[d] for d in diffs], B=2000, alpha=alpha, seed=1)
        if binary:
            b = sum(1 for t in overlap if mean(ta[t]) > mean(tb[t]))
            c = sum(1 for t in overlap if mean(ta[t]) < mean(tb[t]))
            mc = mcnemar_exact(b, c, alpha)
            p, method = mc.p_value, f"mcnemar (b={b}, c={c})"
            needed = _additional_paired(b, c, est, len(overlap), alpha)
        else:
            p, method = _sign_flip_p(diffs), "paired-bootstrap+signflip"
            needed = 0
        verdict = _verdict(est, ci, p, alpha)
        return Comparison(name_a, name_b, est, ci, p, method, True, verdict, needed)

    if binary:
        res = two_proportion_test(
            round(sum(r.outcome for r in runs_a)), len(runs_a),
            round(sum(r.outcome for r in runs_b)), len(runs_b),
            alpha,
        )
        ci, p, method = res.ci, res.p_value, "two-prop-z"
    else:
        ci = _diff_bootstrap(_by_task(runs_a), _by_task(runs_b), alpha)
        p, method = float("nan"), "bootstrap-diff"
    verdict = _verdict(est, ci, p, alpha)
    needed = _additional_unpaired(est_a, est_b, len(runs_a), len(runs_b), alpha, binary)
    return Comparison(name_a, name_b, est, ci, p, method, False, verdict, needed)


def _verdict(est: float, ci: Interval, p: float, alpha: float) -> str:
    if p != p:  # NaN: bootstrap-only path, the CI carries the decision
        if ci.excludes(0):
            return "A-BETTER" if est > 0 else "B-BETTER"
        return "INCONCLUSIVE"
    if p < alpha and ci.excludes(0):
        return "A-BETTER" if est > 0 else "B-BETTER"
    return "INCONCLUSIVE"


def _sign_flip_p(diffs: list) -> float:
    """Two-sided sign-flip permutation p-value for the mean of paired diffs.

    Exact under symmetry of the diff distribution; no small-n normality
    assumptions. Enumeration is 2^n, so n > 20 falls back to the normal
    approximation to the sign-flip null.
    """
    if not diffs:
        return 1.0
    obs = abs(mean(diffs))
    n = len(diffs)
    if n > 20:
        se = stdev(diffs) / math.sqrt(n)
        if se == 0:
            return 1.0
        return min(1.0, 2 * (1 - NormalDist().cdf(obs / se)))
    count = 0
    for mask in range(1 << n):
        s = 0.0
        for i, d in enumerate(diffs):
            s += d if (mask >> i) & 1 else -d
        if abs(s / n) >= obs - 1e-12:
            count += 1
    return count / (1 << n)


def _diff_bootstrap(ta: dict, tb: dict, alpha: float) -> Interval:
    """Task-cluster bootstrap CI for the difference of two unpaired means."""
    rng = random.Random(2)
    ma = [mean(v) for v in ta.values()]
    mb = [mean(v) for v in tb.values()]
    stats = []
    for _ in range(2000):
        sa = [ma[rng.randrange(len(ma))] for _ in ma]
        sb = [mb[rng.randrange(len(mb))] for _ in mb]
        stats.append(mean(sa) - mean(sb))
    stats.sort()
    lo = stats[max(0, int(alpha / 2 * 2000) - 1)]
    hi = stats[min(1999, int((1 - alpha / 2) * 2000) - 1)]
    return Interval(lo, hi, alpha, "bootstrap-diff")


def _additional_paired(b: int, c: int, est: float, n_tasks: int, alpha: float) -> int:
    if b + c > 0 and abs(est) > 0:
        disc = (b + c) / max(1, n_tasks)
        plan = n_paired(disc, abs(est), alpha, 0.8)
        return max(0, plan.n_per_version - n_tasks)
    return 0


def _additional_unpaired(pa: float, pb: float, na: int, nb: int, alpha: float, binary: bool) -> int:
    if not binary or abs(pa - pb) == 0 or pa + pb in (0.0, 2.0):
        return 0
    plan = n_two_proportions(pa, pb, alpha, 0.8)
    return max(0, max(plan.n_per_version - na, plan.n_per_version - nb))


@dataclass
class DecisionReport:
    candidates: dict
    comparisons: list = field(default_factory=list)
    ranking: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["ranking (estimate / Holm-adjusted elimination):"]
        for name, est, keep in self.ranking:
            lines.append(f"  {name:<24} {est:+.4f}  {keep}")
        for c in self.comparisons:
            lines.append(f"  {c}")
        return "\n".join(lines)


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down: adjusted p-values and reject decisions.

    Mandatory when an optimizer compares several candidates against one
    baseline — k uncorrected comparisons at alpha mean up to k*alpha actual
    false-positive rate, which is precisely the failure mode evalsig exists
    to close.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted = {}
    running = 0.0
    m = len(items)
    for i, (name, p) in enumerate(items):
        running = max(running, p * (m - i))
        adj = min(1.0, running)
        adjusted[name] = (adj, adj < alpha)
    return adjusted


def decide(candidates: dict, alpha: float = 0.05) -> DecisionReport:
    """Rank candidates and eliminate the ones that are statistically worse.

    `candidates` maps name -> list of Run. The current best (highest mean
    outcome) becomes the baseline; every other candidate is compared against
    it with Holm correction across the k-1 comparisons. Candidates flagged
    ELIMINATE are safe to drop from the optimizer's population.
    """
    if len(candidates) < 2:
        raise ValueError("decide needs at least two candidates")
    est = {name: mean([r.outcome for r in runs]) for name, runs in candidates.items()}
    order = sorted(est.items(), key=lambda kv: -kv[1])
    best = order[0][0]
    comparisons = []
    pvals = {}
    for name, _ in order[1:]:
        comp = compare(candidates[best], candidates[name], best, name, alpha)
        comparisons.append(comp)
        pvals[name] = comp.p_value if comp.p_value == comp.p_value else 1.0
    adj = holm(pvals, alpha) if pvals else {}
    ranking = [(best, est[best], "KEEP (baseline)")]
    for name, e in order[1:]:
        if name in adj and adj[name][1]:
            ranking.append((name, e, "ELIMINATE (worse)"))
        else:
            ranking.append((name, e, "KEEP (not separable)"))
    return DecisionReport(
        {"estimates": est, "adjusted": {k: v[0] for k, v in adj.items()}},
        comparisons,
        ranking,
    )

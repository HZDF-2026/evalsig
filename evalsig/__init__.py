"""evalsig — error bars and honest decisions for LLM/agent evaluation."""

from .intervals import (
    Interval,
    wilson_interval,
    newcombe_diff_interval,
    two_proportion_test,
    mcnemar_exact,
    cluster_bootstrap_ci,
)
from .variance import variance_components, variance_attribution
from .power import n_two_proportions, n_paired, n_for_ci_width
from .sequential import LookSchedule, SequentialProportion, SequentialAB
from .decide import Run, load_runs, compare, holm, decide

__version__ = "0.1.0"

__all__ = [
    "Interval",
    "wilson_interval",
    "newcombe_diff_interval",
    "two_proportion_test",
    "mcnemar_exact",
    "cluster_bootstrap_ci",
    "variance_components",
    "variance_attribution",
    "n_two_proportions",
    "n_paired",
    "n_for_ci_width",
    "LookSchedule",
    "SequentialProportion",
    "SequentialAB",
    "Run",
    "load_runs",
    "compare",
    "holm",
    "decide",
    "__version__",
]

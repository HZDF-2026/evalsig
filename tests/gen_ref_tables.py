"""Generate the Go unit-test reference tables from the Python reference.

The Go port is validated against this table, which is produced by the Python
implementation on this machine. Sections marked "exact" are bit-stable
everywhere (pure IEEE arithmetic, exact rationals, MT19937, formatting);
sections marked "ulp" pass through libm transcendental functions (erf/log)
where the C runtime itself differs by a few ulp across platforms, so the Go
test compares those with a small relative tolerance.

Regenerate with:  python tests/gen_ref_tables.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist, mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evalsig.intervals import (
    binom_cdf,
    cluster_bootstrap_ci,
    mcnemar_exact,
    newcombe_diff_interval,
    two_proportion_test,
    wilson_interval,
    wilson_interval_roots,
    z_two_sided,
)
from evalsig.power import n_for_ci_width, n_paired, n_two_proportions
from evalsig.variance import variance_components
from evalsig.decide import Run, compare, decide, holm
from evalsig.sequential import SequentialAB, SequentialProportion

OUT = ROOT / "tests" / "refdata" / "core.json"


def num(v):
    """JSON-safe float: NaN/Inf become marker strings the Go test special-cases."""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
    return v


table = {}

# ---------------------------------------------------------------- normal quantiles
table["z_two_sided"] = {"mode": "exact", "cases": [
    {"alpha": a, "want": z_two_sided(a)}
    for a in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.0625, 0.1,
              0.125, 0.15, 0.2, 0.25, 0.3, 0.333, 0.4, 0.5, 0.6, 0.666, 0.7,
              0.75, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 1 - 1e-6, 0.0069)
]}

_ps = [1e-10, 1e-7, 1e-5, 1e-4, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
       0.2, 0.3, 0.4, 0.425, 0.42500001, 0.45, 0.5, 0.55, 0.57499999, 0.575,
       0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999,
       0.999999, 1 - 1e-10, 0.4, 0.6]
nd = NormalDist()
table["norm_inv_cdf"] = {"mode": "exact", "cases": [
    {"p": p, "want": nd.inv_cdf(p)} for p in sorted(set(_ps))
]}
table["norm_cdf"] = {"mode": "ulp", "cases": [
    {"x": x, "want": nd.cdf(x)}
    for x in (-8.5, -6.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.1,
              0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.5, 12.0,
              -12.0, 0.6744897501960817, 1.959963984540054)
]}

# ---------------------------------------------------------------- wilson / newcombe
wilson_cases = []
for n in (1, 2, 3, 5, 10, 25, 50, 100, 200, 1000):
    for s in sorted({0, n, n // 2, n // 3, (2 * n) // 3, max(0, n - 1), 1}):
        if 0 <= s <= n:
            for a in (0.05, 0.01, 0.1, 0.001):
                w = wilson_interval(s, n, a)
                r = wilson_interval_roots(s, n, a)
                wilson_cases.append({"s": s, "n": n, "alpha": a,
                                     "low": w.low, "high": w.high,
                                     "roots_low": r.low, "roots_high": r.high})
table["wilson"] = {"mode": "exact", "cases": wilson_cases}

nc_cases = []
rng = random.Random(7)
for n1, n2 in ((5, 5), (10, 30), (30, 10), (100, 100), (25, 200), (1, 1), (3, 17)):
    for _ in range(6):
        s1, s2 = rng.randrange(n1 + 1), rng.randrange(n2 + 1)
        for a in (0.05, 0.01):
            p1, p2 = s1 / n1, s2 / n2
            nc = newcombe_diff_interval(p1, n1, p2, n2, a)
            nc_cases.append({"p1": p1, "n1": n1, "p2": p2, "n2": n2, "alpha": a,
                             "low": nc.low, "high": nc.high})
table["newcombe"] = {"mode": "exact", "cases": nc_cases}

tp_cases = []
rng = random.Random(11)
for n1, n2 in ((5, 5), (10, 30), (30, 10), (100, 100), (50, 50), (200, 25)):
    for _ in range(5):
        s1, s2 = rng.randrange(n1 + 1), rng.randrange(n2 + 1)
        for a in (0.05, 0.01):
            r = two_proportion_test(s1, n1, s2, n2, a)
            tp_cases.append({"s1": s1, "n1": n1, "s2": s2, "n2": n2, "alpha": a,
                             "diff": r.diff, "low": r.ci.low, "high": r.ci.high,
                             "z": r.z, "p": num(r.p_value)})
# boundary: equal proportions (se == 0), all-success vs all-fail
for (s1, n1, s2, n2) in ((0, 10, 0, 10), (10, 10, 10, 10), (0, 10, 10, 10), (10, 10, 0, 10)):
    r = two_proportion_test(s1, n1, s2, n2, 0.05)
    tp_cases.append({"s1": s1, "n1": n1, "s2": s2, "n2": n2, "alpha": 0.05,
                     "diff": r.diff, "low": r.ci.low, "high": r.ci.high,
                     "z": num(r.z), "p": num(r.p_value)})
table["two_prop"] = {"mode": "exact", "ulp_fields": ["p"], "cases": tp_cases}

# ---------------------------------------------------------------- exact binomial
bc_cases = []
for n in (1, 2, 5, 10, 20, 50, 100, 300):
    for k in sorted({0, n, n // 2, n // 4, (3 * n) // 4, n - 1, 1}):
        if 0 <= k <= n:
            for p in (0.5, 0.3, 0.7, 0.1, 0.9, 0.25, 1 / 3, 0.01, 0.125,
                      0.625, 0.999, 1e-9, 1 - 1e-9, 0.999999999999):
                bc_cases.append({"k": k, "n": n, "p": p, "want": binom_cdf(k, n, p)})
table["binom_cdf"] = {"mode": "exact", "cases": bc_cases}

# ---------------------------------------------------------------- mcnemar
mc_cases = []
rng = random.Random(13)
for _ in range(60):
    b, c = rng.randrange(16), rng.randrange(16)
    for a in (0.05, 0.01):
        m = mcnemar_exact(b, c, a)
        mc_cases.append({"b": b, "c": c, "alpha": a, "diff": m.diff,
                         "low": m.ci.low, "high": m.ci.high, "p": m.p_value})
mc_cases.append({"b": 0, "c": 0, "alpha": 0.05, "diff": 0.0, "low": 0.0, "high": 0.0, "p": 1.0})
table["mcnemar"] = {"mode": "exact", "cases": mc_cases}

# ---------------------------------------------------------------- bootstrap
def gen_clusters(rng, k, size, kind):
    cls = []
    for _ in range(k):
        if kind == "binary":
            cl = [float(rng.random() < 0.5) for _ in range(size)]
        elif kind == "unit":
            cl = [rng.random() for _ in range(size)]
        elif kind == "score":
            cl = [min(1.0, max(0.0, rng.gauss(0.5, 0.2))) for _ in range(size)]
        elif kind == "tiny":
            cl = [rng.random() * 1e-300 for _ in range(size)]
        elif kind == "mixed":
            cl = [rng.choice([rng.random(), rng.random() * 1e300, -rng.random() * 1e-200])
                  for _ in range(size)]
        cls.append(cl)
    return cls

rng = random.Random(123)
bs_cases = []
for kind in ("binary", "unit", "score", "tiny", "mixed"):
    for k, size, B in ((1, 1, 100), (2, 3, 200), (10, 5, 500), (50, 4, 1000), (7, 1, 137)):
        clusters = gen_clusters(rng, k, size, kind)
        for seed in (0, 1, 7, 42):
            ci = cluster_bootstrap_ci(clusters, B=B, alpha=0.05, seed=seed)
            bs_cases.append({"clusters": clusters, "B": B, "alpha": 0.05,
                             "seed": seed, "low": ci.low, "high": ci.high})
table["cluster_bootstrap"] = {"mode": "exact", "cases": bs_cases}

# ---------------------------------------------------------------- exact mean/stdev
rng = random.Random(5)
mean_cases = []
for trial in range(120):
    n = rng.randint(1, 40)
    kind = rng.choice(["unit", "wide", "tiny", "int_like", "neg"])
    if kind == "unit":
        vals = [rng.random() for _ in range(n)]
    elif kind == "wide":
        vals = [rng.choice([rng.random() * 1e300, rng.random() * 1e-300, -rng.random()])
                for _ in range(n)]
    elif kind == "tiny":
        vals = [rng.random() * 1e-310 for _ in range(n)]
    elif kind == "int_like":
        vals = [float(rng.randint(-10**6, 10**6)) for _ in range(n)]
    else:
        vals = [-rng.random() for _ in range(n)]
    m = mean(vals)
    case = {"vals": vals, "mean": m}
    if n >= 2:
        try:
            case["stdev"] = stdev(vals)
        except OverflowError:
            pass  # variance mathematically exceeds float range; not table-able
    mean_cases.append(case)
mean_cases.append({"vals": [1e16, 1.0, -1e16, 1.0, 1.0], "mean": mean([1e16, 1.0, -1e16, 1.0, 1.0]),
                  "stdev": stdev([1e16, 1.0, -1e16, 1.0, 1.0])})
mean_cases.append({"vals": [0.1] * 10, "mean": mean([0.1] * 10), "stdev": stdev([0.1] * 10)})
mean_cases.append({"vals": [1 / 3] * 7, "mean": mean([1 / 3] * 7), "stdev": stdev([1 / 3] * 7)})
mean_cases.append({"vals": [5e-324, 1e-308], "mean": mean([5e-324, 1e-308])})
table["mean_stdev"] = {"mode": "exact", "cases": mean_cases}

# ---------------------------------------------------------------- power
pw_cases = []
rng = random.Random(17)
for _ in range(40):
    p1 = round(rng.uniform(0.05, 0.95), 3)
    p2 = round(rng.uniform(0.05, 0.95), 3)
    if p1 == p2:
        continue
    for a in (0.05, 0.01):
        for power in (0.8, 0.9, 0.7, 0.6, 0.95):
            pw_cases.append({"kind": "two", "p1": p1, "p2": p2, "alpha": a, "power": power,
                             "n": n_two_proportions(p1, p2, a, power).n_per_version})
for disc in (0.3, 0.15, 0.5, 0.8, 1.0):
    for delta in (0.05, 0.1, 0.02, 0.15):
        for a in (0.05, 0.01):
            for power in (0.8, 0.9, 0.95):
                pw_cases.append({"kind": "paired", "disc": disc, "delta": delta,
                                 "alpha": a, "power": power,
                                 "n": n_paired(disc, delta, a, power).n_per_version})
for width in (0.1, 0.05, 0.02, 0.01):
    for p in (0.5, 0.3, 0.8, 0.1, 0.9):
        for a in (0.05, 0.01):
            pw_cases.append({"kind": "width", "width": width, "p": p, "alpha": a,
                             "n": n_for_ci_width(width, p, a).n_per_version})
table["power"] = {"mode": "exact", "cases": pw_cases}

# ---------------------------------------------------------------- variance components
rng = random.Random(23)
vc_cases = []
for trial in range(40):
    a = rng.randint(2, 8)
    groups = {}
    for i in range(a):
        size = rng.randint(1, 6)
        groups[f"g{i}"] = [rng.gauss(0.5 + 0.2 * i, 0.1) for _ in range(size)]
    comps = variance_components(groups)
    vc_cases.append({"groups": [[k, v] for k, v in groups.items()],
                     "between": comps.between, "within": comps.within,
                     "total": comps.total, "icc": comps.icc})
for edge in ({"a": [1.0, 1.0], "b": [1.0, 1.0]},   # degenerate: zero variance everywhere
             {"a": [0.0, 1.0, 0.0, 1.0]}):         # single group: all variance within
    comps = variance_components(edge)
    vc_cases.append({"groups": [[k, v] for k, v in edge.items()],
                     "between": comps.between, "within": comps.within,
                     "total": comps.total, "icc": comps.icc})
table["variance_components"] = {"mode": "exact", "cases": vc_cases}

# ---------------------------------------------------------------- holm
def holm_case(alpha, entries):
    adj = holm(dict(entries), alpha)
    # holm's returned dict follows sorted-p insertion order; canonicalize by p order
    return {"alpha": alpha, "entries": entries,
            "want": [[name, adj[name][0], adj[name][1]] for name, _ in
                     sorted(entries, key=lambda kv: kv[1])]}


holm_cases = [
    holm_case(0.05, [["a", 0.01], ["b", 0.04], ["c", 0.03]]),
    holm_case(0.05, [["a", 0.2], ["b", 0.2], ["c", 0.2]]),
]
rng = random.Random(29)
for _ in range(20):
    k = rng.randint(2, 10)
    entries = [[f"c{i}", round(rng.uniform(0.001, 0.3), 5)] for i in range(k)]
    alpha = rng.choice([0.05, 0.01, 0.1])
    holm_cases.append(holm_case(alpha, entries))
table["holm"] = {"mode": "exact", "cases": holm_cases}

# ---------------------------------------------------------------- MT19937
mt_cases = []
for seed in (0, 1, 2, 42, 5489, 12345, 2**31 - 1, 2**32 + 7):
    r = random.Random(seed)
    mt_cases.append({"seed": seed, "kind": "getrandbits32",
                     "want": [r.getrandbits(32) for _ in range(50)]})
for seed in (0, 7, 99):
    for k in (1, 5, 31, 32, 33, 48, 64):
        r = random.Random(seed)
        mt_cases.append({"seed": seed, "kind": f"getrandbits{k}",
                         "want": [r.getrandbits(k) for _ in range(40)]})
for seed in (0, 1, 42, 12345, -1, -9999):
    for k in (1, 2, 3, 7, 10, 25, 100, 1000):
        r = random.Random(seed)
        mt_cases.append({"seed": seed, "kind": "randrange",
                         "k": k, "want": [r.randrange(k) for _ in range(64)]})
table["mt19937"] = {"mode": "exact", "cases": mt_cases}

# ---------------------------------------------------------------- rounding & formatting
round_cases = []
for x in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.675, 0.49999999999999994, 1.4, 1.6,
          -1.4, -1.6, 1e15 + 0.5, 1e16 + 0.5, -1e15 - 0.5, 0.1, 0.9, 1e-300,
          1.0000000000000002, 0.0, -0.0, 2.5, 3.5, 4.5, 4503599627370495.5,
          1e17, 12345.678, -12345.678, 5e-324,
          # int64 boundary: 2^62 is exactly representable and in range
          4611686018427387904.0, 4611686018427387903.5):
    round_cases.append({"x": x, "want": round(x)})
table["py_round"] = {"mode": "exact", "cases": round_cases}

fmt_vals = [0.05, 1.0, 0.0015785149730094261, 12345.678, 1e-5, 1e-11, 0.0, -0.0,
            1e16, 0.1 + 0.2, 3.0, 1e21, 1.5e-8, 0.1, 1 / 3, 2 / 3, 1e-300, 1e300,
            -1.0, -0.001, 99999.5, 0.000123, 6.02e23, 1.7976931348623157e308,
            5e-324, 100.0, 0.99999, 1e7, 12345678.9]
table["fmt_g4"] = {"mode": "text", "cases": [
    {"x": x, "want": format(x, ".4g")} for x in fmt_vals]}
table["fmt_f4"] = {"mode": "text", "cases": [
    {"x": x, "want": format(x, ".4f")}
    for x in (0.0, -0.0, 1.0, -1.0, 0.12345, -0.00001, 123.456789, -123.456789,
              0.00005, -0.00005, 99999.99999, 1e-5, 0.5, -0.5, 2.5,
              # exact decimal ties at the 4th place (odd multiples of 1/32)
              0.03125, -0.03125, 0.09375, 1.03125, 0.15625, 0.28125, 0.84375)]}
table["fmt_pct"] = {"mode": "text", "cases": [
    {"x": x, "want": format(x, ".0%")}
    for x in (0.05, 0.123, 0.0, 1.0, 0.999, 0.001, -0.5, 0.045, 0.155, 0.049,
              0.051, 0.995, 1.5, -0.04, 0.0001,
              # exact ties at x*100 half-integers (odd multiples of 1/64)
              0.125, 0.375, 0.005, 0.015, 0.625, 0.875, -0.125, 0.245, 0.255)]}
table["py_repr"] = {"mode": "text", "cases": [
    {"x": x, "want": repr(x)} for x in fmt_vals]}

# ---------------------------------------------------------------- json.dumps
dumps_cases = [
    {"value": {"estimate": 0.2, "n": 42, "ok": True, "name": "a", "none": None},
     "want": json.dumps({"estimate": 0.2, "n": 42, "ok": True, "name": "a", "none": None}, indent=2)},
    {"value": {"ci": {"low": 0.07597972498600414, "high": 0.3155697164994789,
                      "width": 0.23958999151347476}},
     "want": json.dumps({"ci": {"low": 0.07597972498600414, "high": 0.3155697164994789,
                                "width": 0.23958999151347476}}, indent=2)},
    {"value": [1, 2.5, "x", [True, False], {"k": [0.1, 0.2]}],
     "want": json.dumps([1, 2.5, "x", [True, False], {"k": [0.1, 0.2]}], indent=2)},
    {"value": {"empty_list": [], "empty_obj": {}, "nested": {"a": {"b": {"c": 1}}}},
     "want": json.dumps({"empty_list": [], "empty_obj": {}, "nested": {"a": {"b": {"c": 1}}}}, indent=2)},
    {"value": {"u": "héllo", "quote": "he said \"hi\"", "back": "a\\b", "nl": "a\nb"},
     "want": json.dumps({"u": "héllo", "quote": "he said \"hi\"", "back": "a\\b", "nl": "a\nb"}, indent=2)},
    {"value": {"z": 0.0, "big": 1e16, "small": 1e-7, "half": 0.5},
     "want": json.dumps({"z": 0.0, "big": 1e16, "small": 1e-7, "half": 0.5}, indent=2)},
]
table["json_dumps"] = {"mode": "text", "cases": dumps_cases}

# ---------------------------------------------------------------- compare (end to end)
def gen_runs(rng, prefix, n_tasks, per_task, binary, shift, start=0):
    runs = []
    for t in range(start, start + n_tasks):
        base = rng.random()
        for _ in range(per_task):
            if binary:
                p = min(0.95, max(0.05, 0.3 + 0.6 * base + shift))
                out = 1.0 if rng.random() < p else 0.0
            else:
                out = min(1.0, max(0.0, base + shift + rng.gauss(0, 0.08)))
            runs.append({"task": f"{prefix}{t:03d}", "success": out,
                         "seed": rng.randrange(3)})
    return runs

cmp_cases = []
scenarios = [
    ("binary-paired", True, 40, 2, 0.12, 0, 0),
    ("binary-paired-weak", True, 40, 1, 0.02, 0, 0),
    ("binary-partial-overlap", True, 60, 2, 0.1, 0, 30),
    ("binary-unpaired", True, 50, 2, 0.12, 0, 1000),
    ("binary-unpaired-weak", True, 30, 2, 0.03, 0, 2000),
    ("score-paired", False, 25, 2, 0.15, 0, 0),
    ("score-paired-weak", False, 25, 3, 0.02, 0, 0),
    ("score-unpaired", False, 30, 2, 0.2, 0, 500),
    ("tiny", True, 2, 1, 0.5, 0, 0),
    ("tiny-identical", True, 4, 1, 0.0, 0, 0),
]
for i, (name, binary, n_tasks, per_task, shift, b_start, b_offset) in enumerate(scenarios):
    rng = random.Random(1000 + i)
    ra = gen_runs(rng, "t", n_tasks, per_task, binary, 0.0)
    if "unpaired" in name:
        rb = gen_runs(rng, "u", n_tasks, per_task, binary, shift, start=b_start)
    elif "partial" in name:
        # tasks 30..89 -> 50% overlap with A's 0..59: paired, but only on the shared half
        rb = gen_runs(rng, "t", n_tasks, per_task, binary, shift, start=30)
    else:
        rb = gen_runs(rng, "t", n_tasks, per_task, binary, shift)
    for alpha in (0.05, 0.01):
        cmp_ = compare([Run.from_dict(d) for d in ra], [Run.from_dict(d) for d in rb],
                       "A", "B", alpha)
        cmp_cases.append({
            "name": f"{name}-a{alpha}", "alpha": alpha,
            "runs_a": ra, "runs_b": rb,
            "want": {"estimate": num(cmp_.estimate), "low": num(cmp_.ci.low),
                     "high": num(cmp_.ci.high), "p": num(cmp_.p_value),
                     "method": cmp_.method, "paired": cmp_.paired,
                     "verdict": cmp_.verdict,
                     "additional_needed": cmp_.additional_needed},
        })
table["compare"] = {"mode": "exact", "ulp_fields": ["p"], "cases": cmp_cases}

# ---------------------------------------------------------------- decide (end to end)
rng = random.Random(4242)
cands = {}
for ci, (cname, shift) in enumerate((("cand-a", 0.0), ("cand-b", -0.12), ("cand-c", 0.03), ("cand-d", -0.02))):
    cands[cname] = [Run.from_dict(d) for d in gen_runs(rng, "t", 30, 1, True, shift, start=ci * 0)]
rep = decide(cands, 0.05)
table["decide"] = {"mode": "text", "cases": [{
    "candidates": {k: [r.flat() for r in v] for k, v in cands.items()},
    "alpha": 0.05,
    "want_ranking": [[n, e, k] for n, e, k in rep.ranking],
    # repr strings: the p pipeline is exact here (mcnemar on binary data), and
    # comparing shortest-repr catches a single-ulp drift as a text diff
    "want_adjusted": [[n, repr(v)] for n, v in rep.candidates["adjusted"].items()],
    "want_verdicts": [c.verdict for c in rep.comparisons],
    "want_methods": [c.method for c in rep.comparisons],
}]}

# ---------------------------------------------------------------- sequential
seq_cases = []
rng = random.Random(77)
for trial, (sweep, drift) in enumerate([(0.45, 0.0), (0.65, 0.002), (0.3, -0.002), (0.5, 0.001)]):
    sp = SequentialProportion(alpha=0.05, target_half_width=0.04)
    steps, verdicts = [], []
    for step in range(1, 13):
        n = 25 * step
        s = int(round(n * min(0.97, max(0.03, sweep + drift * step))))
        steps.append([s, n])
        verdicts.append(sp.update(s, n))
    seq_cases.append({"kind": "prop", "alpha": 0.05, "half": 0.04,
                      "steps": steps, "verdicts": verdicts,
                      "looks_used": sp.looks_used})
for trial, (pa, pb) in enumerate([(0.65, 0.35), (0.5, 0.49), (0.4, 0.6)]):
    ab = SequentialAB(alpha=0.05, target_half_width=0.04)
    steps, verdicts = [], []
    for step in range(1, 13):
        n1 = n2 = 25 * step
        s1 = int(round(n1 * min(0.97, max(0.03, pa))))
        s2 = int(round(n2 * min(0.97, max(0.03, pb))))
        steps.append([s1, n1, s2, n2])
        verdicts.append(ab.update(s1, n1, s2, n2))
    seq_cases.append({"kind": "ab", "alpha": 0.05, "half": 0.04,
                      "steps": steps, "verdicts": verdicts,
                      "looks_used": ab.looks_used})
table["sequential"] = {"mode": "text", "cases": seq_cases}

OUT.parent.mkdir(parents=True, exist_ok=True)
# No sort_keys: the json_dumps cases carry objects whose key order IS the
# payload under test (insertion order must survive into the table verbatim).
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(table, f, indent=1)

n_cases = sum(len(s["cases"]) for s in table.values())
print(f"{OUT.relative_to(ROOT)}: {len(table)} sections, {n_cases} cases, {OUT.stat().st_size:,} bytes")

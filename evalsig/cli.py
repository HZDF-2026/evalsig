"""Command line interface.

Examples:
    evalsig plan --baseline 0.42 --delta 0.03
    evalsig report runs.json
    evalsig check runs.json --factor seed
    evalsig compare a.json b.json
    evalsig decide candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import mean

from .decide import Run, compare, decide, load_runs
from .intervals import wilson_interval
from .power import n_for_ci_width, n_paired, n_two_proportions
from .sequential import SequentialAB, SequentialProportion
from .variance import variance_attribution


def _fmt_ci(ci) -> str:
    return f"[{ci.low:+.4f}, {ci.high:+.4f}] width={ci.width:.4f}"


def cmd_plan(args) -> int:
    print(f"planning to detect delta={args.delta:.3f} at alpha={args.alpha}, power={args.power}")
    if 0 <= args.baseline <= 1:
        un = n_two_proportions(args.baseline, args.baseline + args.delta, args.alpha, args.power)
        print(f"  unpaired (two independent run sets): {un.n_per_version} runs per version")
    if args.discordance:
        paired = n_paired(args.discordance, args.delta, args.alpha, args.power)
        print(f"  paired   (same task set, discordance={args.discordance}): {paired.n_per_version} tasks total")
        print("  -> reusing one task set is typically 3-10x cheaper; always pair when you can")
    p_eff = args.baseline if 0 <= args.baseline <= 1 else 0.5
    w = n_for_ci_width(args.width, p_eff, args.alpha)
    print(f"  to get CI width {args.width:.2f} at p~{p_eff}: {w.n_per_version} runs")
    return 0


def cmd_report(args) -> int:
    runs = load_runs(args.runs)
    if not runs:
        print("no runs")
        return 1
    binary = all(r.outcome in (0.0, 1.0) for r in runs)
    n = len(runs)
    est = mean([r.outcome for r in runs])
    ci = None
    if binary:
        s = round(sum(r.outcome for r in runs))
        ci = wilson_interval(s, n, args.alpha)
        print(f"runs={n}  p_hat={est:.4f}  CI={_fmt_ci(ci)}  (wilson, {1 - args.alpha:.0%})")
    else:
        print(f"runs={n}  mean={est:.4f}  (continuous outcome; use compare for CIs)")
    if args.width_warning and ci is not None and ci.width > args.width_warning:
        print(
            f"  WARNING: CI width {ci.width:.3f} exceeds {args.width_warning:.3f}; "
            "run-to-run noise for agent benchmarks is commonly 2-6pp — your current n "
            "cannot separate changes of that size"
        )
    if args.factor:
        for f in args.factor:
            r = variance_attribution([run.flat() for run in runs], f)
            comp = r["components"]
            print(f"\nvariance attribution by '{f}': {comp.summary()}")
            for g, m in r["group_means"].items():
                print(f"    {g!s:<28} mean={m:+.4f}")
            if comp.icc > 0.3:
                print(f"    -> ICC {comp.icc:.2f}: this factor explains a large share of noise; pin or stratify it")
    return 0


def cmd_check(args) -> int:
    runs = load_runs(args.runs)
    for f in args.factor:
        r = variance_attribution([run.flat() for run in runs], f)
        comp = r["components"]
        print(f"factor '{f}': {comp.summary()}")
        for g, m in r["group_means"].items():
            print(f"    {g!s:<28} mean={m:+.4f}")
    return 0


def cmd_compare(args) -> int:
    a = load_runs(args.a)
    b = load_runs(args.b)
    comp = compare(a, b, args.name_a, args.name_b, args.alpha)
    print(comp)
    if args.json:
        print(json.dumps({
            "estimate": comp.estimate,
            "ci": {"low": comp.ci.low, "high": comp.ci.high, "width": comp.ci.width},
            "p_value": comp.p_value if comp.p_value == comp.p_value else None,
            "method": comp.method,
            "paired": comp.paired,
            "verdict": comp.verdict,
            "additional_needed": comp.additional_needed,
        }, indent=2))
    return 0 if comp.verdict != "INCONCLUSIVE" else 2


def cmd_decide(args) -> int:
    with open(args.candidates, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        cand = {k: [Run.from_dict(d) for d in v] for k, v in data.items()}
    else:
        cand = {}
        for d in data:
            cand.setdefault(str(d.get("version", d.get("name", "?"))), []).append(Run.from_dict(d))
    report = decide(cand, args.alpha)
    print(report)
    if args.json:
        print(json.dumps({
            "ranking": [{"name": n, "estimate": e, "action": a} for n, e, a in report.ranking],
            "adjusted_p": report.candidates.get("adjusted", {}),
        }, indent=2))
    return 0


def cmd_seq(args) -> int:
    """Replay a runs file through the sequential gate (demo of stopping rules)."""
    runs = load_runs(args.runs)
    binary = all(r.outcome in (0.0, 1.0) for r in runs)
    if binary:
        seq = SequentialProportion(alpha=args.alpha, target_half_width=args.half_width)
        s = 0
        for i, r in enumerate(runs, 1):
            s += int(r.outcome)
            v = seq.update(s, i)
            if v != "CONTINUE":
                print(f"stopped at n={i}/{len(runs)}: {v}")
                print(json.dumps(seq.report(), indent=2, default=str))
                return 0
        print(f"budget path: still CONTINUE at n={len(runs)}")
        print(json.dumps(seq.report(), indent=2, default=str))
    else:
        print("sequential gate currently supports binary outcomes; use compare for continuous")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="evalsig",
        description="Error bars and honest decisions for LLM/agent evaluation.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="how many reruns do I need?")
    p.add_argument("--baseline", type=float, default=0.5, help="expected success rate of the baseline")
    p.add_argument("--delta", type=float, default=0.03, help="effect size you must detect")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--discordance", type=float, default=0.3, help="expected task flip rate for paired design")
    p.add_argument("--width", type=float, default=0.05, help="target CI width")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("report", help="estimate + CI + noise audit for one run set")
    r.add_argument("runs", help="runs JSON file")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--factor", action="append", default=[], help="noise factor to attribute (repeatable)")
    r.add_argument("--width-warning", type=float, default=0.05)
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("check", help="variance attribution by factor")
    c.add_argument("runs")
    c.add_argument("--factor", action="append", required=True)
    c.set_defaults(func=cmd_check)

    m = sub.add_parser("compare", help="A/B comparison with CI and verdict")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--name-a", dest="name_a", default="A")
    m.add_argument("--name-b", dest="name_b", default="B")
    m.add_argument("--alpha", type=float, default=0.05)
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_compare)

    d = sub.add_parser("decide", help="rank candidates, eliminate the significantly worse (Holm)")
    d.add_argument("candidates", help="JSON: {name: [runs...]} or runs with a version field")
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_decide)

    s = sub.add_parser("seq", help="replay runs through the sequential stopping gate")
    s.add_argument("runs")
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--half-width", dest="half_width", type=float, default=0.05)
    s.set_defaults(func=cmd_seq)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

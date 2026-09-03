"""Randomized differential test: the full command surface over random data.

Where diff_errors.py walks argparse edge cases on fixed example files, this
script generates seeded random run sets — varying outcome kind (binary/score),
task counts, per-task multiplicities, factor fields and value types, extreme
outcomes, and task-id overlap ratios around the paired/unpaired detection
threshold — then replays report / check / compare / decide / seq / plan through
the Python reference and the Go and C++ ports, requiring byte-identical
stdout and identical exit codes. Runtime errors (exit 1) are compared by code
only: Python prints a traceback, the ports a clean one-line error.

The Python side is deterministic across processes (every set iteration goes
through sorted(); every dict iteration is insertion order), so no PYTHONHASHSEED
pinning is needed.

Usage:  python tests/diff_random.py
Skips (exit 0) for a port whose binary has not been built.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXE = "evalsig.exe" if os.name == "nt" else "evalsig"
GO_BIN = ROOT / "dist" / "go" / EXE
CPP_BIN = ROOT / "dist" / "cpp" / EXE
TMP = ROOT / "tests" / "tmp_random"

OUTCOME_KEYS = ("success", "resolved", "passed", "score", "value", "outcome")

# P-values printed at full precision flow through libm erf (Python: platform
# C runtime; Go: fdlibm port), which diverge by 1-2 ulp on roughly a quarter
# of inputs — an absolute error <= ~1e-15 in the p-value, exactly the
# tolerance model documented in internal/evalsig/refdata_test.go. Every other
# byte of the output is still compared strictly.
P_VALUE_RE = re.compile(r'"p_value": (null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
ADJUSTED_RE = re.compile(r'"adjusted_p": \{.*?\}', re.S)
ADJ_NUMBER_RE = re.compile(r': (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')


def _pval_eq(a, b):
    if a == b:
        return True
    d = abs(a - b)
    return d <= 1e-15 or d / abs(b) < 4e-15


def p_tolerant_match(py_out, go_out):
    """Byte-strict except p_value / adjusted_p literals, compared pairwise."""
    if py_out == go_out:
        return True

    def skeleton(text):
        vals = []

        def repl(m):
            v = m.group(1)
            if v == "null":
                vals.append(None)
            else:
                vals.append(float(v))
            return '"p_value": \x00'

        skel = P_VALUE_RE.sub(repl, text)

        def adj_repl(m):
            def num(nm):
                vals.append(float(nm.group(1)))
                return ": \x00"

            return ADJ_NUMBER_RE.sub(num, m.group(0))

        return ADJUSTED_RE.sub(adj_repl, skel), vals

    skel_py, vals_py = skeleton(py_out)
    skel_go, vals_go = skeleton(go_out)
    if skel_py != skel_go or len(vals_py) != len(vals_go):
        return False
    for a, b in zip(vals_py, vals_go):
        if (a is None) != (b is None):
            return False
        if a is not None and not _pval_eq(a, b):
            return False
    return bool(vals_py)


# ---------------------------------------------------------------- data generation

def _factor_value(rng, kind):
    """A factor value of a given JSON type: str, int, float, or bool."""
    if kind == "str":
        return f"{rng.choice(('seed', 'engine'))}-{rng.randint(0, 3)}"
    if kind == "int":
        return rng.randint(1, 4)
    if kind == "float":
        return round(rng.uniform(0.1, 0.9), 2)
    return rng.random() < 0.5


def gen_runs(rng, *, n_tasks, per_task, binary, key="task", outcome_key=None,
             prefix="t", start=0, factor_specs=(), extreme=None, shift=0.0):
    """A run set. factor_specs: list of (field, kind) tuples."""
    runs = []
    for t in range(start, start + n_tasks):
        base = rng.random()
        k = per_task if isinstance(per_task, int) else rng.randint(*per_task)
        for _ in range(k):
            rec = {key: f"{prefix}{t}"}
            ok = outcome_key or rng.choice(OUTCOME_KEYS)
            if binary:
                if extreme == "all1":
                    v = 1
                elif extreme == "all0":
                    v = 0
                else:
                    p = min(0.95, max(0.05, 0.3 + 0.4 * base + shift))
                    v = 1 if rng.random() < p else 0
                rec[ok] = v
            else:
                if isinstance(extreme, tuple):
                    lo, hi = extreme
                else:
                    lo, hi = 0.0, 1.0
                v = lo + (hi - lo) * (0.25 + 0.5 * base + 0.25 * rng.random() + shift * 0.5)
                rec[ok] = round(v, 6)
            for field, kind in factor_specs:
                rec[field] = _factor_value(rng, kind)
            runs.append(rec)
    return runs


def write(name, data):
    path = TMP / name
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, list):
            json.dump(data, f, indent=1)
        else:
            json.dump(data, f, indent=1)
    return f"tests/tmp_random/{name}"


# ---------------------------------------------------------------- scenarios

def build_cases(rng):
    """Returns (files, commands): commands are argv lists using the file refs."""
    files, cmds = [], []

    def add(path, *argv):
        cmds.append([path] if not argv else [path, *argv])

    # ---- random scenarios
    for i in range(14):
        binary = rng.random() < 0.6
        n_tasks = rng.choice((1, 2, 3, 5, 8, 12))
        per_task = rng.choice((1, 2, (1, 3)))
        specs = []
        if rng.random() < 0.8:
            specs.append(("seed", rng.choice(("int", "str"))))
        if rng.random() < 0.5:
            specs.append(("engine", "str"))
        if rng.random() < 0.25:
            specs.append(("temp", "float"))
        if rng.random() < 0.15:
            specs.append(("flagged", "bool"))
        key = rng.choice(("task", "id"))
        extreme = rng.choice((None, None, None, "all1", "all0", (0.0, 10.0)))
        a = gen_runs(rng, n_tasks=n_tasks, per_task=per_task, binary=binary,
                     key=key, factor_specs=specs, extreme=extreme)
        fa = write(f"s{i}_a.json", a)
        files.append(fa)

        # report surface
        add("report", fa)
        if specs:
            add("report", fa, "--factor", specs[0][0])
        if rng.random() < 0.4:
            add("report", fa, "--alpha", rng.choice(("0.1", "0.01")),
                "--width-warning", rng.choice(("0.03", "0.2")))
        if specs:
            add("check", fa, "--factor", specs[0][0])
        if len(specs) > 1:
            add("check", fa, "--factor", specs[0][0], "--factor", specs[1][0])

        # compare surface: paired (same tasks), then overlap variants
        b = gen_runs(rng, n_tasks=n_tasks, per_task=per_task, binary=binary,
                     key=key, factor_specs=specs, extreme=extreme, shift=0.15)
        fb = write(f"s{i}_b.json", b)
        files.append(fb)
        add("compare", fa, fb)
        add("compare", fa, fb, "--json")
        if rng.random() < 0.5:
            add("compare", fa, fb, "--alpha", "0.01",
                "--name-a", "ours", "--name-b", "theirs")

        # unpaired / partial overlap: b2 starts at a task offset
        overlap_mode = rng.choice(("disjoint", "partial-over", "partial-under"))
        if overlap_mode == "disjoint":
            c = gen_runs(rng, n_tasks=n_tasks, per_task=per_task, binary=binary,
                         key=key, factor_specs=specs, extreme=extreme,
                         prefix="u", start=100, shift=-0.1)
        else:
            # overlap just above/below the 0.5*min(len) pairing threshold
            k = max(1, n_tasks // 2)
            shared = k if overlap_mode == "partial-over" else max(0, k - 1)
            c = gen_runs(rng, n_tasks=n_tasks, per_task=per_task, binary=binary,
                         key=key, factor_specs=specs, extreme=extreme,
                         start=n_tasks - shared, prefix="x", shift=-0.1)
        fc = write(f"s{i}_c.json", c)
        files.append(fc)
        add("compare", fa, fc, "--json")

        # seq: binary only
        if binary:
            add("seq", fa)
            if rng.random() < 0.5:
                add("seq", fa, "--half-width", rng.choice(("0.02", "0.3", "0.5")))

        # decide: dict form and list form
        if i % 3 == 0:
            k_cand = rng.choice((2, 3, 4))
            cand = {}
            for c_i in range(k_cand):
                cand[f"cand{c_i}"] = gen_runs(
                    rng, n_tasks=max(2, n_tasks), per_task=per_task, binary=binary,
                    key="task", factor_specs=(), extreme=None, shift=0.12 * c_i)
            fd = write(f"s{i}_cand.json", cand)
            files.append(fd)
            add("decide", fd)
            add("decide", fd, "--json")
            flat = []
            for c_i, runs in cand.items():
                for r in runs:
                    r2 = dict(r)
                    r2["version"] = c_i
                    flat.append(r2)
            fl = write(f"s{i}_cand_flat.json", flat)
            files.append(fl)
            add("decide", fl)

    # ---- fixed edge scenarios
    edge = [
        ("e_empty.json", []),
        ("e_one.json", gen_runs(rng, n_tasks=1, per_task=1, binary=True)),
        ("e_all1.json", gen_runs(rng, n_tasks=6, per_task=2, binary=True, extreme="all1")),
        ("e_all0.json", gen_runs(rng, n_tasks=6, per_task=2, binary=True, extreme="all0")),
        ("e_big.json", gen_runs(rng, n_tasks=5, per_task=2, binary=False, extreme=(0.0, 100.0))),
        ("e_int_factor.json", gen_runs(rng, n_tasks=6, per_task=(1, 3), binary=True,
                                       factor_specs=(("seed", "int"),))),
        ("e_bool_factor.json", gen_runs(rng, n_tasks=5, per_task=2, binary=True,
                                        factor_specs=(("flagged", "bool"),))),
        ("e_float_factor.json", gen_runs(rng, n_tasks=5, per_task=2, binary=False,
                                         factor_specs=(("temp", "float"),))),
    ]
    for name, data in edge:
        f = write(name, data)
        files.append(f)
        add("report", f)
        if "factor" in name:
            field = "seed" if "int" in name else ("flagged" if "bool" in name else "temp")
            add("report", f, "--factor", field)
            add("check", f, "--factor", field)
        add("seq", f, "--half-width", "0.4")

    # identical A/B (diff exactly 0 everywhere)
    ia = gen_runs(rng, n_tasks=8, per_task=2, binary=True)
    fia = write("e_ident_a.json", ia)
    fib = write("e_ident_b.json", list(ia))
    add("compare", fia, fib, "--json")

    # tiny compare: 1 task vs 1 task, and 1 run vs 2 runs
    fta = write("e_tiny_a.json", gen_runs(rng, n_tasks=1, per_task=1, binary=True))
    ftb = write("e_tiny_b.json", gen_runs(rng, n_tasks=1, per_task=1, binary=True, shift=0.3))
    add("compare", fta, ftb)
    add("compare", fta, ftb, "--json")

    # decide with exact ties (identical run lists under two names)
    tie = gen_runs(rng, n_tasks=6, per_task=2, binary=True)
    ftie = write("e_tie.json", {"alpha": tie, "beta": list(tie)})
    add("decide", ftie)
    add("decide", ftie, "--json")

    # decide with a single candidate (error path)
    fone = write("e_single.json", {"only": gen_runs(rng, n_tasks=3, per_task=1, binary=True)})
    add("decide", fone)

    # fixed unpaired examples shipped with the repo
    add("compare", "examples/unpaired_a.json", "examples/unpaired_b.json")
    add("compare", "examples/unpaired_a.json", "examples/unpaired_b.json", "--json")

    # plan fuzz: valid, boundary, and out-of-range (error path)
    for baseline, delta in ((0.5, 0.03), (0.0, 0.1), (0.9, 0.05), (0.2, 0.4),
                            (1.0, 0.05), (0.5, 0.0), (-0.2, 0.03), (1.5, 0.03)):
        add("plan", "--baseline", str(baseline), "--delta", str(delta))
    add("plan", "--baseline", "0.42", "--delta", "0.03", "--discordance", "0.1")
    add("plan", "--baseline", "0.42", "--delta", "0.03", "--width", "0.9")
    add("plan", "--baseline", "0.42", "--delta", "0.03", "--power", "0.95", "--alpha", "0.01")

    return cmds


def main() -> int:
    if not GO_BIN.exists():
        print(f"skip: {GO_BIN} not built")
        return 0
    ports = [("go", GO_BIN)]
    if CPP_BIN.exists():
        ports.append(("cpp", CPP_BIN))
    else:
        print(f"skip: {CPP_BIN} not built")
    rng = random.Random(20260831)
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    cmds = build_cases(rng)
    failures = 0
    tol_hits = 0
    for idx, case in enumerate(cmds):
        label = " ".join(case)
        py = subprocess.run([sys.executable, "-m", "evalsig"] + case,
                            capture_output=True, text=True, cwd=ROOT)
        py_err = "" if py.returncode == 1 else py.stderr
        ok = True
        for port_name, bin_path in ports:
            r = subprocess.run([str(bin_path)] + case,
                               capture_output=True, text=True, cwd=ROOT)
            r_err = "" if r.returncode == 1 else r.stderr
            if not (py.returncode == r.returncode and py_err == r_err and
                    p_tolerant_match(py.stdout, r.stdout)):
                ok = False
                failures += 1
                print(f"FAIL #{idx} [{py.returncode} vs {r.returncode}] {label} ({port_name})")
                if py.stdout != r.stdout:
                    print("--- py stdout ---")
                    print(repr(py.stdout[:3000]))
                    print(f"--- {port_name} stdout ---")
                    print(repr(r.stdout[:3000]))
                if py_err != r_err:
                    print("--- py stderr ---")
                    print(repr(py_err[:1500]))
                    print(f"--- {port_name} stderr ---")
                    print(repr(r_err[:1500]))
            elif py.stdout != r.stdout:
                tol_hits += 1
        if ok:
            continue
    if failures:
        print(f"\n{failures} of {len(cmds)} randomized cases mismatch")
        return 1
    note = f" ({tol_hits} within p-value libm tolerance)" if tol_hits else ""
    print(f"all {len(cmds)} randomized CLI cases match "
          f"(python vs {' + '.join(n for n, _ in ports)}){note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Differential test for CLI surface: help and usage-error paths.

Compares stdout, stderr, and exit code of the Go and C++ binaries against the
Python reference for a battery of argparse edge cases: bare invocation, bad
subcommand, missing required args (positional and option), unrecognized
arguments, invalid option values, help at both levels.

Usage:  python tests/diff_errors.py
Skips (exit 0) for a port whose binary has not been built.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = "evalsig.exe" if os.name == "nt" else "evalsig"
GO_BIN = ROOT / "dist" / "go" / EXE
CPP_BIN = ROOT / "dist" / "cpp" / EXE

CASES = [
    [],
    ["-h"],
    ["--help"],
    ["help"],
    ["bogus"],
    ["plan"],
    ["plan", "-h"],
    ["plan", "--help"],
    ["plan", "--bogus"],
    ["plan", "--alpha", "abc"],
    ["plan", "--alpha"],
    ["plan", "--baseline", "0.4", "--delta", "0.02"],
    ["plan", "--baseline=0.42", "--delta=0.03", "--width", "0.04"],
    ["plan", "--disc", "0.2"],
    ["plan", "extra"],
    ["report"],
    ["report", "-h"],
    ["report", "--bogus", "x.json"],
    ["report", "examples/swe_ab/a.json"],
    ["report", "examples/swe_ab/a.json", "--factor", "seed"],
    ["report", "examples/swe_ab/a.json", "--factor", "seed", "--factor", "engine"],
    ["report", "x.json", "y.json"],
    ["check"],
    ["check", "-h"],
    ["check", "examples/swe_ab/a.json"],
    ["check", "examples/swe_ab/a.json", "--factor", "seed"],
    ["check", "examples/swe_ab/a.json", "--factor", "f", "extra"],
    ["compare"],
    ["compare", "-h"],
    ["compare", "examples/swe_ab/a.json"],
    ["compare", "examples/swe_ab/a.json", "examples/swe_ab/b.json"],
    ["compare", "examples/swe_ab/a.json", "examples/swe_ab/b.json", "--json"],
    ["compare", "examples/swe_ab/a.json", "examples/swe_ab/b.json", "--alpha", "0.1"],
    ["compare", "a.json", "b.json", "--name-a", "x", "extra"],
    ["compare", "examples/swe_ab/a.json", "examples/swe_ab/b.json", "--json=5"],
    ["decide"],
    ["decide", "-h"],
    ["decide", "examples/candidates.json"],
    ["decide", "examples/candidates.json", "--json"],
    ["seq"],
    ["seq", "-h"],
    ["seq", "examples/swe_ab/a.json"],
    ["seq", "examples/swe_ab/a.json", "--half-width", "0.02"],
]


def main() -> int:
    if not GO_BIN.exists():
        print(f"skip: {GO_BIN} not built")
        return 0
    ports = [("go", GO_BIN)]
    if CPP_BIN.exists():
        ports.append(("cpp", CPP_BIN))
    else:
        print(f"skip: {CPP_BIN} not built")
    failures = 0
    for case in CASES:
        py = subprocess.run([sys.executable, "-m", "evalsig"] + case,
                            capture_output=True, text=True, cwd=ROOT)
        label = " ".join(case) or "(bare)"
        # Python's runtime errors are tracebacks; the ports print a clean
        # one-line error. Same exit code, different text — compare code only.
        py_err = "" if py.returncode == 1 else py.stderr
        port_outputs = []
        for port_name, bin_path in ports:
            r = subprocess.run([str(bin_path)] + case,
                               capture_output=True, text=True, cwd=ROOT)
            r_err = "" if r.returncode == 1 else r.stderr
            port_outputs.append((port_name, r, r_err))
        ok = True
        for port_name, r, r_err in port_outputs:
            if not (py.returncode == r.returncode and py.stdout == r.stdout and py_err == r_err):
                ok = False
                failures += 1
                print(f"FAIL [{py.returncode} vs {r.returncode}] {label} ({port_name})")
                if py.stdout != r.stdout:
                    print("--- py stdout ---")
                    print(repr(py.stdout[:2000]))
                    print(f"--- {port_name} stdout ---")
                    print(repr(r.stdout[:2000]))
                if py_err != r_err:
                    print("--- py stderr ---")
                    print(repr(py_err[:2000]))
                    print(f"--- {port_name} stderr ---")
                    print(repr(r_err[:2000]))
        if ok:
            print(f"ok   [{py.returncode}] {label}")
    if failures:
        print(f"\n{failures} mismatch(es)")
        return 1
    print(f"\nall {len(CASES)} CLI surface cases match "
          f"(python vs {' + '.join(n for n, _ in ports)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

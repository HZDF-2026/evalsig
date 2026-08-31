"""Cross-compile the evalsig Go CLI for the release matrix.

Nine targets covering the platforms that matter — Windows/macOS/Linux across
amd64, arm64, 386, and armv7 — plus the host build that tests/diff_*.py run
against:

    python scripts/build_matrix.py            # host + 9-target matrix
    python scripts/build_matrix.py --matrix   # matrix only
    python scripts/build_matrix.py --host     # host binary only

Artifacts land in dist/release/: one evalsig-<os>-<arch>.(zip|tar.gz) per
target (binary + LICENSE + README.md) and a checksums.txt with SHA-256 for
every archive. The same matrix is reproduced by .github/workflows/release.yml,
which just calls this script on Linux.

CGO is disabled everywhere: the port is pure Go, so every binary is static and
the float paths are the pure-Go fdlibm implementations — identical results per
architecture, independent of the host's C runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "./cmd/evalsig"

# (GOOS, GOARCH, extra env)
TARGETS = [
    ("windows", "amd64", {}),
    ("windows", "arm64", {}),
    ("windows", "386", {}),
    ("darwin", "amd64", {}),
    ("darwin", "arm64", {}),
    ("linux", "amd64", {}),
    ("linux", "arm64", {}),
    ("linux", "386", {}),
    ("linux", "arm", {"GOARM": "7"}),
]


def go_build(out: Path, goos: str, goarch: str, extra: dict) -> None:
    env = dict(os.environ)
    env.update({
        "GOOS": goos,
        "GOARCH": goarch,
        "CGO_ENABLED": "0",
        # amd64 baseline: SSE2 only, no AVX/FMA scheduling divergence; 386
        # forces SSE2 so the 387 stack machine never touches our float paths.
        "GOAMD64": "v1",
        "GO386": "sse2",
    })
    env.update(extra)
    cmd = ["go", "build", "-trimpath", "-ldflags", "-s -w", "-o", str(out), PKG]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"go build {goos}/{goarch} failed:\n{r.stdout}{r.stderr}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_host() -> None:
    out_dir = ROOT / "dist" / "go"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "evalsig.exe" if os.name == "nt" else "evalsig"
    print(f"host   {platform.system().lower()}/{go_host_arch()} -> {out_dir / name}")
    env = dict(os.environ)
    env["CGO_ENABLED"] = "0"
    r = subprocess.run(["go", "build", "-trimpath", "-ldflags", "-s -w",
                        "-o", str(out_dir / name), PKG],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"host go build failed:\n{r.stdout}{r.stderr}")


def go_host_arch() -> str:
    a = platform.machine().lower()
    return {"amd64": "amd64", "x86_64": "amd64", "arm64": "arm64",
            "aarch64": "arm64"}.get(a, a)


def build_matrix() -> list[Path]:
    release = ROOT / "dist" / "release"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)
    archives = []
    for goos, goarch, extra in TARGETS:
        tag = f"evalsig-{goos}-{goarch}"
        ext = ".exe" if goos == "windows" else ""
        workdir = release / tag
        workdir.mkdir()
        print(f"build  {goos}/{goarch}" + (f" ({extra})" if extra else ""))
        go_build(workdir / f"evalsig{ext}", goos, goarch, extra)
        for doc in ("LICENSE", "README.md"):
            shutil.copy2(ROOT / doc, workdir / doc)
        archive = release / (tag + (".zip" if goos == "windows" else ".tar.gz"))
        if goos == "windows":
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(workdir.iterdir()):
                    z.write(p, p.name)
        else:
            with tarfile.open(archive, "w:gz") as t:
                for p in sorted(workdir.iterdir()):
                    t.add(p, arcname=p.name)
        shutil.rmtree(workdir)
        archives.append(archive)

    with open(release / "checksums.txt", "w", encoding="utf-8") as f:
        for a in archives:
            f.write(f"{sha256(a)}  {a.name}\n")
    print(f"\n{len(archives)} archives in {release}")
    return archives


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true", help="release matrix only")
    ap.add_argument("--host", action="store_true", help="host binary only")
    args = ap.parse_args()
    if not args.matrix:
        build_host()
    if not args.host:
        build_matrix()
    return 0


if __name__ == "__main__":
    sys.exit(main())

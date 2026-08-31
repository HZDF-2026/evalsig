"""Generate the seeded example run files used in the README.

Reproduces a realistic agent-benchmark A/B situation: two harness versions
with a small real difference (0.62 vs 0.55), per-task difficulty effects,
and run-to-run noise — the regime where naive single-run comparison gives
the wrong answer most of the time.

Run:  python examples/make_examples.py
"""

import json
import random
from pathlib import Path


def main():
    rng = random.Random(42)
    n_tasks = 150
    reps = 6
    difficulty = {f"GH-{i:04d}": rng.random() for i in range(n_tasks)}
    engines = ["vllm-0.8", "vllm-0.8", "sglang-0.4", "vllm-0.8", "sglang-0.4", "sglang-0.4"]

    def gen(version, p_shift, path):
        runs = []
        total_p = 0.0
        for r in range(reps):
            for task, diff in difficulty.items():
                p = min(0.95, max(0.05, 0.15 + 0.85 * diff + p_shift))
                total_p += p
                success = 1 if rng.random() < p else 0
                runs.append({
                    "task": task,
                    "success": success,
                    "version": version,
                    "seed": r,
                    "engine": engines[r],
                })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=1)
        rate = sum(x["success"] for x in runs) / len(runs)
        print(f"{path}: n={len(runs)} tasks={n_tasks} observed={rate:.3f} (mean true p={total_p / len(runs):.3f})")

    gen("harness-v2", 0.10, "examples/swe_ab/a.json")
    gen("harness-v1", 0.00, "examples/swe_ab/b.json")

    # candidate pool for `evalsig decide`: one good, one clearly worse, two near-ties
    cands = {}
    for name, shift in [
        ("prompt-A", 0.07), ("prompt-B", 0.055), ("prompt-C", 0.05), ("prompt-D", -0.14),
    ]:
        runs = []
        for task, diff in difficulty.items():
            p = min(0.95, max(0.05, 0.15 + 0.85 * diff + shift))
            runs.append({
                "task": task,
                "success": 1 if rng.random() < p else 0,
                "version": name,
            })
        cands[name] = runs
    with open("examples/candidates.json", "w", encoding="utf-8") as f:
        json.dump(cands, f, indent=1)
    print(f"examples/candidates.json: 4 candidates, {n_tasks} tasks each")

    # disjoint task sets: an unpaired A/B (no overlap -> two-proportion z path).
    # Own seed so this block never perturbs the frozen data above.
    ur = random.Random(43)
    for name, path, shift, tasks in [
        ("unpaired-A", "examples/unpaired_a.json", 0.10, 120),
        ("unpaired-B", "examples/unpaired_b.json", -0.08, 130),
    ]:
        runs = []
        for i in range(tasks):
            diff = ur.random()
            p = min(0.95, max(0.05, 0.15 + 0.85 * diff + shift))
            runs.append({
                "task": f"{name[8:].lower()}{i:03d}",
                "success": 1 if ur.random() < p else 0,
                "version": name,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=1)
        print(f"{path}: n={len(runs)} tasks={tasks}")


if __name__ == "__main__":
    main()

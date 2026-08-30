# evalsig

**Error bars and honest decisions for LLM / agent evaluation.**

> Your eval says 71.4% vs 68.9%. Before you tweet that, ask: what is the
> confidence interval? How many of those "wins" are one lucky run?

Run-to-run variance in agent benchmarks is not a rounding error. It is
measured at **2.2–6.0 percentage points of pass@1 spread — with
temperature 0**, across inference engines and environments
([SWE-bench variance study](https://www.arxiv.org/pdf/2602.07150)).
Process-aware scoring shows **0.5–23.2% "lucky passes"** — tasks resolved
by a trajectory that would fail on re-run
([Lucky Pass analysis](https://arxiv.org/html/2605.12925v2)). Meanwhile the
default tooling compares single numbers with no interval, no power analysis,
and no correction for peeking or multiple comparisons.

`evalsig` is the missing statistics layer. It is not another eval harness —
it is what your harness should call before it believes its own numbers.

## What it gives you

| Command | Question it answers |
|---|---|
| `evalsig plan` | How many reruns do I need to detect a 3pp change? (spoiler: more than you ran) |
| `evalsig report` | Estimate + confidence interval + noise audit for one run set |
| `evalsig check` | Which factor (seed / engine / env) is my variance actually coming from? |
| `evalsig compare` | Is A really better than B — paired test, CI, and a verdict |
| `evalsig decide` | Rank N candidates, eliminate only the statistically worse ones (Holm-corrected) |
| `evalsig seq` | Stop the run when the *numbers* justify stopping, not the budget |

Zero dependencies. Python ≥ 3.9, stdlib only. Deterministic given seeds.

## Quickstart

```bash
git clone https://github.com/HZDF-2026/evalsig && cd evalsig
python examples/make_examples.py          # seeded, reproducible demo data

python -m evalsig plan --baseline 0.55 --delta 0.05
```

```
planning to detect delta=0.050 at alpha=0.05, power=0.8
  unpaired (two independent run sets): 1534 runs per version
  paired   (same task set, discordance=0.3): 942 tasks total
  -> reusing one task set is typically 3-10x cheaper; always pair when you can
  to get CI width 0.05 at p~0.55: 1522 runs
```

The A/B you were about to decide from 2 runs each:

```bash
python -m evalsig compare examples/swe_ab/a.json examples/swe_ab/b.json \
    --name-a harness-v2 --name-b harness-v1
```

```
harness-v2 vs harness-v1: +0.1078 CI [+0.0689, +0.1444] (cluster-bootstrap, 95%)
p=3.778e-07 [mcnemar (b=81, c=28)/paired] -> A-BETTER, ~342 more runs to resolve
```

A real difference (true effect 10pp, 150 tasks × 6 reps) resolves cleanly.
Now watch what happens with the prompt variants people usually pick between
on vibes:

```bash
python -m evalsig decide examples/candidates.json
```

```
ranking (estimate / Holm-adjusted elimination):
  prompt-B                 +0.6000  KEEP (baseline)
  prompt-C                 +0.5733  KEEP (not separable)
  prompt-A                 +0.5533  KEEP (not separable)
  prompt-D                 +0.4067  ELIMINATE (worse)
```

Three candidates you cannot distinguish at this sample size — the honest
answer, with the reason attached. Trying to separate B from A anyway would
need ~30k more tasks: "too small to chase, treat as no difference."

Sequential stopping — the same 900-run eval, ending when the statistics say
so:

```bash
python -m evalsig seq examples/swe_ab/a.json --half-width 0.04
```

```
stopped at n=100/900: CONFIRMED
```

89% of the compute never needed to run. (The stopping rule is
Bonferroni-guarded — see [Sequential stopping](#sequential-stopping), this
is not "peek until significant".)

## The statistics (and why each choice is the boring, correct one)

**Wilson intervals, not Wald.** Your pass rate is 47/50. The Wald interval
says [0.874, 1.006] — it overshoots 1, which is impossible, and it is
narrower than the truth exactly where benchmark success rates live. Wilson
gives [0.838, 0.979] and stays valid at n=1 and p=0 or 1. Every proportion
in evalsig reports Wilson.

**McNemar on the fixed task set, not two-sample z.** When you re-run the
same benchmark on the same tasks, the comparison is *paired*. Throwing the
pairing away and z-testing two proportions is how you end up needing 1534
runs when 942 tasks would do — and missing real effects at the sample size
you actually have. evalsig auto-detects task overlap and pairs.

**Cluster bootstrap by task, not by run.** Runs on the same task share
difficulty. A run-level bootstrap treats them as exchangeable and
understates variance — textbook pseudoreplication. evalsig resamples task
clusters, so the interval reflects the number of *independent units* you
actually have.

**Holm correction in the decide() loop.** An optimizer comparing k
candidates against a baseline at uncorrected α=0.05 is running k chances of
a false "improvement". DSPy's own docs warn that metric noise makes
MIPROv2-style optimizers latch onto noise. `decide()` is the drop-in
replacement call: it eliminates only what is *statistically* worse, keeps
everything else honestly labeled "not separable".

**Power analysis you read before running, not after.** `plan` tells you the
n for your intended effect size in both designs before you spend GPU-hours.
`compare` tells you how many *more* runs would resolve an inconclusive
result — or that the effect is too small to ever be worth chasing.

### Sequential stopping

"Run until it looks significant" silently inflates the false-positive rate
— optional stopping is the classic trap. evalsig's gate uses a
**pre-registered doubling look schedule** (n₀, 2n₀, 4n₀, …) with
**Bonferroni α-spending** across looks: each boundary test runs at α/K, so
family-wise error stays ≤ α under the registered schedule. The second exit
is **precision-based**: stop when the CI half-width hits your target —
width stopping does not affect error rates at all. Both are conservative on
purpose; fancier alpha-spending functions are on the roadmap, but
conservative-and-correct ships first.

## API for harness and optimizer authors

```python
from evalsig import Run, compare, decide

# one run per dict: task id + binary outcome (+ any factor metadata)
runs_v2 = [Run.from_dict(r) for r in load_json("a.json")]

comp = compare(runs_v2, runs_v1, "harness-v2", "harness-v1")
if comp.verdict == "A-BETTER" and comp.ci.low > 0.03:
    promote_v2()

report = decide({"prompt-A": ra, "prompt-B": rb, "prompt-D": rd})
for name, _, action in report.ranking:
    if action.startswith("ELIMINATE"):
        population.remove(name)
```

### Run schema

```json
{"task": "GH-0042", "success": 1, "seed": 3, "engine": "sglang-0.4"}
```

- `task` (or `id`): string identifier; overlapping ids trigger paired tests
- outcome: `success` / `resolved` / `passed` (binary) or `score` / `value`
  (continuous) — binary routes to exact tests, continuous to bootstrap
- everything else is factor metadata, usable in `check`/`report --factor`

## Honest limitations

- McNemar with multiple runs per task aggregates to per-task means first;
  the rep-level information enters through the bootstrap CI, not the exact
  test. Documented, deliberate v0.1 scope.
- Sequential boundaries are valid under the registered schedule (doubling,
  ≤ K looks), not fully anytime-valid confidence sequences. Roadmap item,
  not hidden caveat.
- Cluster bootstrap assumes runs within a task are exchangeable. If your
  later runs systematically differ (cache warmup, API drift), that is a
  trend, not noise — evalsig cannot detect it for you; `check --factor seed`
  will usually surface it as a nonzero ICC.
- No regression modeling, no covariates, no hierarchical partial pooling.
  Those are v0.2+ if this layer proves itself.

## Verified against

- Lean 4 + Mathlib machine-checked proofs ([`proofs/`](proofs/README.md)):
  the Wilson interval never needs its [0, 1] clamp (endpoints proved in
  bounds), always covers the sample proportion, and the closed form is
  *exactly* the quadratic-root derivation — the dual-derivation cross-check
  below is a theorem, not a coincidence. The McNemar variance is proved
  nonnegative for all count inputs and the paired difference stays in
  [−1, 1]. 0 errors / 0 warnings / 0 sorry.
- 43 unit tests: hand-computed reference values (Wilson 50/100, McNemar
  22/1024, Holm adjustment, power n=1565/942/1537, random-effects ICC on a
  worked ANOVA example), plus dual-derivation cross-checks (Wilson closed
  form vs quadratic roots).
- Seeded end-to-end examples with known ground-truth effects (see
  `examples/make_examples.py`).

## Roadmap

- [ ] Anytime-valid confidence sequences (Howard et al. stitching)
- [ ] Alpha-spending functions beyond Bonferroni (Pocock, O'Brien-Fleming)
- [ ] Adapters: SWE-bench result logs, lm-eval-harness samples, DSPy metric
      callbacks
- [ ] Hierarchical (partial-pooling) task difficulty model
- [ ] `--changed-only` reuse across evaluation rounds

## License

MIT — see [LICENSE](LICENSE).

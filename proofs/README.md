# Formal proofs

Machine-checked proofs (Lean 4.32.2 + Mathlib v4.32.2) of the core
statistics in `evalsig.intervals`. The unit-test suite can only sample these
properties; here they hold for all inputs.

| File | Theorems |
|---|---|
| `WilsonInterval.lean` | `wilson_lo_nonneg`, `wilson_hi_le_one` — the [0, 1] clamp on Wilson endpoints is mathematically inert (float armor only). `wilson_covers_phat` — the sample proportion is always inside the interval. `wilson_score_eq`, `wilson_score_equation_lo/hi` — the closed-form endpoints are exactly the roots of the score quadratic `n (p̂−p)² = z² p (1−p)`, so the dual-derivation cross-check in the test suite (`wilson_interval` vs `wilson_interval_roots`) is an identity, not an accident. |
| `McNemar.lean` | `mcnemar_variance_nonneg` — the Wald variance of the paired difference is nonnegative for all count inputs (the `max(0, ·)` guard never fires). `mcnemar_diff_le_one` — the paired-difference estimate stays in [−1, 1]. |

Both files import only `Mathlib.Analysis.Real.Sqrt`; any Lean 4.32.2 +
Mathlib v4.32.2 environment checks them.

## Reproduce

In a lake project with Mathlib v4.32.2, copy the `.lean` files in and:

```bash
lake build
```

Or compile them directly against an existing Mathlib checkout:

```powershell
powershell -ExecutionPolicy Bypass -File proofs\verify_proofs.ps1
# override locations if needed:
#   $env:LEAN_BIN = '<lean-4.32.2 bin dir>'
#   $env:MATHLIB  = '<mathlib4 checkout>'
```

Expected result — every file: `errors=0 warnings=0 sorry=0`.

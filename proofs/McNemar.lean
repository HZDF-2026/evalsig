/-
Formal verification of the paired statistics in
`evalsig.intervals.mcnemar_exact`.

Two properties of the point estimate `d = (b − c) / (b + c)` and its
Wald variance `var(d) = (N − (b − c)²/N) / N²` with `N = b + c`:
  * `mcnemar_variance_nonneg` — the variance is nonnegative whenever the
    inputs are counts, so the `max(0, ·)` guard in Python is mathematically
    inert (it remains float armor only). The degenerate case `N = 0` is
    returned early by the Python code and handled here by total division.
  * `mcnemar_diff_le_one` — the difference estimate stays in [−1, 1],
    matching the `Interval` contract of the module.
-/

import Mathlib.Analysis.Real.Sqrt

noncomputable section

variable {b c : ℝ}

/-- The paired-difference variance is nonnegative for count inputs. -/
theorem mcnemar_variance_nonneg (hb : 0 ≤ b) (hc : 0 ≤ c) :
    0 ≤ (b + c - (b - c) ^ 2 / (b + c)) / (b + c) ^ 2 := by
  by_cases hpos : 0 < b + c
  · have h1 : (b - c) ^ 2 ≤ (b + c) ^ 2 :=
      sq_le_sq' (by linarith) (by linarith)
    have h2 : (b - c) ^ 2 / (b + c) ≤ b + c := by
      rw [div_le_iff₀ hpos]
      nlinarith [h1]
    exact div_nonneg (sub_nonneg.2 h2) (by positivity)
  · have hle : b + c ≤ 0 := le_of_not_gt hpos
    have h0 : b + c = 0 := by linarith [add_nonneg hb hc]
    have hb0 : b = 0 := by linarith
    have hc0 : c = 0 := by linarith
    rw [h0, hb0, hc0]
    norm_num

/-- The paired-difference estimate `d = (b − c)/(b + c)` stays in [−1, 1]. -/
theorem mcnemar_diff_le_one (hb : 0 ≤ b) (hc : 0 ≤ c) (hpos : 0 < b + c) :
    |(b - c) / (b + c)| ≤ 1 := by
  have h1 : |b - c| ≤ b + c := by
    rw [abs_le]
    constructor <;> linarith
  rw [abs_div, abs_of_pos hpos]
  exact (div_le_one hpos).mpr h1

end

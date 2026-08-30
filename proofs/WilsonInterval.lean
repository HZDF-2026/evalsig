/-
Formal verification of the Wilson score interval in `evalsig.intervals`.

The Python module carries two independent derivations of the interval:
  * `wilson_interval`       — closed form (center ± half-width)
  * `wilson_interval_roots` — direct quadratic roots of `n (p̂ − p)² = z² p (1−p)`
`wilson_score_eq` proves they agree: the closed-form endpoints are exactly
the roots of the score quadratic, so the cross-check in the test suite is
a theorem, not an accident.

Conventions: `k` successes in `n > 0` trials, critical value `z ≥ 0`.
Denominators cleared:
  rad    = k (n − k) / n + z² / 4        (n² × radicand of the half-width)
  center = (k + z² / 2) / (n + z²)
  half   = z · √rad / (n + z²)
The Python code clamps endpoints into [0, 1] as float armor;
`wilson_lo_nonneg` and `wilson_hi_le_one` show the clamp is mathematically
inert on the statistical domain.
-/

import Mathlib.Analysis.Real.Sqrt

noncomputable section

/-- `p̂(1−p̂)/n + z²/(4n²)` scaled by `n²`: the radicand of the half-width. -/
def wilsonRad (k n z : ℝ) : ℝ := k * (n - k) / n + z ^ 2 / 4

/-- Wilson center `(p̂ + z²/(2n)) / (1 + z²/n)`, denominators cleared. -/
def wilsonCenter (k n z : ℝ) : ℝ := (k + z ^ 2 / 2) / (n + z ^ 2)

/-- Wilson half-width, denominators cleared. -/
def wilsonHalf (k n z : ℝ) : ℝ := z * Real.sqrt (wilsonRad k n z) / (n + z ^ 2)

/-- Wilson lower endpoint `max(0, center − half)` before the float clamp. -/
def wilsonLo (k n z : ℝ) : ℝ := wilsonCenter k n z - wilsonHalf k n z

/-- Wilson upper endpoint `min(1, center + half)` before the float clamp. -/
def wilsonHi (k n z : ℝ) : ℝ := wilsonCenter k n z + wilsonHalf k n z

variable {k n z : ℝ}

/-! ### Radicand and the key bound -/

theorem wilsonRad_nonneg (hn : 0 < n) (hk : 0 ≤ k) (hkn : k ≤ n) :
    0 ≤ wilsonRad k n z := by
  have h1 : 0 ≤ k * (n - k) / n := div_nonneg (mul_nonneg hk (sub_nonneg.2 hkn)) hn.le
  have h2 : 0 ≤ z ^ 2 / 4 := by positivity
  exact add_nonneg h1 h2

/-- Algebraic core, cleared of denominators:
`n · z² · rad ≤ n · (k + z²/2)²`, equivalent to `k² (n + z²) ≥ 0`. -/
private theorem key_mul (hn : 0 < n) (x z : ℝ) :
    z ^ 2 * wilsonRad x n z * n ≤ (x + z ^ 2 / 2) ^ 2 * n := by
  have hn' : n ≠ 0 := ne_of_gt hn
  have hD : 0 < n + z ^ 2 := by linarith [hn, sq_nonneg z]
  have key : 0 ≤ x ^ 2 * (n + z ^ 2) := mul_nonneg (sq_nonneg x) hD.le
  have e1 : z ^ 2 * wilsonRad x n z * n = z ^ 2 * (x * (n - x)) + n * z ^ 4 / 4 := by
    unfold wilsonRad
    field_simp
  have e2 : (x + z ^ 2 / 2) ^ 2 * n = n * x ^ 2 + n * x * z ^ 2 + n * z ^ 4 / 4 := by
    ring
  rw [e1, e2]
  nlinarith [key]

theorem sq_half_le (hn : 0 < n) (x z : ℝ) :
    z ^ 2 * wilsonRad x n z ≤ (x + z ^ 2 / 2) ^ 2 :=
  le_of_mul_le_mul_right (key_mul hn x z) hn

/-- The half-width numerator never exceeds the center numerator. -/
theorem half_num_le (hn : 0 < n) (hz : 0 ≤ z) (hx : 0 ≤ x) :
    z * Real.sqrt (wilsonRad x n z) ≤ x + z ^ 2 / 2 := by
  have hB : 0 ≤ x + z ^ 2 / 2 := by linarith [sq_nonneg z]
  have e : Real.sqrt (z ^ 2 * wilsonRad x n z) = z * Real.sqrt (wilsonRad x n z) := by
    rw [Real.sqrt_mul (sq_nonneg z), Real.sqrt_sq hz]
  rw [← e]
  exact (Real.sqrt_le_iff).mpr ⟨hB, sq_half_le hn x z⟩

/-! ### Boundary validity: the [0, 1] clamp is inert -/

theorem wilson_lo_nonneg (hn : 0 < n) (hz : 0 ≤ z) (hk : 0 ≤ k) :
    0 ≤ wilsonLo k n z := by
  have hD : 0 < n + z ^ 2 := by linarith [hn, sq_nonneg z]
  have hnum : 0 ≤ (k + z ^ 2 / 2) - z * Real.sqrt (wilsonRad k n z) :=
    sub_nonneg.2 (half_num_le hn hz hk)
  have e : wilsonLo k n z = ((k + z ^ 2 / 2) - z * Real.sqrt (wilsonRad k n z)) / (n + z ^ 2) := by
    unfold wilsonLo wilsonCenter wilsonHalf
    rw [← sub_div]
  rw [e]
  exact div_nonneg hnum hD.le

theorem wilson_hi_le_one (hn : 0 < n) (hz : 0 ≤ z) (hkn : k ≤ n) :
    wilsonHi k n z ≤ 1 := by
  have hD : 0 < n + z ^ 2 := by linarith [hn, sq_nonneg z]
  have hrad : wilsonRad (n - k) n z = wilsonRad k n z := by
    unfold wilsonRad
    ring
  have hc := half_num_le hn hz (by linarith : 0 ≤ n - k)
  rw [hrad] at hc
  have ehi : wilsonHi k n z = ((k + z ^ 2 / 2) + z * Real.sqrt (wilsonRad k n z)) / (n + z ^ 2) := by
    unfold wilsonHi wilsonCenter wilsonHalf
    rw [← add_div]
  rw [ehi, div_le_iff₀ hD]
  linarith

theorem wilson_lo_le_hi (hn : 0 < n) (hz : 0 ≤ z) : wilsonLo k n z ≤ wilsonHi k n z := by
  have hD : 0 < n + z ^ 2 := by linarith [hn, sq_nonneg z]
  have h2 : 0 ≤ wilsonHalf k n z := by
    unfold wilsonHalf
    exact div_nonneg (mul_nonneg hz (Real.sqrt_nonneg _)) hD.le
  have e : wilsonHi k n z - wilsonLo k n z = 2 * wilsonHalf k n z := by
    unfold wilsonHi wilsonLo
    ring
  linarith

/-! ### The interval covers the point estimate -/

theorem wilson_covers_phat (hn : 0 < n) (hk : 0 ≤ k) (hkn : k ≤ n) (hz : 0 ≤ z) :
    wilsonLo k n z ≤ k / n ∧ k / n ≤ wilsonHi k n z := by
  have hD : 0 < n + z ^ 2 := by linarith [hn, sq_nonneg z]
  have hA : 0 ≤ wilsonRad k n z := wilsonRad_nonneg hn hk hkn
  have hz2 : 0 ≤ k * z ^ 2 := mul_nonneg hk (sq_nonneg z)
  have hsqrt : z / 2 ≤ Real.sqrt (wilsonRad k n z) := by
    refine (Real.le_sqrt (by linarith) hA).mpr ?_
    have h1 : 0 ≤ k * (n - k) / n := div_nonneg (mul_nonneg hk (sub_nonneg.2 hkn)) hn.le
    have e : (z / 2) ^ 2 = z ^ 2 / 4 := by ring
    rw [e]
    unfold wilsonRad
    linarith
  have htail : n * z ^ 2 / 2 ≤ n * z * Real.sqrt (wilsonRad k n z) := by
    have h1 : z * (z / 2) ≤ z * Real.sqrt (wilsonRad k n z) :=
      mul_le_mul_of_nonneg_left hsqrt hz
    have h2 : n * (z * (z / 2)) ≤ n * (z * Real.sqrt (wilsonRad k n z)) :=
      mul_le_mul_of_nonneg_left h1 hn.le
    have e : n * z ^ 2 / 2 = n * (z * (z / 2)) := by ring
    linarith
  have hbound : k * z ^ 2 ≤ n * z ^ 2 := mul_le_mul_of_nonneg_right hkn (sq_nonneg z)
  have elo : wilsonLo k n z = ((k + z ^ 2 / 2) - z * Real.sqrt (wilsonRad k n z)) / (n + z ^ 2) := by
    unfold wilsonLo wilsonCenter wilsonHalf
    rw [← sub_div]
  have ehi : wilsonHi k n z = ((k + z ^ 2 / 2) + z * Real.sqrt (wilsonRad k n z)) / (n + z ^ 2) := by
    unfold wilsonHi wilsonCenter wilsonHalf
    rw [← add_div]
  constructor
  · rw [elo, div_le_div_iff₀ hD hn]
    linarith
  · rw [ehi, div_le_div_iff₀ hn hD]
    linarith

/-! ### Equivalence with the quadratic-root derivation -/

/-- Vieta sum: `lo + hi = (2k + z²) / (n + z²)`. -/
theorem wilson_sum (k n z : ℝ) :
    wilsonLo k n z + wilsonHi k n z = (2 * k + z ^ 2) / (n + z ^ 2) := by
  unfold wilsonLo wilsonHi wilsonCenter wilsonHalf
  ring

/-- Vieta product: `lo · hi = k² / (n (n + z²))`. -/
theorem wilson_prod (hn : 0 < n) (hk : 0 ≤ k) (hkn : k ≤ n) :
    wilsonLo k n z * wilsonHi k n z = k ^ 2 / (n * (n + z ^ 2)) := by
  have hn' : n ≠ 0 := ne_of_gt hn
  have hD' : n + z ^ 2 ≠ 0 := ne_of_gt (by linarith [hn, sq_nonneg z])
  have hA : 0 ≤ wilsonRad k n z := wilsonRad_nonneg hn hk hkn
  have e2 : (z * Real.sqrt (wilsonRad k n z)) ^ 2 = z ^ 2 * wilsonRad k n z := by
    rw [mul_pow, Real.sq_sqrt hA]
  have e1 : wilsonLo k n z * wilsonHi k n z
      = ((k + z ^ 2 / 2) ^ 2 - (z * Real.sqrt (wilsonRad k n z)) ^ 2) / (n + z ^ 2) ^ 2 := by
    unfold wilsonLo wilsonHi wilsonCenter wilsonHalf
    rw [← sub_div, ← add_div]
    field_simp
    ring
  have e3 : (k + z ^ 2 / 2) ^ 2 - z ^ 2 * wilsonRad k n z = k ^ 2 * (n + z ^ 2) / n := by
    unfold wilsonRad
    field_simp
    ring
  rw [e1, e2, e3]
  field_simp

/-- The score quadratic factors through the closed-form endpoints:
`n (p̂ − p)² − z² p (1 − p) = (n + z²) (p − lo) (p − hi)` identically in `p`. -/
theorem wilson_score_eq (hn : 0 < n) (hk : 0 ≤ k) (hkn : k ≤ n) (x : ℝ) :
    n * (k / n - x) ^ 2 - z ^ 2 * x * (1 - x)
      = (n + z ^ 2) * (x - wilsonLo k n z) * (x - wilsonHi k n z) := by
  have hn' : n ≠ 0 := ne_of_gt hn
  have hD' : n + z ^ 2 ≠ 0 := ne_of_gt (by linarith [hn, sq_nonneg z])
  have hsum : wilsonLo k n z + wilsonHi k n z = (2 * k + z ^ 2) / (n + z ^ 2) :=
    wilson_sum k n z
  have hprod : wilsonLo k n z * wilsonHi k n z = k ^ 2 / (n * (n + z ^ 2)) :=
    wilson_prod hn hk hkn
  have hquad : (n + z ^ 2) * (x - wilsonLo k n z) * (x - wilsonHi k n z)
      = (n + z ^ 2) * x ^ 2 - (2 * k + z ^ 2) * x + k ^ 2 / n := by
    have h1 : (n + z ^ 2) * (x - wilsonLo k n z) * (x - wilsonHi k n z)
        = (n + z ^ 2) * (x * x - x * (wilsonLo k n z + wilsonHi k n z)
            + wilsonLo k n z * wilsonHi k n z) := by
      ring
    rw [h1, hsum, hprod]
    field_simp
  calc n * (k / n - x) ^ 2 - z ^ 2 * x * (1 - x)
      = (n + z ^ 2) * x ^ 2 - (2 * k + z ^ 2) * x + k ^ 2 / n := by
        field_simp
        ring
    _ = (n + z ^ 2) * (x - wilsonLo k n z) * (x - wilsonHi k n z) := hquad.symm

/-- The closed-form lower endpoint solves the score equation
`n (p̂ − p)² = z² p (1 − p)` — the defining property of
`wilson_interval_roots`' lower root. -/
theorem wilson_score_equation_lo (hn : 0 < n) (hk : 0 ≤ k) (hkn : k ≤ n) :
    n * (k / n - wilsonLo k n z) ^ 2
      = z ^ 2 * wilsonLo k n z * (1 - wilsonLo k n z) := by
  have h : n * (k / n - wilsonLo k n z) ^ 2 - z ^ 2 * wilsonLo k n z * (1 - wilsonLo k n z)
      = (n + z ^ 2) * (wilsonLo k n z - wilsonLo k n z) * (wilsonLo k n z - wilsonHi k n z) :=
    wilson_score_eq hn hk hkn (wilsonLo k n z)
  simp only [sub_self, mul_zero] at h
  linarith

/-- The closed-form upper endpoint solves the score equation. -/
theorem wilson_score_equation_hi (hn : 0 < n) (hk : 0 ≤ k) (hkn : k ≤ n) :
    n * (k / n - wilsonHi k n z) ^ 2
      = z ^ 2 * wilsonHi k n z * (1 - wilsonHi k n z) := by
  have h : n * (k / n - wilsonHi k n z) ^ 2 - z ^ 2 * wilsonHi k n z * (1 - wilsonHi k n z)
      = (n + z ^ 2) * (wilsonHi k n z - wilsonLo k n z) * (wilsonHi k n z - wilsonHi k n z) :=
    wilson_score_eq hn hk hkn (wilsonHi k n z)
  simp only [sub_self, mul_zero] at h
  linarith

end

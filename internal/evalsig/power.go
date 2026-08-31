// Power analysis: how many reruns before a decision means anything.
package evalsig

import (
	"fmt"
	"math"
)

// PowerPlan is a sample-size plan with the design it was computed for.
type PowerPlan struct {
	NPerVersion int
	Alpha       float64
	Power       float64
	Method      string
	Note        string
}

// NTwoProportions is the minimum n per version to detect p1 vs p2
// (unpaired z-test). Normal approximation with pooled-variance form:
// n = (z_{a/2} sqrt(2 p_bar q_bar) + z_b sqrt(p1 q1 + p2 q2))^2 / delta^2.
func NTwoProportions(p1, p2, alpha, power float64) PowerPlan {
	validatePower(p1, p2, alpha, power)
	delta := math.Abs(p1 - p2)
	if delta == 0 {
		panic("p1 == p2: no effect to detect, n is undefined")
	}
	zA := zTwoSided(alpha)
	zB := normInvCDF(power)
	pbar := (p1 + p2) / 2
	num := (zA*math.Sqrt(2*pbar*(1-pbar)) + zB*math.Sqrt(p1*(1-p1)+p2*(1-p2))) 
	num = num * num
	n := int(math.Ceil(num / (delta * delta)))
	if n < 1 {
		n = 1
	}
	return PowerPlan{n, alpha, power, "two-proportion z", ""}
}

// NPaired is the minimum total N (paired, fixed task set) for McNemar.
//
// delta = |p1 - p2| is the win-rate difference; expectedDiscordance is the
// fraction of tasks expected to flip between versions (b + c)/N — from a
// pilot run, or 0.3 as a common first guess for agent benchmarks.
// N = (z_{a/2} + z_b)^2 * psi / delta^2 with psi = discordance, derived
// from the paired variance of d = (b - c)/N.
func NPaired(expectedDiscordance, delta, alpha, power float64) PowerPlan {
	validatePower(0.5, 0.5, alpha, power)
	if !(expectedDiscordance > 0 && expectedDiscordance <= 1) {
		panic(fmt.Sprintf("discordance must be in (0, 1], got %v", pyReprFloat(expectedDiscordance)))
	}
	if delta <= 0 {
		panic("delta must be > 0")
	}
	zA := zTwoSided(alpha)
	zB := normInvCDF(power)
	n := int(math.Ceil((zA + zB) * (zA + zB) * expectedDiscordance / (delta * delta)))
	if n < 1 {
		n = 1
	}
	return PowerPlan{
		NPerVersion: n,
		Alpha:       alpha,
		Power:       power,
		Method:      "mcnemar",
		Note:        "paired on the same task set; multiply-unpaired designs need far more",
	}
}

// NForCIWidth is the minimum n for a CI half-width <= width/2 at success
// rate p. Wald-based planning formula n = z^2 p(1-p) / (width/2)^2;
// conservative because it ignores Wilson's shrinkage, which is the right
// direction for planning.
func NForCIWidth(width, p, alpha float64) PowerPlan {
	if !(width > 0 && width < 1) {
		panic(fmt.Sprintf("width must be in (0, 1), got %v", pyReprFloat(width)))
	}
	if !(p >= 0 && p <= 1) {
		panic(fmt.Sprintf("p must be in [0, 1], got %v", pyReprFloat(p)))
	}
	z := zTwoSided(alpha)
	half := width / 2
	n := int(math.Ceil(z * z * p * (1 - p) / (half * half)))
	if n < 1 {
		n = 1
	}
	return PowerPlan{n, alpha, 0.0, "ci-width", ""}
}

func validatePower(p1, p2, alpha, power float64) {
	for _, p := range []float64{p1, p2} {
		if !(p >= 0 && p <= 1) {
			panic(fmt.Sprintf("proportion out of [0,1]: %v", pyReprFloat(p)))
		}
	}
	if !(alpha > 0 && alpha < 1) {
		panic(fmt.Sprintf("alpha must be in (0,1), got %v", pyReprFloat(alpha)))
	}
	if !(power > 0 && power < 1) {
		panic(fmt.Sprintf("power must be in (0,1), got %v", pyReprFloat(power)))
	}
}

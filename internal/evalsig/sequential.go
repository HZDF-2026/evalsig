// Sequential stopping: end the eval when the number, not the budget, says so.
//
// The statistical hazard of "run until it looks significant" is real and
// mostly invisible: peeking at a CI after every batch and stopping on the
// first hit inflates the false-positive rate far above the nominal alpha.
// The honest fixes are (a) a pre-registered look schedule with alpha
// correction, and (b) stopping on precision (CI width), which does not
// inflate type-I error at all. This module implements both, conservatively.
package evalsig

import (
	"fmt"
	"math"
)

// LookSchedule is a pre-registered doubling look schedule with Bonferroni
// alpha spending. Checks happen only at n = n0, 2*n0, 4*n0, ... up to
// maxLooks. Each look tests at alpha/K (Bonferroni over all K looks), which
// keeps the family type-I error <= alpha under the registered schedule.
type LookSchedule struct {
	N0       int
	MaxLooks int
}

func NewLookSchedule() LookSchedule { return LookSchedule{N0: 25, MaxLooks: 10} }

func (s LookSchedule) Ns() []int {
	out := make([]int, s.MaxLooks)
	for k := 0; k < s.MaxLooks; k++ {
		out[k] = s.N0 * (1 << k)
	}
	return out
}

func (s LookSchedule) NMax() int {
	return s.N0 * (1 << (s.MaxLooks - 1))
}

// LookFor returns the index of the largest scheduled look <= n, or -1
// before the first.
func (s LookSchedule) LookFor(n int) int {
	last := -1
	for i, t := range s.Ns() {
		if t <= n {
			last = i
		} else {
			break
		}
	}
	return last
}

// SequentialProportion tracks one proportion; stop on significance vs a
// reference or on width. Width-based stopping ("PRECISION-REACHED") is the
// estimator-friendly exit: it ends the run once the CI is tight enough to
// be useful, with no effect on error rates. Significance checking only
// fires at scheduled looks.
type SequentialProportion struct {
	Reference       float64
	Alpha           float64
	TargetHalfWidth float64
	Schedule        LookSchedule
	Successes       int
	N               int
	LooksUsed       int
	Verdict         string
}

func NewSequentialProportion(alpha, targetHalfWidth float64) *SequentialProportion {
	return &SequentialProportion{
		Reference:       0.5,
		Alpha:           alpha,
		TargetHalfWidth: targetHalfWidth,
		Schedule:        NewLookSchedule(),
		Verdict:         "CONTINUE",
	}
}

func (s *SequentialProportion) Update(successes, n int) string {
	if n < s.N || successes < 0 || successes > n {
		panic("non-monotone update")
	}
	if n < 1 {
		panic(fmt.Sprintf("n must be >= 1, got %d", n))
	}
	s.Successes, s.N = successes, n
	if s.Verdict != "CONTINUE" {
		return s.Verdict
	}
	k := s.Schedule.MaxLooks
	alphaLook := s.Alpha / float64(k)
	pHat := float64(successes) / float64(n)
	// width exit: plain-alpha Wilson width, any n (no error-rate cost)
	if WilsonInterval(successes, n, s.Alpha).Half() <= s.TargetHalfWidth {
		s.Verdict = "PRECISION-REACHED"
		return s.Verdict
	}
	look := s.Schedule.LookFor(n)
	if look >= 0 && look > s.LooksUsed-1 {
		s.LooksUsed = look + 1
		// z-test vs reference at Bonferroni-corrected alpha, scheduled looks only
		if float64(n)*pHat*(1-pHat) > 0 {
			se := math.Sqrt(pHat * (1 - pHat) / float64(n))
			z := (pHat - s.Reference) / se
			pVal := pyMul(2, 1-normCDF(math.Abs(z)))
			if pVal < alphaLook {
				s.Verdict = "CONFIRMED"
				return s.Verdict
			}
		}
	}
	if n >= s.Schedule.NMax() {
		s.Verdict = "BUDGET-EXHAUSTED"
	}
	return s.Verdict
}

// CI returns the plain-alpha Wilson CI at the current n (reporting, not
// testing).
func (s *SequentialProportion) CI() Interval {
	return WilsonInterval(s.Successes, s.N, s.Alpha)
}

// Report is the report() dict, in Python's key order.
func (s *SequentialProportion) Report() *pyObj {
	p := math.NaN()
	if s.N != 0 {
		p = float64(s.Successes) / float64(s.N)
	}
	return newPyObj().
		set("p_hat", p).
		set("n", s.N).
		set("verdict", s.Verdict).
		set("ci", s.CI().String()).
		set("looks_used", s.LooksUsed).
		set("test_alpha_per_look", s.Alpha/float64(s.Schedule.MaxLooks))
}

// SequentialAB is sequential A/B on two run streams of binary outcomes.
// Same discipline as SequentialProportion: scheduled looks with Bonferroni
// alpha for the significance exit, any-n width exit for precision. Use the
// paired McNemar machinery in Decide for final analysis; this is the online
// gate that tells the runner when to stop spending.
type SequentialAB struct {
	Alpha           float64
	TargetHalfWidth float64
	Schedule        LookSchedule
	S1, N1          int
	S2, N2          int
	LooksUsed       int
	Verdict         string
}

func NewSequentialAB(alpha, targetHalfWidth float64) *SequentialAB {
	return &SequentialAB{
		Alpha:           alpha,
		TargetHalfWidth: targetHalfWidth,
		Schedule:        NewLookSchedule(),
		Verdict:         "CONTINUE",
	}
}

func (s *SequentialAB) Update(s1, n1, s2, n2 int) string {
	for _, sc := range []struct{ s, n int }{{s1, n1}, {s2, n2}} {
		if sc.s < 0 || sc.s > sc.n {
			panic("successes out of range")
		}
	}
	s.S1, s.N1, s.S2, s.N2 = s1, n1, s2, n2
	if s.Verdict != "CONTINUE" {
		return s.Verdict
	}
	k := s.Schedule.MaxLooks
	alphaLook := s.Alpha / float64(k)
	if n1 == n2 && n1 > 0 {
		p1 := float64(s1) / float64(n1)
		p2 := float64(s2) / float64(n2)
		half := zTwoSided(s.Alpha) * math.Sqrt(p1*(1-p1)/float64(n1)+p2*(1-p2)/float64(n2))
		if half <= s.TargetHalfWidth {
			s.Verdict = "PRECISION-REACHED"
			return s.Verdict
		}
	}
	mn := n1
	if n2 < mn {
		mn = n2
	}
	look := s.Schedule.LookFor(mn)
	if look >= 0 && look > s.LooksUsed-1 && mn > 0 {
		s.LooksUsed = look + 1
		p1 := float64(s1) / float64(n1)
		p2 := float64(s2) / float64(n2)
		pooled := float64(s1+s2) / float64(n1+n2)
		se := math.Sqrt(pooled * (1 - pooled) * (1/float64(n1) + 1/float64(n2)))
		if se > 0 {
			z := (p1 - p2) / se
			pVal := pyMul(2, 1-normCDF(math.Abs(z)))
			if pVal < alphaLook {
				s.Verdict = "CONFIRMED"
				return s.Verdict
			}
		}
	}
	if mn >= s.Schedule.NMax() {
		s.Verdict = "BUDGET-EXHAUSTED"
	}
	return s.Verdict
}

// Report is the report() dict, in Python's key order.
func (s *SequentialAB) Report() *pyObj {
	p1 := math.NaN()
	if s.N1 != 0 {
		p1 = float64(s.S1) / float64(s.N1)
	}
	p2 := math.NaN()
	if s.N2 != 0 {
		p2 = float64(s.S2) / float64(s.N2)
	}
	return newPyObj().
		set("p1", p1).
		set("p2", p2).
		set("diff", p1-p2).
		set("n1", s.N1).
		set("n2", s.N2).
		set("verdict", s.Verdict).
		set("looks_used", s.LooksUsed).
		set("test_alpha_per_look", s.Alpha/float64(s.Schedule.MaxLooks))
}

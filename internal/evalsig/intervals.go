// Confidence intervals and exact tests for benchmark proportions.
//
// All estimators are chosen for validity at small n and near-boundary
// success rates — the regime where agent benchmarks actually live. Wald
// intervals (the default in most eval scripts) are wrong exactly there.
package evalsig

import (
	"fmt"
	"math"
	"math/big"
	"sort"
)

// Interval is a confidence interval with the alpha it was built at.
type Interval struct {
	Low   float64
	High  float64
	Alpha float64
	Method string
}

func (i Interval) Width() float64  { return i.High - i.Low }
func (i Interval) Half() float64   { return i.Width() / 2 }
func (i Interval) Center() float64 { return (i.Low + i.High) / 2 }

func (i Interval) Excludes(value float64) bool {
	return value < i.Low || value > i.High
}

func (i Interval) String() string {
	return fmt.Sprintf("[%+.4f, %+.4f] (%s, %s)",
		i.Low, i.High, i.Method, pyPercent0(1-i.Alpha))
}

// zTwoSided is z_{alpha/2}, the two-sided normal quantile.
func zTwoSided(alpha float64) float64 {
	if !(alpha > 0 && alpha < 1) {
		panic(fmt.Sprintf("alpha must be in (0, 1), got %v", pyReprFloat(alpha)))
	}
	return normInvCDF(1 - alpha/2)
}

// WilsonInterval is the Wilson score interval for a binomial proportion.
//
// Valid at n as small as 1 and for p near 0 or 1, where the Wald interval
// collapses (e.g. 0/50 -> Wald [-0.0, 0.0], Wilson [0.0, 0.072]).
func WilsonInterval(successes, n int, alpha float64) Interval {
	if n < 1 {
		panic(fmt.Sprintf("n must be >= 1, got %d", n))
	}
	if successes < 0 || successes > n {
		panic(fmt.Sprintf("successes must be in [0, n], got %d/%d", successes, n))
	}
	z := zTwoSided(alpha)
	z2 := z * z
	p := float64(successes) / float64(n)
	denom := 1 + z2/float64(n)
	center := (p + z2/(2*float64(n))) / denom
	half := z * math.Sqrt(p*(1-p)/float64(n)+z2/(4*float64(n)*float64(n))) / denom
	return Interval{
		Low:   math.Max(0.0, center-half),
		High:  math.Min(1.0, center+half),
		Alpha: alpha,
		Method: "wilson",
	}
}

// WilsonIntervalRoots computes the Wilson interval via direct quadratic
// roots — an independent derivation used as a cross-check in the tests:
// (p_hat - p)^2 * n = z^2 * p * (1 - p).
func WilsonIntervalRoots(successes, n int, alpha float64) Interval {
	if n < 1 {
		panic(fmt.Sprintf("n must be >= 1, got %d", n))
	}
	if successes < 0 || successes > n {
		panic(fmt.Sprintf("successes must be in [0, n], got %d/%d", successes, n))
	}
	z := zTwoSided(alpha)
	z2 := z * z
	p := float64(successes) / float64(n)
	a := float64(n) + z2
	b := -(2*float64(n)*p + z2)
	c := float64(n) * p * p
	disc := b*b - 4*a*c
	if disc < 0 { // cannot happen; guard for float safety
		disc = 0.0
	}
	r := math.Sqrt(disc)
	return Interval{
		Low:   math.Max(0.0, (-b-r)/(2*a)),
		High:  math.Min(1.0, (-b+r)/(2*a)),
		Alpha: alpha,
		Method: "wilson",
	}
}

// NewcombeDiffInterval is the Newcombe hybrid-score CI for p1 - p2
// (unpaired). Composes the two Wilson intervals; keeps near-boundary
// validity where the Wald CI for a difference overstates certainty.
// Newcombe, Stat. Med. 17 (1998), method 10.
func NewcombeDiffInterval(p1 float64, n1 int, p2 float64, n2 int, alpha float64) Interval {
	for _, pc := range []struct {
		p float64
		n int
	}{{p1, n1}, {p2, n2}} {
		if !(pc.p >= 0 && pc.p <= 1) {
			panic(fmt.Sprintf("proportion out of [0,1]: %v", pyReprFloat(pc.p)))
		}
		if pc.n < 1 {
			panic(fmt.Sprintf("n must be >= 1, got %d", pc.n))
		}
	}
	s1 := int(pyRound(p1 * float64(n1)))
	s2 := int(pyRound(p2 * float64(n2)))
	w1 := WilsonInterval(s1, n1, alpha)
	w2 := WilsonInterval(s2, n2, alpha)
	d := p1 - p2
	lower := d - math.Sqrt((p1-w1.Low)*(p1-w1.Low)+(w2.High-p2)*(w2.High-p2))
	upper := d + math.Sqrt((w1.High-p1)*(w1.High-p1)+(p2-w2.Low)*(p2-w2.Low))
	return Interval{
		Low:   math.Max(-1.0, lower),
		High:  math.Min(1.0, upper),
		Alpha: alpha,
		Method: "newcombe",
	}
}

// TwoPropResult is the outcome of the pooled two-proportion z-test.
type TwoPropResult struct {
	Diff   float64
	CI     Interval
	Z      float64
	PValue float64
}

func (r TwoPropResult) Significant() bool { return r.PValue < r.CI.Alpha }

// TwoProportionTest runs the pooled z-test for two proportions plus the
// Newcombe CI. The z-test decides significance (p-value), the Newcombe
// interval reports the effect with honest width. Reporting both is
// intentional: with small n they can disagree, and that disagreement is
// signal, not bug.
func TwoProportionTest(s1, n1, s2, n2 int, alpha float64) TwoPropResult {
	for _, sc := range []struct{ s, n int }{{s1, n1}, {s2, n2}} {
		if sc.s < 0 || sc.s > sc.n {
			panic(fmt.Sprintf("successes must be in [0, n], got %d/%d", sc.s, sc.n))
		}
		if sc.n < 1 {
			panic(fmt.Sprintf("n must be >= 1, got %d", sc.n))
		}
	}
	p1 := float64(s1) / float64(n1)
	p2 := float64(s2) / float64(n2)
	pooled := float64(s1+s2) / float64(n1+n2)
	se := math.Sqrt(pooled * (1 - pooled) * (1/float64(n1) + 1/float64(n2)))
	var z float64
	if se == 0 {
		if p1 == p2 {
			z = 0.0
		} else if p1-p2 < 0 {
			z = math.Inf(-1)
		} else {
			z = math.Inf(1)
		}
	} else {
		z = (p1 - p2) / se
	}
	pValue := 2 * (1 - normCDF(math.Abs(z)))
	return TwoPropResult{
		Diff:   p1 - p2,
		CI:     NewcombeDiffInterval(p1, n1, p2, n2, alpha),
		Z:      z,
		PValue: math.Min(1.0, pValue),
	}
}

var (
	ratHalf = new(big.Rat).SetFrac(big.NewInt(1), big.NewInt(2))
	ratOne  = new(big.Rat).SetInt64(1)
	// 1 - 1/10^12: the exact rational bound of Python's clamp
	ratOneMinusEps = new(big.Rat).SetFrac(big.NewInt(999999999999), big.NewInt(1000000000000))
)

// ratPow is exponentiation by squaring for exact rationals.
func ratPow(x *big.Rat, n int) *big.Rat {
	res := new(big.Rat).SetInt64(1)
	base := new(big.Rat).Set(x)
	for n > 0 {
		if n&1 == 1 {
			res.Mul(res, base)
		}
		base.Mul(base, base)
		n >>= 1
	}
	return res
}

// BinomCDF is the exact binomial CDF via exact rational arithmetic: p is
// taken as its exact binary rational, the pmf recurrence runs in exact
// rationals, and the sum is rounded to a double once. No libm dependence —
// bit-identical across platforms and language ports.
func BinomCDF(k, n int, p float64) float64 {
	if k < 0 || k > n {
		panic(fmt.Sprintf("k must be in [0, n], got %d/%d", k, n))
	}
	if p <= 0 {
		return 1.0
	}
	if p >= 1 {
		if k < n {
			return 0.0
		}
		return 1.0
	}
	var r *big.Rat
	if ratOf(p).Cmp(ratHalf) == 0 {
		s := new(big.Int)
		c := big.NewInt(1)
		s.Add(s, c)
		for i := 1; i <= k; i++ {
			c.Mul(c, big.NewInt(int64(n-i+1)))
			c.Div(c, big.NewInt(int64(i)))
			s.Add(s, c)
		}
		r = new(big.Rat).SetFrac(s, new(big.Int).Lsh(big.NewInt(1), uint(n)))
	} else {
		pf := ratOf(p)
		qf := new(big.Rat).Sub(new(big.Rat).SetInt64(1), pf)
		pmf := ratPow(qf, n)
		total := new(big.Rat).Set(pmf)
		for i := 1; i <= k; i++ {
			pmf.Mul(pmf, new(big.Rat).SetInt64(int64(n-i+1)))
			pmf.Mul(pmf, pf)
			pmf.Quo(pmf, new(big.Rat).SetInt64(int64(i)))
			pmf.Quo(pmf, qf)
			total.Add(total, pmf)
		}
		r = total
	}
	if r.Cmp(ratOneMinusEps) > 0 {
		return 1.0
	}
	if r.Cmp(ratOne) > 0 {
		return 1.0
	}
	f, _ := r.Float64()
	return f
}

// McNemarResult is the exact McNemar test for paired binary outcomes.
type McNemarResult struct {
	B      int
	C      int
	Diff   float64
	CI     Interval
	PValue float64
}

func (r McNemarResult) Significant() bool { return r.PValue < r.CI.Alpha }

// McNemarExact tests paired binary outcomes (b = A-solved-B-failed,
// c = A-failed-B-solved) with the exact conditional binomial on the
// discordant pairs — the correct instrument for benchmark A/B on a fixed
// task set. CI for d = (b - c)/N follows the paired-proportion interval
// var(d) = (b + c - (b - c)^2 / N) / N^2.
func McNemarExact(b, c int, alpha float64) McNemarResult {
	if b < 0 || c < 0 {
		panic(fmt.Sprintf("b, c must be >= 0, got %d, %d", b, c))
	}
	n := b + c
	if n == 0 {
		return McNemarResult{b, c, 0.0, Interval{0.0, 0.0, alpha, "mcnemar-degenerate"}, 1.0}
	}
	x := b
	if c < x {
		x = c
	}
	pValue := math.Min(1.0, 2*BinomCDF(x, n, 0.5))
	d := float64(b-c) / float64(n)
	bc := float64(b - c)
	varD := (float64(n) - bc*bc/float64(n)) / (float64(n) * float64(n))
	if varD < 0 {
		varD = 0.0
	}
	half := zTwoSided(alpha) * math.Sqrt(varD)
	return McNemarResult{
		B:      b,
		C:      c,
		Diff:   d,
		CI:     Interval{d - half, d + half, alpha, "mcnemar-wald"},
		PValue: pValue,
	}
}

// exactScaledInts is the exact dyadic representation of finite floats under
// a common scale: values[i] == ints[i] / scale exactly. ok=false when any
// value is not a finite float. Every finite float is m * 2^-e, so a common
// scale 2^e_max makes all values integers and integer sums exact.
func exactScaledInts(values []float64) (*big.Int, []*big.Int, bool) {
	type ratio struct {
		m *big.Int
		e int
	}
	ratios := make([]ratio, 0, len(values))
	eMax := 0
	for _, v := range values {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return nil, nil, false
		}
		r := new(big.Rat).SetFloat64(v)
		m := r.Num()
		d := r.Denom()
		e := d.BitLen() - 1
		if e > eMax {
			eMax = e
		}
		ratios = append(ratios, ratio{m, e})
	}
	scale := new(big.Int).Lsh(big.NewInt(1), uint(eMax))
	ints := make([]*big.Int, len(values))
	for i, rt := range ratios {
		shift := uint(eMax - rt.e)
		ints[i] = new(big.Int).Lsh(rt.m, shift)
	}
	return scale, ints, true
}

func percentileInterval(stats []float64, B int, alpha float64, method string) Interval {
	sort.Float64s(stats)
	loIdx := int(math.Floor(alpha/2*float64(B))) - 1
	if loIdx < 0 {
		loIdx = 0
	}
	hiIdx := int(math.Ceil((1-alpha/2)*float64(B))) - 1
	if hiIdx > B-1 {
		hiIdx = B - 1
	}
	return Interval{stats[loIdx], stats[hiIdx], alpha, method}
}

// exactClusterSums gives exact per-cluster sums as big integers on a common
// power-of-two scale.
func exactClusterSums(clusters [][]float64) (*big.Int, []*big.Int, []int, bool) {
	flat := make([]float64, 0, 32)
	for _, cl := range clusters {
		flat = append(flat, cl...)
	}
	scale, ints, ok := exactScaledInts(flat)
	if !ok {
		return nil, nil, nil, false
	}
	sums := make([]*big.Int, 0, len(clusters))
	counts := make([]int, 0, len(clusters))
	pos := 0
	for _, cl := range clusters {
		n := len(cl)
		s := new(big.Int)
		for _, x := range ints[pos : pos+n] {
			s.Add(s, x)
		}
		sums = append(sums, s)
		counts = append(counts, n)
		pos += n
	}
	return scale, sums, counts, true
}

// ClusterBootstrapCI is a percentile bootstrap CI that resamples clusters
// (tasks), not runs. Benchmark observations are not iid: runs on the same
// task share task difficulty. A run-level bootstrap pretends every run is
// exchangeable and understates variance when task effects dominate — the
// classic pseudoreplication failure. Resampling task clusters is the fix.
//
// Deterministic given seed. The default statistic (the pooled mean) takes
// an exact-arithmetic fast path: per-cluster sums are precomputed as
// integers on a common power-of-two scale and each resample is one integer
// sum plus one correctly-rounded division — bit-identical to the exact-mean
// generic path, ~10x faster. A nil stat selects the mean.
func ClusterBootstrapCI(clusters [][]float64, stat func([]float64) float64, B int, alpha float64, seed int) Interval {
	if B < 2 {
		panic(fmt.Sprintf("B must be >= 2, got %d", B))
	}
	if len(clusters) == 0 {
		panic("no clusters given")
	}
	rng := &pyRandom{}
	rng.seedInt(int64(seed))
	k := len(clusters)
	if stat == nil {
		scale, sums, counts, ok := exactClusterSums(clusters)
		if ok {
			stats := make([]float64, 0, B)
			denom := new(big.Int)
			total := new(big.Int)
			for i := 0; i < B; i++ {
				total.SetInt64(0)
				cnt := 0
				for j := 0; j < k; j++ {
					idx := rng.randrange(k)
					total.Add(total, sums[idx])
					cnt += counts[idx]
				}
				if cnt == 0 {
					panic("mean requires at least one data point")
				}
				denom.Mul(scale, big.NewInt(int64(cnt)))
				res := new(big.Rat).SetFrac(total, denom)
				f, _ := res.Float64()
				stats = append(stats, f)
			}
			return percentileInterval(stats, B, alpha, "cluster-bootstrap")
		}
		stat = exactMean
	}
	stats := make([]float64, 0, B)
	for i := 0; i < B; i++ {
		sample := make([][]float64, 0, k)
		for j := 0; j < k; j++ {
			sample = append(sample, clusters[rng.randrange(k)])
		}
		flat := make([]float64, 0, 16)
		for _, cl := range sample {
			flat = append(flat, cl...)
		}
		stats = append(stats, stat(flat))
	}
	return percentileInterval(stats, B, alpha, "cluster-bootstrap")
}

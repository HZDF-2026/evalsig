// The decision layer: statistically sound comparisons and candidate
// elimination. This is what API prompt/harness optimizers (DSPy-style
// pipelines, agent tuning loops) should call instead of comparing single
// noisy numbers. Every verdict comes with an interval, a p-value, and —
// when inconclusive — the number of additional runs that would make it
// conclusive.
package evalsig

import (
	"fmt"
	"math"
	"math/big"
	"sort"
)

// Comparison is one A/B verdict with everything needed to trust it.
type Comparison struct {
	NameA            string
	NameB            string
	Estimate         float64
	CI               Interval
	PValue           float64
	Method           string
	Paired           bool
	Verdict          string
	AdditionalNeeded int
}

func (c Comparison) String() string {
	extra := ""
	if c.AdditionalNeeded > 10000 {
		extra = fmt.Sprintf("; resolving an effect this small needs ~%d runs — too small to chase, treat as no difference", c.AdditionalNeeded)
	} else if c.AdditionalNeeded > 0 {
		extra = fmt.Sprintf(", ~%d more runs to resolve", c.AdditionalNeeded)
	}
	paired := ""
	if c.Paired {
		paired = "/paired"
	}
	return fmt.Sprintf("%s vs %s: %s CI %s p=%s [%s%s] -> %s%s",
		c.NameA, c.NameB,
		fmt.Sprintf("%+.4f", c.Estimate),
		c.CI.String(),
		pyFormatG(c.PValue, 4),
		c.Method, paired,
		c.Verdict, extra)
}

// taskGroups maps task -> outcomes, remembering first-encounter order
// (Python dicts iterate in insertion order, and the bootstrap CI is a
// deterministic function of that order).
type taskGroups struct {
	order  []string
	groups map[string][]float64
}

func byTask(runs []Run) *taskGroups {
	g := &taskGroups{groups: map[string][]float64{}}
	for _, r := range runs {
		if _, ok := g.groups[r.Task]; !ok {
			g.order = append(g.order, r.Task)
		}
		g.groups[r.Task] = append(g.groups[r.Task], r.Outcome)
	}
	return g
}

func isBinary(runs []Run) bool {
	for _, r := range runs {
		if r.Outcome != 0.0 && r.Outcome != 1.0 {
			return false
		}
	}
	return true
}

// Compare compares two run sets on the same tasks (paired) or
// independently. Pairing is auto-detected from task-id overlap — using it
// when available is not an option but the whole point: a fixed task set
// makes the paired design dramatically more powerful than two independent
// samples.
func Compare(runsA, runsB []Run, nameA, nameB string, alpha float64) (Comparison, error) {
	if len(runsA) == 0 || len(runsB) == 0 {
		return Comparison{}, fmt.Errorf("both run sets must be non-empty")
	}
	ta, tb := byTask(runsA), byTask(runsB)
	overlap := sortedStringSet(intersectKeys(ta, tb))
	paired := len(overlap) > 0 &&
		float64(len(overlap)) >= math.Max(0.5*float64(minInt(len(ta.order), len(tb.order))), 1)
	binary := isBinary(runsA) && isBinary(runsB)
	outA := make([]float64, 0, len(runsA))
	for _, r := range runsA {
		outA = append(outA, r.Outcome)
	}
	outB := make([]float64, 0, len(runsB))
	for _, r := range runsB {
		outB = append(outB, r.Outcome)
	}
	estA := exactMean(outA)
	estB := exactMean(outB)
	est := estA - estB

	if paired {
		diffs := make([]float64, 0, len(overlap))
		for _, t := range overlap {
			diffs = append(diffs, exactMean(ta.groups[t])-exactMean(tb.groups[t]))
		}
		clusters := make([][]float64, len(diffs))
		for i, d := range diffs {
			clusters[i] = []float64{d}
		}
		ci := ClusterBootstrapCI(clusters, nil, 2000, alpha, 1)
		var p float64
		var method string
		needed := 0
		if binary {
			b, c := 0, 0
			for _, t := range overlap {
				ma := exactMean(ta.groups[t])
				mb := exactMean(tb.groups[t])
				if ma > mb {
					b++
				} else if ma < mb {
					c++
				}
			}
			mc := McNemarExact(b, c, alpha)
			p = mc.PValue
			method = fmt.Sprintf("mcnemar (b=%d, c=%d)", b, c)
			needed = additionalPaired(b, c, est, len(overlap), alpha)
		} else {
			p = signFlipP(diffs)
			method = "paired-bootstrap+signflip"
		}
		verdict := verdictOf(est, ci, p, alpha)
		return Comparison{nameA, nameB, est, ci, p, method, true, verdict, needed}, nil
	}

	var ci Interval
	var p float64
	var method string
	if binary {
		sumA, sumB := 0.0, 0.0
		for _, v := range outA {
			sumA += v
		}
		for _, v := range outB {
			sumB += v
		}
		res := TwoProportionTest(int(pyRound(sumA)), len(runsA), int(pyRound(sumB)), len(runsB), alpha)
		ci, p, method = res.CI, res.PValue, "two-prop-z"
	} else {
		ci = diffBootstrap(ta, tb, alpha)
		p = math.NaN()
		method = "bootstrap-diff"
	}
	verdict := verdictOf(est, ci, p, alpha)
	needed := additionalUnpaired(estA, estB, len(runsA), len(runsB), alpha, binary)
	return Comparison{nameA, nameB, est, ci, p, method, false, verdict, needed}, nil
}

func intersectKeys(a, b *taskGroups) []string {
	var keys []string
	for _, k := range a.order {
		if _, ok := b.groups[k]; ok {
			keys = append(keys, k)
		}
	}
	return keys
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func verdictOf(est float64, ci Interval, p, alpha float64) string {
	if math.IsNaN(p) { // bootstrap-only path, the CI carries the decision
		if ci.Excludes(0) {
			if est > 0 {
				return "A-BETTER"
			}
			return "B-BETTER"
		}
		return "INCONCLUSIVE"
	}
	if p < alpha && ci.Excludes(0) {
		if est > 0 {
			return "A-BETTER"
		}
		return "B-BETTER"
	}
	return "INCONCLUSIVE"
}

// signFlipP is the two-sided sign-flip permutation p-value for the mean of
// paired diffs. Exact under symmetry of the diff distribution; no small-n
// normality assumptions. Enumeration is 2^n, so n > 20 falls back to the
// normal approximation to the sign-flip null.
func signFlipP(diffs []float64) float64 {
	if len(diffs) == 0 {
		return 1.0
	}
	obs := math.Abs(exactMean(diffs))
	n := len(diffs)
	if n > 20 {
		se := exactStdev(diffs) / math.Sqrt(float64(n))
		if se == 0 {
			return 1.0
		}
		return math.Min(1.0, 2*(1-normCDF(obs/se)))
	}
	count := 0
	for mask := 0; mask < 1<<n; mask++ {
		s := 0.0
		for i, d := range diffs {
			if (mask>>i)&1 == 1 {
				s += d
			} else {
				s -= d
			}
		}
		if math.Abs(s/float64(n)) >= obs-1e-12 {
			count++
		}
	}
	return float64(count) / float64(int(1)<<n)
}

// diffBootstrap is the task-cluster bootstrap CI for the difference of two
// unpaired means.
func diffBootstrap(ta, tb *taskGroups, alpha float64) Interval {
	rng := &pyRandom{}
	rng.seedInt(2)
	ma := make([]float64, 0, len(ta.order))
	for _, k := range ta.order {
		ma = append(ma, exactMean(ta.groups[k]))
	}
	mb := make([]float64, 0, len(tb.order))
	for _, k := range tb.order {
		mb = append(mb, exactMean(tb.groups[k]))
	}
	scaleA, intsA, okA := exactScaledInts(ma)
	scaleB, intsB, okB := exactScaledInts(mb)
	const B = 2000
	stats := make([]float64, 0, B)
	if okA && okB {
		ka, kb := len(ma), len(mb)
		totalA := newInt()
		totalB := newInt()
		denA := newInt().Mul(scaleA, bigOf(ka))
		denB := newInt().Mul(scaleB, bigOf(kb))
		for i := 0; i < B; i++ {
			totalA.SetInt64(0)
			for j := 0; j < ka; j++ {
				totalA.Add(totalA, intsA[rng.randrange(ka)])
			}
			totalB.SetInt64(0)
			for j := 0; j < kb; j++ {
				totalB.Add(totalB, intsB[rng.randrange(kb)])
			}
			fa, _ := new(big.Rat).SetFrac(totalA, denA).Float64()
			fb, _ := new(big.Rat).SetFrac(totalB, denB).Float64()
			stats = append(stats, fa-fb)
		}
	} else {
		for i := 0; i < B; i++ {
			sa := make([]float64, 0, len(ma))
			for j := 0; j < len(ma); j++ {
				sa = append(sa, ma[rng.randrange(len(ma))])
			}
			sb := make([]float64, 0, len(mb))
			for j := 0; j < len(mb); j++ {
				sb = append(sb, mb[rng.randrange(len(mb))])
			}
			stats = append(stats, exactMean(sa)-exactMean(sb))
		}
	}
	sort.Float64s(stats)
	loIdx := int(alpha/2*2000) - 1
	if loIdx < 0 {
		loIdx = 0
	}
	hiIdx := int((1-alpha/2)*2000) - 1
	if hiIdx > 1999 {
		hiIdx = 1999
	}
	return Interval{stats[loIdx], stats[hiIdx], alpha, "bootstrap-diff"}
}

func newInt() *big.Int { return new(big.Int) }

func bigOf(n int) *big.Int { return big.NewInt(int64(n)) }

func additionalPaired(b, c int, est float64, nTasks int, alpha float64) int {
	if b+c > 0 && math.Abs(est) > 0 {
		disc := float64(b+c) / float64(maxInt(1, nTasks))
		plan := NPaired(disc, math.Abs(est), alpha, 0.8)
		return maxInt(0, plan.NPerVersion-nTasks)
	}
	return 0
}

func additionalUnpaired(pa, pb float64, na, nb int, alpha float64, binary bool) int {
	if !binary || math.Abs(pa-pb) == 0 || pa+pb == 0.0 || pa+pb == 2.0 {
		return 0
	}
	plan := NTwoProportions(pa, pb, alpha, 0.8)
	return maxInt(0, maxInt(plan.NPerVersion-na, plan.NPerVersion-nb))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// HolmEntry is one candidate's Holm-adjusted p-value and reject decision.
type HolmEntry struct {
	Name     string
	Adjusted float64
	Reject   bool
}

// Holm applies Holm-Bonferroni step-down: adjusted p-values and reject
// decisions. Mandatory when an optimizer compares several candidates
// against one baseline — k uncorrected comparisons at alpha mean up to
// k*alpha actual false-positive rate, which is precisely the failure mode
// evalsig exists to close. entries must be in comparison order; ties keep
// that order (Python's sort is stable).
func Holm(entries []HolmP, alpha float64) []HolmEntry {
	items := make([]HolmP, len(entries))
	copy(items, entries)
	sort.SliceStable(items, func(i, j int) bool { return items[i].P < items[j].P })
	out := make([]HolmEntry, 0, len(items))
	running := 0.0
	m := len(items)
	for i, it := range items {
		if v := it.P * float64(m-i); v > running {
			running = v
		}
		adj := math.Min(1.0, running)
		out = append(out, HolmEntry{it.Name, adj, adj < alpha})
	}
	return out
}

// HolmP is a candidate's raw p-value for Holm().
type HolmP struct {
	Name string
	P    float64
}

// RankingEntry is one line of the decide() ranking.
type RankingEntry struct {
	Name     string
	Estimate float64
	Action   string
}

// DecisionReport is the decide() output: estimates, adjusted p-values,
// pairwise comparisons against the baseline, and the ranking.
type DecisionReport struct {
	Estimates   []RankingEntry // name -> estimate, insertion order
	Adjusted    []HolmEntry    // Holm order
	Comparisons []Comparison
	Ranking     []RankingEntry
}

func (r DecisionReport) String() string {
	lines := []string{"ranking (estimate / Holm-adjusted elimination):"}
	for _, e := range r.Ranking {
		lines = append(lines, fmt.Sprintf("  %-24s %+.4f  %s", e.Name, e.Estimate, e.Action))
	}
	for _, c := range r.Comparisons {
		lines = append(lines, "  "+c.String())
	}
	return joinLines(lines)
}

func joinLines(lines []string) string {
	out := ""
	for i, l := range lines {
		if i > 0 {
			out += "\n"
		}
		out += l
	}
	return out
}

// Candidate is a named run set; order of the slice is the candidate order.
type Candidate struct {
	Name string
	Runs []Run
}

// Decide ranks candidates and eliminates the ones that are statistically
// worse. The current best (highest mean outcome) becomes the baseline;
// every other candidate is compared against it with Holm correction across
// the k-1 comparisons. Candidates flagged ELIMINATE are safe to drop from
// the optimizer's population.
func Decide(cands []Candidate, alpha float64) (DecisionReport, error) {
	if len(cands) < 2 {
		return DecisionReport{}, fmt.Errorf("decide needs at least two candidates")
	}
	estimates := make([]RankingEntry, 0, len(cands))
	runsByName := map[string][]Run{}
	for _, c := range cands {
		outs := make([]float64, 0, len(c.Runs))
		for _, r := range c.Runs {
			outs = append(outs, r.Outcome)
		}
		estimates = append(estimates, RankingEntry{c.Name, exactMean(outs), ""})
		runsByName[c.Name] = c.Runs
	}
	order := make([]RankingEntry, len(estimates))
	copy(order, estimates)
	sort.SliceStable(order, func(i, j int) bool { return order[j].Estimate < order[i].Estimate })
	best := order[0].Name
	comparisons := make([]Comparison, 0, len(order)-1)
	pvals := make([]HolmP, 0, len(order)-1)
	for _, e := range order[1:] {
		comp, err := Compare(runsByName[best], runsByName[e.Name], best, e.Name, alpha)
		if err != nil {
			return DecisionReport{}, err
		}
		comparisons = append(comparisons, comp)
		p := comp.PValue
		if math.IsNaN(p) {
			p = 1.0
		}
		pvals = append(pvals, HolmP{e.Name, p})
	}
	var adjusted []HolmEntry
	if len(pvals) > 0 {
		adjusted = Holm(pvals, alpha)
	}
	adjustedByName := map[string]HolmEntry{}
	for _, h := range adjusted {
		adjustedByName[h.Name] = h
	}
	ranking := []RankingEntry{{best, order[0].Estimate, "KEEP (baseline)"}}
	for _, e := range order[1:] {
		if h, ok := adjustedByName[e.Name]; ok && h.Reject {
			ranking = append(ranking, RankingEntry{e.Name, e.Estimate, "ELIMINATE (worse)"})
		} else {
			ranking = append(ranking, RankingEntry{e.Name, e.Estimate, "KEEP (not separable)"})
		}
	}
	return DecisionReport{Estimates: estimates, Adjusted: adjusted, Comparisons: comparisons, Ranking: ranking}, nil
}

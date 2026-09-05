// Variance attribution: where does run-to-run noise actually come from?
//
// One-way random-effects decomposition (method of moments). Given runs
// grouped by a suspected noise source (seed, engine version, environment
// instance), splits total variance into between-group and within-group
// components and reports the ICC — the fraction of noise attributable to
// the grouping factor.
package evalsig

import (
	"fmt"
	"sort"
)

// VarianceComponents is the method-of-moments decomposition.
type VarianceComponents struct {
	Between    float64
	Within     float64
	Total      float64
	ICC        float64
	NGroups    int
	NObs       int
	GroupSizes []int
}

func (v VarianceComponents) Summary() string {
	return fmt.Sprintf("between=%s within=%s icc=%s (groups=%d, obs=%d)",
		pyFormatG(v.Between, 4), pyFormatG(v.Within, 4), pyFormatF(v.ICC, 3), v.NGroups, v.NObs)
}

// group is an insertion-ordered (first-encounter) group of outcomes; the
// order matters because the float accumulations below follow it exactly.
type group struct {
	key   string
	vals  []float64
}

// VarianceComponentsOf computes the one-way random effects decomposition.
//
// sigma^2_between = max(0, (MSB - MSW) / n0), with the unbalanced-design
// effective group size n0 = (N - sum(n_i^2) / N) / (a - 1). Negative
// variance estimates are truncated at zero (standard practice; the
// truncation makes the estimator biased conservative, which is the safe
// direction for a noise audit).
func VarianceComponentsOf(groups []*group) VarianceComponents {
	cleaned := make([]*group, 0, len(groups))
	for _, g := range groups {
		if len(g.vals) > 0 {
			cleaned = append(cleaned, g)
		}
	}
	if len(cleaned) == 0 {
		panic("no non-empty groups")
	}
	sizes := make([]int, len(cleaned))
	for i, g := range cleaned {
		sizes[i] = len(g.vals)
	}
	a := len(cleaned)
	N := 0
	for _, s := range sizes {
		N += s
	}
	if a < 2 {
		// single group: all variance is within, none between
		msw := mWithin(cleaned, N, a)
		return VarianceComponents{0.0, msw, msw, 0.0, a, N, sizes}
	}
	flat := make([]float64, 0, N)
	for _, g := range cleaned {
		flat = append(flat, g.vals...)
	}
	grand := exactMean(flat)
	ssb := 0.0
	for _, g := range cleaned {
		m := exactMean(g.vals)
		d := m - grand
		sq := d * d
		ssb += pyMul(float64(len(g.vals)), sq)
	}
	msb := ssb / float64(a-1)
	msw := mWithin(cleaned, N, a)
	sumSq := 0
	for _, s := range sizes {
		sumSq += s * s
	}
	n0 := (float64(N) - float64(sumSq)/float64(N)) / float64(a-1)
	if n0 <= 0 {
		n0 = 1.0
	}
	between := (msb - msw) / n0
	if between < 0 {
		between = 0.0
	}
	within := msw
	if within < 0 {
		within = 0.0
	}
	total := between + within
	icc := 0.0
	if total > 0 {
		icc = between / total
	}
	return VarianceComponents{between, within, total, icc, a, N, sizes}
}

func mWithin(groups []*group, N, a int) float64 {
	if N <= a {
		return 0.0
	}
	ssw := 0.0
	for _, g := range groups {
		m := exactMean(g.vals)
		inner := 0.0
		for _, v := range g.vals {
			d := v - m
			inner += pyMul(d, d)
		}
		ssw += inner
	}
	return ssw / float64(N-a)
}

// Attribution is the variance_attribution result for one factor.
type Attribution struct {
	Factor     string
	Components VarianceComponents
	GroupMeans []GroupMean // sorted by group key
}

// GroupMean is one group's mean, in sorted-key order.
type GroupMean struct {
	Key  string
	Mean float64
}

// VarianceAttribution splits runs by metadata factor and decomposes outcome
// variance. Groups appear in first-encounter order for the float-sum
// fidelity of the decomposition, and the reported means in sorted-key order
// (Python iterates sorted(groups.items())).
func VarianceAttribution(runs []Run, factor string) Attribution {
	var groups []*group
	index := map[string]*group{}
	for _, r := range runs {
		v, ok := r.factorValue(factor)
		if !ok {
			continue // Python: key is None -> skip
		}
		key := pyStr(v)
		g, seen := index[key]
		if !seen {
			g = &group{key: key}
			index[key] = g
			groups = append(groups, g)
		}
		g.vals = append(g.vals, r.Outcome)
	}
	comp := VarianceComponentsOf(groups)
	means := make([]GroupMean, 0, len(groups))
	for _, g := range groups {
		means = append(means, GroupMean{g.key, exactMean(g.vals)})
	}
	sort.Slice(means, func(i, j int) bool { return means[i].Key < means[j].Key })
	return Attribution{factor, comp, means}
}

// OutcomeError reports a run dict without any recognized outcome field.
type OutcomeError struct {
	Keys []string
}

func (e *OutcomeError) Error() string {
	return fmt.Sprintf("run has no outcome field (looked for success/resolved/passed/score/value): %v", e.Keys)
}

// variance.h — variance attribution: where does run-to-run noise actually
// come from? Port of variance.go.
//
// One-way random-effects decomposition (method of moments). Given runs
// grouped by a suspected noise source (seed, engine version, environment
// instance), splits total variance into between-group and within-group
// components and reports the ICC — the fraction of noise attributable to
// the grouping factor.
#pragma once

#include <string>
#include <vector>

#include "pyjson.h"

namespace evalsig {

// Method-of-moments decomposition.
struct VarianceComponents {
    double between = 0.0;
    double within = 0.0;
    double total = 0.0;
    double icc = 0.0;
    int nGroups = 0;
    int nObs = 0;
    std::vector<int> groupSizes;

    std::string summary() const;
};

// An insertion-ordered (first-encounter) group of outcomes; the order
// matters because the float accumulations follow it exactly.
struct Group {
    std::string key;
    std::vector<double> vals;
};

// One-way random effects decomposition.
//
// sigma^2_between = max(0, (MSB - MSW) / n0), with the unbalanced-design
// effective group size n0 = (N - sum(n_i^2) / N) / (a - 1). Negative
// variance estimates are truncated at zero (standard practice; the
// truncation makes the estimator biased conservative, which is the safe
// direction for a noise audit). Throws when no group is non-empty.
VarianceComponents varianceComponentsOf(std::vector<const Group*> groups);

// One group's mean, in sorted-key order.
struct GroupMean {
    std::string key;
    double mean = 0.0;
};

// The variance_attribution result for one factor.
struct Attribution {
    std::string factor;
    VarianceComponents components;
    std::vector<GroupMean> groupMeans;  // sorted by group key
};

// Splits runs by metadata factor and decomposes outcome variance. Groups
// appear in first-encounter order for the float-sum fidelity of the
// decomposition, and the reported means in sorted-key order.
Attribution varianceAttribution(const std::vector<Run>& runs, const std::string& factor);

}  // namespace evalsig

// intervals.h — confidence intervals and exact tests for benchmark
// proportions. Port of intervals.go.
//
// All estimators are chosen for validity at small n and near-boundary
// success rates — the regime where agent benchmarks actually live.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace evalsig {

// Confidence interval with the alpha it was built at.
struct Interval {
    double low = 0.0;
    double high = 0.0;
    double alpha = 0.0;
    std::string method;

    Interval() = default;
    Interval(double lo, double hi, double a, std::string m)
        : low(lo), high(hi), alpha(a), method(std::move(m)) {}

    double width() const { return high - low; }
    double half() const { return width() / 2; }
    double center() const { return (low + high) / 2; }
    bool excludes(double value) const { return value < low || value > high; }
    std::string str() const;
};

// z_{alpha/2}, the two-sided normal quantile. Throws on alpha outside (0,1).
double zTwoSided(double alpha);

// Wilson score interval for a binomial proportion. Throws on bad s/n.
Interval wilsonInterval(int successes, int n, double alpha);

// The same interval via direct quadratic roots — an independent derivation
// used as a cross-check.
Interval wilsonIntervalRoots(int successes, int n, double alpha);

// Newcombe hybrid-score CI for p1 - p2 (unpaired). Newcombe, Stat. Med. 17
// (1998), method 10.
Interval newcombeDiffInterval(double p1, int n1, double p2, int n2, double alpha);

// Pooled two-proportion z-test plus the Newcombe CI.
struct TwoPropResult {
    double diff = 0.0;
    Interval ci;
    double z = 0.0;
    double pValue = 0.0;

    bool significant() const { return pValue < ci.alpha; }
};

TwoPropResult twoProportionTest(int s1, int n1, int s2, int n2, double alpha);

// Exact binomial CDF via exact rational arithmetic: bit-identical across
// platforms and language ports. Throws on k outside [0, n].
double binomCDF(int k, int n, double p);

// Exact McNemar test for paired binary outcomes.
struct McNemarResult {
    int b = 0;
    int c = 0;
    double diff = 0.0;
    Interval ci;
    double pValue = 0.0;

    bool significant() const { return pValue < ci.alpha; }
};

McNemarResult mcnemarExact(int b, int c, double alpha);

// Percentile bootstrap CI that resamples clusters (tasks), not runs.
// Deterministic given seed. A null stat selects the exact-arithmetic mean
// fast path. Throws on B < 2, empty clusters, or an all-empty resample.
Interval clusterBootstrapCI(const std::vector<std::vector<double>>& clusters,
                            const std::function<double(const std::vector<double>&)>& stat,
                            int B, double alpha, int seed);

}  // namespace evalsig

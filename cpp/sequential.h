// sequential.h — sequential stopping: end the eval when the number, not the
// budget, says so. Port of sequential.go.
//
// The statistical hazard of "run until it looks significant" is real and
// mostly invisible: peeking at a CI after every batch and stopping on the
// first hit inflates the false-positive rate far above the nominal alpha.
// The honest fixes are (a) a pre-registered look schedule with alpha
// correction, and (b) stopping on precision (CI width), which does not
// inflate type-I error at all.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intervals.h"
#include "pyjson.h"

namespace evalsig {

// A pre-registered doubling look schedule with Bonferroni alpha spending.
// Checks happen only at n = n0, 2*n0, 4*n0, ... up to maxLooks. Each look
// tests at alpha/K (Bonferroni over all K looks), which keeps the family
// type-I error <= alpha under the registered schedule.
struct LookSchedule {
    int n0 = 25;
    int maxLooks = 10;

    std::vector<int> ns() const;
    int nMax() const { return n0 * (1 << (maxLooks - 1)); }
    // Index of the largest scheduled look <= n, or -1 before the first.
    int lookFor(int n) const;
};

// Tracks one proportion; stop on significance vs a reference or on width.
// Width-based stopping ("PRECISION-REACHED") is the estimator-friendly exit:
// it ends the run once the CI is tight enough to be useful, with no effect
// on error rates. Significance checking only fires at scheduled looks.
struct SequentialProportion {
    double reference = 0.5;
    double alpha = 0.0;
    double targetHalfWidth = 0.0;
    LookSchedule schedule;
    int successes = 0;
    int n = 0;
    int looksUsed = 0;
    std::string verdict = "CONTINUE";

    SequentialProportion(double alpha, double targetHalfWidth)
        : alpha(alpha), targetHalfWidth(targetHalfWidth) {}

    std::string update(int successes, int n);
    // Plain-alpha Wilson CI at the current n (reporting, not testing).
    Interval ci() const;
    // The report() dict, in Python's key order.
    PyValue report() const;
};

// Sequential A/B on two run streams of binary outcomes. Same discipline as
// SequentialProportion: scheduled looks with Bonferroni alpha for the
// significance exit, any-n width exit for precision.
struct SequentialAB {
    double alpha = 0.0;
    double targetHalfWidth = 0.0;
    LookSchedule schedule;
    int s1 = 0, n1 = 0;
    int s2 = 0, n2 = 0;
    int looksUsed = 0;
    std::string verdict = "CONTINUE";

    SequentialAB(double alpha, double targetHalfWidth)
        : alpha(alpha), targetHalfWidth(targetHalfWidth) {}

    std::string update(int s1, int n1, int s2, int n2);
    // The report() dict, in Python's key order.
    PyValue report() const;
};

}  // namespace evalsig

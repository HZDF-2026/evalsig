// decide.h — the decision layer: statistically sound comparisons and
// candidate elimination. Port of decide.go.
//
// This is what API prompt/harness optimizers (DSPy-style pipelines, agent
// tuning loops) should call instead of comparing single noisy numbers.
// Every verdict comes with an interval, a p-value, and — when inconclusive —
// the number of additional runs that would make it conclusive.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "intervals.h"
#include "pyjson.h"
#include <stdexcept>

namespace evalsig {

// One A/B verdict with everything needed to trust it.
struct Comparison {
    std::string nameA;
    std::string nameB;
    double estimate = 0.0;
    Interval ci;
    double pValue = 0.0;
    std::string method;
    bool paired = false;
    std::string verdict;
    int additionalNeeded = 0;

    std::string str() const;
};

// task -> outcomes, remembering first-encounter order (Python dicts iterate
// in insertion order, and the bootstrap CI is a deterministic function of
// that order).
struct TaskGroups {
    std::vector<std::string> order;
    std::map<std::string, std::vector<double>> groups;

    bool has(const std::string& task) const { return groups.count(task) > 0; }
    const std::vector<double>& at(const std::string& task) const { return groups.at(task); }
};

TaskGroups byTask(const std::vector<Run>& runs);
bool isBinary(const std::vector<Run>& runs);

// Compares two run sets on the same tasks (paired) or independently.
// Pairing is auto-detected from task-id overlap — using it when available is
// not an option but the whole point: a fixed task set makes the paired
// design dramatically more powerful than two independent samples.
// Throws std::runtime_error when either run set is empty.
Comparison compare(const std::vector<Run>& runsA, const std::vector<Run>& runsB,
                    const std::string& nameA, const std::string& nameB, double alpha);

// Two-sided sign-flip permutation p-value for the mean of paired diffs.
double signFlipP(const std::vector<double>& diffs);

// Task-cluster bootstrap CI for the difference of two unpaired means.
Interval diffBootstrap(const TaskGroups& ta, const TaskGroups& tb, double alpha);

// One candidate's Holm-adjusted p-value and reject decision.
struct HolmEntry {
    std::string name;
    double adjusted = 0.0;
    bool reject = false;
};

// A candidate's raw p-value for holm().
struct HolmP {
    std::string name;
    double p = 0.0;
};

// Holm-Bonferroni step-down: adjusted p-values and reject decisions.
// Mandatory when an optimizer compares several candidates against one
// baseline — k uncorrected comparisons at alpha mean up to k*alpha actual
// false-positive rate. entries must be in comparison order; ties keep that
// order (Python's sort is stable).
std::vector<HolmEntry> holm(std::vector<HolmP> entries, double alpha);

// One line of the decide() ranking.
struct RankingEntry {
    std::string name;
    double estimate = 0.0;
    std::string action;
};

// The decide() output: estimates, adjusted p-values, pairwise comparisons
// against the baseline, and the ranking.
struct DecisionReport {
    std::vector<RankingEntry> estimates;  // name -> estimate, insertion order
    std::vector<HolmEntry> adjusted;      // Holm order
    std::vector<Comparison> comparisons;
    std::vector<RankingEntry> ranking;

    std::string str() const;
};

// A named run set; order of the slice is the candidate order.
struct Candidate {
    std::string name;
    std::vector<Run> runs;
};

// Ranks candidates and eliminates the ones that are statistically worse.
// The current best (highest mean outcome) becomes the baseline; every other
// candidate is compared against it with Holm correction across the k-1
// comparisons. Candidates flagged ELIMINATE are safe to drop from the
// optimizer's population. Throws std::runtime_error on bad input.
DecisionReport decide(const std::vector<Candidate>& cands, double alpha);

}  // namespace evalsig

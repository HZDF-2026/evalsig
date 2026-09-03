// decide.cpp — see decide.h. Port of decide.go.

#include "decide.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "power.h"
#include "pynum.h"

namespace evalsig {

std::string Comparison::str() const {
    std::string extra;
    if (additionalNeeded > 10000) {
        extra = "; resolving an effect this small needs ~" + std::to_string(additionalNeeded) +
                " runs — too small to chase, treat as no difference";
    } else if (additionalNeeded > 0) {
        extra = ", ~" + std::to_string(additionalNeeded) + " more runs to resolve";
    }
    std::string pairedSuffix = paired ? "/paired" : "";
    return nameA + " vs " + nameB + ": " + fmtPlusF(estimate, 4) + " CI " + ci.str() +
           " p=" + pyFormatG(pValue, 4) + " [" + method + pairedSuffix + "] -> " + verdict + extra;
}

TaskGroups byTask(const std::vector<Run>& runs) {
    TaskGroups g;
    for (const Run& r : runs) {
        if (!g.has(r.task)) {
            g.order.push_back(r.task);
        }
        g.groups[r.task].push_back(r.outcome);
    }
    return g;
}

bool isBinary(const std::vector<Run>& runs) {
    for (const Run& r : runs) {
        if (r.outcome != 0.0 && r.outcome != 1.0) {
            return false;
        }
    }
    return true;
}

namespace {

std::vector<std::string> intersectKeys(const TaskGroups& a, const TaskGroups& b) {
    std::vector<std::string> keys;
    for (const auto& k : a.order) {
        if (b.has(k)) {
            keys.push_back(k);
        }
    }
    return keys;
}

int minInt(int a, int b) { return a < b ? a : b; }
int maxInt(int a, int b) { return a > b ? a : b; }

std::string verdictOf(double est, const Interval& ci, double p, double alpha) {
    if (std::isnan(p)) {  // bootstrap-only path, the CI carries the decision
        if (ci.excludes(0)) {
            return est > 0 ? "A-BETTER" : "B-BETTER";
        }
        return "INCONCLUSIVE";
    }
    if (p < alpha && ci.excludes(0)) {
        return est > 0 ? "A-BETTER" : "B-BETTER";
    }
    return "INCONCLUSIVE";
}

int additionalPaired(int b, int c, double est, int nTasks, double alpha) {
    if (b + c > 0 && std::fabs(est) > 0) {
        double disc = static_cast<double>(b + c) / static_cast<double>(maxInt(1, nTasks));
        PowerPlan plan = nPaired(disc, std::fabs(est), alpha, 0.8);
        return maxInt(0, plan.nPerVersion - nTasks);
    }
    return 0;
}

int additionalUnpaired(double pa, double pb, int na, int nb, double alpha, bool binary) {
    if (!binary || std::fabs(pa - pb) == 0 || pa + pb == 0.0 || pa + pb == 2.0) {
        return 0;
    }
    PowerPlan plan = nTwoProportions(pa, pb, alpha, 0.8);
    return maxInt(0, maxInt(plan.nPerVersion - na, plan.nPerVersion - nb));
}

}  // namespace

double signFlipP(const std::vector<double>& diffs) {
    size_t n = diffs.size();
    if (n == 0) {
        return 1.0;
    }
    double obs = std::fabs(exactMean(diffs));
    if (n > 20) {
        double se = exactStdev(diffs) / std::sqrt(static_cast<double>(n));
        if (se == 0) {
            return 1.0;
        }
        return std::min(1.0, 2 * (1 - normCDF(obs / se)));
    }
    long long count = 0;
    for (long long mask = 0; mask < (1ll << n); mask++) {
        double s = 0.0;
        for (size_t i = 0; i < n; i++) {
            if ((mask >> i) & 1) {
                s += diffs[i];
            } else {
                s -= diffs[i];
            }
        }
        if (std::fabs(s / static_cast<double>(n)) >= obs - 1e-12) {
            count++;
        }
    }
    return static_cast<double>(count) / static_cast<double>(1ll << n);
}

Interval diffBootstrap(const TaskGroups& ta, const TaskGroups& tb, double alpha) {
    PyRandom rng;
    rng.seedInt(2);
    std::vector<double> ma;
    ma.reserve(ta.order.size());
    for (const auto& k : ta.order) {
        ma.push_back(exactMean(ta.at(k)));
    }
    std::vector<double> mb;
    mb.reserve(tb.order.size());
    for (const auto& k : tb.order) {
        mb.push_back(exactMean(tb.at(k)));
    }
    ScaledInts sa = exactScaledInts(ma);
    ScaledInts sb = exactScaledInts(mb);
    const int B = 2000;
    std::vector<double> stats;
    stats.reserve(B);
    if (sa.ok && sb.ok) {
        int ka = static_cast<int>(ma.size());
        int kb = static_cast<int>(mb.size());
        BigInt denA = sa.scale * BigInt::fromI64(ka);
        BigInt denB = sb.scale * BigInt::fromI64(kb);
        for (int i = 0; i < B; i++) {
            BigInt totalA;
            for (int j = 0; j < ka; j++) {
                totalA = totalA + sa.ints[static_cast<size_t>(rng.randrange(ka))];
            }
            BigInt totalB;
            for (int j = 0; j < kb; j++) {
                totalB = totalB + sb.ints[static_cast<size_t>(rng.randrange(kb))];
            }
            double fa = Rat{totalA, denA}.toF64();
            double fb = Rat{totalB, denB}.toF64();
            stats.push_back(fa - fb);
        }
    } else {
        for (int i = 0; i < B; i++) {
            std::vector<double> sampled;
            sampled.reserve(ma.size());
            for (size_t j = 0; j < ma.size(); j++) {
                sampled.push_back(ma[static_cast<size_t>(rng.randrange(static_cast<int>(ma.size())))]);
            }
            std::vector<double> sampledB;
            sampledB.reserve(mb.size());
            for (size_t j = 0; j < mb.size(); j++) {
                sampledB.push_back(mb[static_cast<size_t>(rng.randrange(static_cast<int>(mb.size())))]);
            }
            stats.push_back(exactMean(sampled) - exactMean(sampledB));
        }
    }
    std::sort(stats.begin(), stats.end());
    int loIdx = static_cast<int>(alpha / 2 * 2000) - 1;
    if (loIdx < 0) {
        loIdx = 0;
    }
    int hiIdx = static_cast<int>((1 - alpha / 2) * 2000) - 1;
    if (hiIdx > 1999) {
        hiIdx = 1999;
    }
    return Interval(stats[static_cast<size_t>(loIdx)], stats[static_cast<size_t>(hiIdx)], alpha,
                    "bootstrap-diff");
}

Comparison compare(const std::vector<Run>& runsA, const std::vector<Run>& runsB,
                    const std::string& nameA, const std::string& nameB, double alpha) {
    if (runsA.empty() || runsB.empty()) {
        throw std::runtime_error("both run sets must be non-empty");
    }
    TaskGroups ta = byTask(runsA);
    TaskGroups tb = byTask(runsB);
    std::vector<std::string> overlap = sortedStringSet(intersectKeys(ta, tb));
    bool paired = !overlap.empty() &&
                  static_cast<double>(overlap.size()) >=
                      std::max(0.5 * static_cast<double>(minInt(static_cast<int>(ta.order.size()),
                                                               static_cast<int>(tb.order.size()))),
                               1.0);
    bool binary = isBinary(runsA) && isBinary(runsB);
    std::vector<double> outA;
    outA.reserve(runsA.size());
    for (const Run& r : runsA) {
        outA.push_back(r.outcome);
    }
    std::vector<double> outB;
    outB.reserve(runsB.size());
    for (const Run& r : runsB) {
        outB.push_back(r.outcome);
    }
    double estA = exactMean(outA);
    double estB = exactMean(outB);
    double est = estA - estB;

    if (paired) {
        std::vector<double> diffs;
        diffs.reserve(overlap.size());
        for (const auto& t : overlap) {
            diffs.push_back(exactMean(ta.at(t)) - exactMean(tb.at(t)));
        }
        std::vector<std::vector<double>> clusters;
        clusters.reserve(diffs.size());
        for (double d : diffs) {
            clusters.push_back({d});
        }
        Interval ci = clusterBootstrapCI(clusters, nullptr, 2000, alpha, 1);
        double p;
        std::string method;
        int needed = 0;
        if (binary) {
            int b = 0, c = 0;
            for (const auto& t : overlap) {
                double maVal = exactMean(ta.at(t));
                double mbVal = exactMean(tb.at(t));
                if (maVal > mbVal) {
                    b++;
                } else if (maVal < mbVal) {
                    c++;
                }
            }
            McNemarResult mc = mcnemarExact(b, c, alpha);
            p = mc.pValue;
            method = "mcnemar (b=" + std::to_string(b) + ", c=" + std::to_string(c) + ")";
            needed = additionalPaired(b, c, est, static_cast<int>(overlap.size()), alpha);
        } else {
            p = signFlipP(diffs);
            method = "paired-bootstrap+signflip";
        }
        Comparison cmp;
        cmp.nameA = nameA;
        cmp.nameB = nameB;
        cmp.estimate = est;
        cmp.ci = ci;
        cmp.pValue = p;
        cmp.method = method;
        cmp.paired = true;
        cmp.verdict = verdictOf(est, ci, p, alpha);
        cmp.additionalNeeded = needed;
        return cmp;
    }

    Interval ci;
    double p;
    std::string method;
    if (binary) {
        double sumA = 0.0;
        double sumB = 0.0;
        for (double v : outA) {
            sumA += v;
        }
        for (double v : outB) {
            sumB += v;
        }
        TwoPropResult res = twoProportionTest(static_cast<int>(pyRound(sumA)),
                                              static_cast<int>(runsA.size()),
                                              static_cast<int>(pyRound(sumB)),
                                              static_cast<int>(runsB.size()), alpha);
        ci = res.ci;
        p = res.pValue;
        method = "two-prop-z";
    } else {
        ci = diffBootstrap(ta, tb, alpha);
        p = std::nan("");
        method = "bootstrap-diff";
    }
    Comparison cmp;
    cmp.nameA = nameA;
    cmp.nameB = nameB;
    cmp.estimate = est;
    cmp.ci = ci;
    cmp.pValue = p;
    cmp.method = method;
    cmp.paired = false;
    cmp.verdict = verdictOf(est, ci, p, alpha);
    cmp.additionalNeeded = additionalUnpaired(estA, estB, static_cast<int>(runsA.size()),
                                               static_cast<int>(runsB.size()), alpha, binary);
    return cmp;
}

std::vector<HolmEntry> holm(std::vector<HolmP> entries, double alpha) {
    std::vector<HolmP> items = entries;
    std::stable_sort(items.begin(), items.end(),
                     [](const HolmP& a, const HolmP& b) { return a.p < b.p; });
    std::vector<HolmEntry> out;
    out.reserve(items.size());
    double running = 0.0;
    size_t m = items.size();
    for (size_t i = 0; i < m; i++) {
        double v = items[i].p * static_cast<double>(m - i);
        if (v > running) {
            running = v;
        }
        double adj = std::min(1.0, running);
        HolmEntry e;
        e.name = items[i].name;
        e.adjusted = adj;
        e.reject = adj < alpha;
        out.push_back(e);
    }
    return out;
}

std::string DecisionReport::str() const {
    std::vector<std::string> lines{"ranking (estimate / Holm-adjusted elimination):"};
    for (const auto& e : ranking) {
        lines.push_back("  " + fmtPadLeft(e.name, 24) + " " + fmtPlusF(e.estimate, 4) + "  " +
                        e.action);
    }
    for (const auto& c : comparisons) {
        lines.push_back("  " + c.str());
    }
    std::string out;
    for (size_t i = 0; i < lines.size(); i++) {
        if (i > 0) {
            out += "\n";
        }
        out += lines[i];
    }
    return out;
}

DecisionReport decide(const std::vector<Candidate>& cands, double alpha) {
    if (cands.size() < 2) {
        throw std::runtime_error("decide needs at least two candidates");
    }
    std::vector<RankingEntry> estimates;
    estimates.reserve(cands.size());
    std::map<std::string, const std::vector<Run>*> runsByName;
    for (const auto& c : cands) {
        std::vector<double> outs;
        outs.reserve(c.runs.size());
        for (const Run& r : c.runs) {
            outs.push_back(r.outcome);
        }
        RankingEntry e;
        e.name = c.name;
        e.estimate = exactMean(outs);
        estimates.push_back(e);
        runsByName[c.name] = &c.runs;
    }
    std::vector<RankingEntry> order = estimates;
    std::stable_sort(order.begin(), order.end(), [](const RankingEntry& a, const RankingEntry& b) {
        return b.estimate < a.estimate;
    });
    const std::string& best = order.front().name;
    std::vector<Comparison> comparisons;
    std::vector<HolmP> pvals;
    for (size_t i = 1; i < order.size(); i++) {
        Comparison comp = compare(*runsByName[best], *runsByName[order[i].name], best,
                                 order[i].name, alpha);
        comparisons.push_back(comp);
        double p = comp.pValue;
        if (std::isnan(p)) {
            p = 1.0;
        }
        HolmP hp;
        hp.name = order[i].name;
        hp.p = p;
        pvals.push_back(hp);
    }
    std::vector<HolmEntry> adjusted;
    if (!pvals.empty()) {
        adjusted = holm(pvals, alpha);
    }
    std::map<std::string, HolmEntry> adjustedByName;
    for (const auto& h : adjusted) {
        adjustedByName[h.name] = h;
    }
    std::vector<RankingEntry> ranking;
    RankingEntry base;
    base.name = best;
    base.estimate = order.front().estimate;
    base.action = "KEEP (baseline)";
    ranking.push_back(base);
    for (size_t i = 1; i < order.size(); i++) {
        RankingEntry e;
        e.name = order[i].name;
        e.estimate = order[i].estimate;
        auto it = adjustedByName.find(order[i].name);
        if (it != adjustedByName.end() && it->second.reject) {
            e.action = "ELIMINATE (worse)";
        } else {
            e.action = "KEEP (not separable)";
        }
        ranking.push_back(e);
    }
    DecisionReport rep;
    rep.estimates = estimates;
    rep.adjusted = adjusted;
    rep.comparisons = comparisons;
    rep.ranking = ranking;
    return rep;
}

}  // namespace evalsig

// variance.cpp — see variance.h. Port of variance.go.

#include "variance.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "pynum.h"

namespace evalsig {

std::string VarianceComponents::summary() const {
    return "between=" + pyFormatG(between, 4) + " within=" + pyFormatG(within, 4) +
           " icc=" + pyFormatF(icc, 3) + " (groups=" + std::to_string(nGroups) +
           ", obs=" + std::to_string(nObs) + ")";
}

namespace {

double mWithin(const std::vector<const Group*>& groups, int N, int a) {
    if (N <= a) {
        return 0.0;
    }
    double ssw = 0.0;
    for (const Group* g : groups) {
        double m = exactMean(g->vals);
        double inner = 0.0;
        for (double v : g->vals) {
            double d = v - m;
            inner += d * d;
        }
        ssw += inner;
    }
    return ssw / static_cast<double>(N - a);
}

}  // namespace

VarianceComponents varianceComponentsOf(std::vector<const Group*> groups) {
    std::vector<const Group*> cleaned;
    cleaned.reserve(groups.size());
    for (const Group* g : groups) {
        if (!g->vals.empty()) {
            cleaned.push_back(g);
        }
    }
    if (cleaned.empty()) {
        throw std::runtime_error("no non-empty groups");
    }
    std::vector<int> sizes;
    sizes.reserve(cleaned.size());
    for (const Group* g : cleaned) {
        sizes.push_back(static_cast<int>(g->vals.size()));
    }
    int a = static_cast<int>(cleaned.size());
    long long N = 0;
    for (int s : sizes) {
        N += s;
    }
    VarianceComponents vc;
    vc.nGroups = a;
    vc.nObs = static_cast<int>(N);
    vc.groupSizes = sizes;
    if (a < 2) {
        // single group: all variance is within, none between
        double msw = mWithin(cleaned, static_cast<int>(N), a);
        vc.within = msw;
        vc.total = msw;
        return vc;
    }
    std::vector<double> flat;
    flat.reserve(static_cast<size_t>(N));
    for (const Group* g : cleaned) {
        flat.insert(flat.end(), g->vals.begin(), g->vals.end());
    }
    double grand = exactMean(flat);
    double ssb = 0.0;
    for (const Group* g : cleaned) {
        double m = exactMean(g->vals);
        double d = m - grand;
        double sq = d * d;
        ssb += static_cast<double>(g->vals.size()) * sq;
    }
    double msb = ssb / static_cast<double>(a - 1);
    double msw = mWithin(cleaned, static_cast<int>(N), a);
    long long sumSq = 0;
    for (int s : sizes) {
        sumSq += static_cast<long long>(s) * s;
    }
    double n0 = (static_cast<double>(N) - static_cast<double>(sumSq) /
                                             static_cast<double>(N)) /
                 static_cast<double>(a - 1);
    if (n0 <= 0) {
        n0 = 1.0;
    }
    double between = (msb - msw) / n0;
    if (between < 0) {
        between = 0.0;
    }
    double within = msw;
    if (within < 0) {
        within = 0.0;
    }
    double total = between + within;
    double icc = 0.0;
    if (total > 0) {
        icc = between / total;
    }
    vc.between = between;
    vc.within = within;
    vc.total = total;
    vc.icc = icc;
    return vc;
}

Attribution varianceAttribution(const std::vector<Run>& runs, const std::string& factor) {
    std::vector<std::shared_ptr<Group>> groups;
    std::map<std::string, Group*> index;
    for (const Run& r : runs) {
        const PyValue* v = r.factorValue(factor);
        if (!v) {
            continue;  // Python: key is None -> skip
        }
        std::string key = pyStr(*v);
        auto it = index.find(key);
        if (it == index.end()) {
            auto g = std::make_shared<Group>();
            g->key = key;
            index[key] = g.get();
            groups.push_back(std::move(g));
            it = index.find(key);
        }
        it->second->vals.push_back(r.outcome);
    }
    std::vector<const Group*> ptrs;
    ptrs.reserve(groups.size());
    for (const auto& g : groups) ptrs.push_back(g.get());
    Attribution at;
    at.factor = factor;
    at.components = varianceComponentsOf(ptrs);
    for (const auto& g : groups) {
        GroupMean gm;
        gm.key = g->key;
        gm.mean = exactMean(g->vals);
        at.groupMeans.push_back(gm);
    }
    std::stable_sort(at.groupMeans.begin(), at.groupMeans.end(),
                     [](const GroupMean& x, const GroupMean& y) { return x.key < y.key; });
    return at;
}

}  // namespace evalsig

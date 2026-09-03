// sequential.cpp — see sequential.h. Port of sequential.go.

#include "sequential.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "pynum.h"

namespace evalsig {

std::vector<int> LookSchedule::ns() const {
    std::vector<int> out(static_cast<size_t>(maxLooks));
    for (int k = 0; k < maxLooks; k++) {
        out[static_cast<size_t>(k)] = n0 * (1 << k);
    }
    return out;
}

int LookSchedule::lookFor(int n) const {
    int last = -1;
    std::vector<int> ts = ns();
    for (size_t i = 0; i < ts.size(); i++) {
        if (ts[i] <= n) {
            last = static_cast<int>(i);
        } else {
            break;
        }
    }
    return last;
}

std::string SequentialProportion::update(int successes, int n) {
    if (n < this->n || successes < 0 || successes > n) {
        throw std::runtime_error("non-monotone update");
    }
    if (n < 1) {
        throw std::runtime_error("n must be >= 1, got " + std::to_string(n));
    }
    this->successes = successes;
    this->n = n;
    if (verdict != "CONTINUE") {
        return verdict;
    }
    int k = schedule.maxLooks;
    double alphaLook = alpha / static_cast<double>(k);
    double pHat = static_cast<double>(successes) / static_cast<double>(n);
    // width exit: plain-alpha Wilson width, any n (no error-rate cost)
    if (wilsonInterval(successes, n, alpha).half() <= targetHalfWidth) {
        verdict = "PRECISION-REACHED";
        return verdict;
    }
    int look = schedule.lookFor(n);
    if (look >= 0 && look > looksUsed - 1) {
        looksUsed = look + 1;
        // z-test vs reference at Bonferroni-corrected alpha, scheduled looks only
        if (static_cast<double>(n) * pHat * (1 - pHat) > 0) {
            double se = std::sqrt(pHat * (1 - pHat) / static_cast<double>(n));
            double z = (pHat - reference) / se;
            double pVal = 2 * (1 - normCDF(std::fabs(z)));
            if (pVal < alphaLook) {
                verdict = "CONFIRMED";
                return verdict;
            }
        }
    }
    if (n >= schedule.nMax()) {
        verdict = "BUDGET-EXHAUSTED";
    }
    return verdict;
}

Interval SequentialProportion::ci() const {
    return wilsonInterval(successes, n, alpha);
}

PyValue SequentialProportion::report() const {
    double p = std::nan("");
    if (n != 0) {
        p = static_cast<double>(successes) / static_cast<double>(n);
    }
    auto o = std::make_shared<PyObj>();
    o->set("p_hat", p)
        .set("n", static_cast<long long>(n))
        .set("verdict", verdict)
        .set("ci", ci().str())
        .set("looks_used", static_cast<long long>(looksUsed))
        .set("test_alpha_per_look", alpha / static_cast<double>(schedule.maxLooks));
    return o;
}

std::string SequentialAB::update(int s1, int n1, int s2, int n2) {
    const int ss[2] = {s1, s2};
    const int nn[2] = {n1, n2};
    for (int i = 0; i < 2; i++) {
        if (ss[i] < 0 || ss[i] > nn[i]) {
            throw std::runtime_error("successes out of range");
        }
    }
    this->s1 = s1;
    this->n1 = n1;
    this->s2 = s2;
    this->n2 = n2;
    if (verdict != "CONTINUE") {
        return verdict;
    }
    int k = schedule.maxLooks;
    double alphaLook = alpha / static_cast<double>(k);
    if (n1 == n2 && n1 > 0) {
        double p1 = static_cast<double>(s1) / static_cast<double>(n1);
        double p2 = static_cast<double>(s2) / static_cast<double>(n2);
        double half = zTwoSided(alpha) * std::sqrt(p1 * (1 - p1) / static_cast<double>(n1) +
                                                   p2 * (1 - p2) / static_cast<double>(n2));
        if (half <= targetHalfWidth) {
            verdict = "PRECISION-REACHED";
            return verdict;
        }
    }
    int mn = n1;
    if (n2 < mn) {
        mn = n2;
    }
    int look = schedule.lookFor(mn);
    if (look >= 0 && look > looksUsed - 1 && mn > 0) {
        looksUsed = look + 1;
        double p1 = static_cast<double>(s1) / static_cast<double>(n1);
        double p2 = static_cast<double>(s2) / static_cast<double>(n2);
        double pooled = static_cast<double>(s1 + s2) / static_cast<double>(n1 + n2);
        double se = std::sqrt(pooled * (1 - pooled) *
                             (1 / static_cast<double>(n1) + 1 / static_cast<double>(n2)));
        if (se > 0) {
            double z = (p1 - p2) / se;
            double pVal = 2 * (1 - normCDF(std::fabs(z)));
            if (pVal < alphaLook) {
                verdict = "CONFIRMED";
                return verdict;
            }
        }
    }
    if (mn >= schedule.nMax()) {
        verdict = "BUDGET-EXHAUSTED";
    }
    return verdict;
}

PyValue SequentialAB::report() const {
    double p1 = std::nan("");
    if (n1 != 0) {
        p1 = static_cast<double>(s1) / static_cast<double>(n1);
    }
    double p2 = std::nan("");
    if (n2 != 0) {
        p2 = static_cast<double>(s2) / static_cast<double>(n2);
    }
    auto o = std::make_shared<PyObj>();
    o->set("p1", p1)
        .set("p2", p2)
        .set("diff", p1 - p2)
        .set("n1", static_cast<long long>(n1))
        .set("n2", static_cast<long long>(n2))
        .set("verdict", verdict)
        .set("looks_used", static_cast<long long>(looksUsed))
        .set("test_alpha_per_look", alpha / static_cast<double>(schedule.maxLooks));
    return o;
}

}  // namespace evalsig

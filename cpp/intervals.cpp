// intervals.cpp — port of intervals.go.

#include "intervals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "pynum.h"
#include <utility>

namespace evalsig {

std::string Interval::str() const {
    return "[" + fmtPlusF(low, 4) + ", " + fmtPlusF(high, 4) + "] (" + method + ", " +
           pyPercent0(1 - alpha) + ")";
}

double zTwoSided(double alpha) {
    if (!(alpha > 0 && alpha < 1)) {
        throw std::runtime_error("alpha must be in (0, 1), got " + pyReprFloat(alpha));
    }
    return normInvCDF(1 - alpha / 2);
}

Interval wilsonInterval(int successes, int n, double alpha) {
    if (n < 1) {
        throw std::runtime_error("n must be >= 1, got " + std::to_string(n));
    }
    if (successes < 0 || successes > n) {
        throw std::runtime_error("successes must be in [0, n], got " +
                                 std::to_string(successes) + "/" + std::to_string(n));
    }
    double z = zTwoSided(alpha);
    double z2 = z * z;
    double p = static_cast<double>(successes) / static_cast<double>(n);
    double denom = 1 + z2 / static_cast<double>(n);
    double center = (p + z2 / (2 * static_cast<double>(n))) / denom;
    double half = z * std::sqrt(p * (1 - p) / static_cast<double>(n) +
                                 z2 / (4 * static_cast<double>(n) * static_cast<double>(n))) /
                  denom;
    Interval iv;
    iv.low = std::max(0.0, center - half);
    iv.high = std::min(1.0, center + half);
    iv.alpha = alpha;
    iv.method = "wilson";
    return iv;
}

Interval wilsonIntervalRoots(int successes, int n, double alpha) {
    if (n < 1) {
        throw std::runtime_error("n must be >= 1, got " + std::to_string(n));
    }
    if (successes < 0 || successes > n) {
        throw std::runtime_error("successes must be in [0, n], got " +
                                 std::to_string(successes) + "/" + std::to_string(n));
    }
    double z = zTwoSided(alpha);
    double z2 = z * z;
    double p = static_cast<double>(successes) / static_cast<double>(n);
    double a = static_cast<double>(n) + z2;
    double b = -(2 * static_cast<double>(n) * p + z2);
    double c = static_cast<double>(n) * p * p;
    double disc = b * b - 4 * a * c;
    if (disc < 0) {  // cannot happen; guard for float safety
        disc = 0.0;
    }
    double r = std::sqrt(disc);
    Interval iv;
    iv.low = std::max(0.0, (-b - r) / (2 * a));
    iv.high = std::min(1.0, (-b + r) / (2 * a));
    iv.alpha = alpha;
    iv.method = "wilson";
    return iv;
}

Interval newcombeDiffInterval(double p1, int n1, double p2, int n2, double alpha) {
    const double ps[2] = {p1, p2};
    const int ns[2] = {n1, n2};
    for (int i = 0; i < 2; i++) {
        if (!(ps[i] >= 0 && ps[i] <= 1)) {
            throw std::runtime_error("proportion out of [0,1]: " + pyReprFloat(ps[i]));
        }
        if (ns[i] < 1) {
            throw std::runtime_error("n must be >= 1, got " + std::to_string(ns[i]));
        }
    }
    int s1 = static_cast<int>(pyRound(p1 * static_cast<double>(n1)));
    int s2 = static_cast<int>(pyRound(p2 * static_cast<double>(n2)));
    Interval w1 = wilsonInterval(s1, n1, alpha);
    Interval w2 = wilsonInterval(s2, n2, alpha);
    double d = p1 - p2;
    double lower = d - std::sqrt((p1 - w1.low) * (p1 - w1.low) +
                                 (w2.high - p2) * (w2.high - p2));
    double upper = d + std::sqrt((w1.high - p1) * (w1.high - p1) +
                                 (p2 - w2.low) * (p2 - w2.low));
    Interval iv;
    iv.low = std::max(-1.0, lower);
    iv.high = std::min(1.0, upper);
    iv.alpha = alpha;
    iv.method = "newcombe";
    return iv;
}

TwoPropResult twoProportionTest(int s1, int n1, int s2, int n2, double alpha) {
    const int ss[2] = {s1, s2};
    const int ns[2] = {n1, n2};
    for (int i = 0; i < 2; i++) {
        if (ss[i] < 0 || ss[i] > ns[i]) {
            throw std::runtime_error("successes must be in [0, n], got " +
                                     std::to_string(ss[i]) + "/" + std::to_string(ns[i]));
        }
        if (ns[i] < 1) {
            throw std::runtime_error("n must be >= 1, got " + std::to_string(ns[i]));
        }
    }
    double p1 = static_cast<double>(s1) / static_cast<double>(n1);
    double p2 = static_cast<double>(s2) / static_cast<double>(n2);
    double pooled = static_cast<double>(s1 + s2) / static_cast<double>(n1 + n2);
    double se = std::sqrt(pooled * (1 - pooled) *
                          (1 / static_cast<double>(n1) + 1 / static_cast<double>(n2)));
    double z;
    if (se == 0) {
        if (p1 == p2) {
            z = 0.0;
        } else if (p1 - p2 < 0) {
            z = -std::numeric_limits<double>::infinity();
        } else {
            z = std::numeric_limits<double>::infinity();
        }
    } else {
        z = (p1 - p2) / se;
    }
    double pValue = 2 * (1 - normCDF(std::fabs(z)));
    TwoPropResult r;
    r.diff = p1 - p2;
    r.ci = newcombeDiffInterval(p1, n1, p2, n2, alpha);
    r.z = z;
    r.pValue = std::min(1.0, pValue);
    return r;
}

namespace {

const Rat& ratHalf() {
    static const Rat r = Rat{BigInt::fromI64(1), BigInt::fromI64(2)};
    return r;
}
const Rat& ratOne() {
    static const Rat r = Rat{BigInt::fromI64(1), BigInt::fromI64(1)};
    return r;
}
// 1 - 1/10^12: the exact rational bound of Python's clamp.
const Rat& ratOneMinusEps() {
    static const Rat r = Rat{BigInt::fromI64(999999999999LL), BigInt::fromI64(1000000000000LL)};
    return r;
}

BigInt powU64(const BigInt& b, uint64_t e) {
    BigInt result = BigInt::fromI64(1);
    BigInt base = b;
    while (e > 0) {
        if (e & 1) result = result * base;
        e >>= 1;
        if (e > 0) base = base * base;
    }
    return result;
}

}  // namespace

double binomCDF(int k, int n, double p) {
    if (k < 0 || k > n) {
        throw std::runtime_error("k must be in [0, n], got " + std::to_string(k) + "/" +
                                 std::to_string(n));
    }
    if (p <= 0) {
        return 1.0;
    }
    if (p >= 1) {
        if (k < n) {
            return 0.0;
        }
        return 1.0;
    }
    Rat r;
    if (Rat::fromF64(p).cmp(ratHalf()) == 0) {
        BigInt s = BigInt::fromI64(1);
        BigInt c = BigInt::fromI64(1);
        for (int i = 1; i <= k; i++) {
            c = c * BigInt::fromI64(n - i + 1);
            BigInt q, rem;
            BigInt::divmod(c, BigInt::fromI64(i), q, rem);
            c = q;
            s = s + c;
        }
        r = Rat{s, BigInt::pow2(n)};
    } else {
        // p = m_p / 2^e (dyadic, den a power of two), so q = 1 - p = m_q / 2^e
        // over the SAME denominator. Every pmf term C(n,i)·p^i·q^(n-i) is then
        // C(n,i)·m_p^i·m_q^(n-i) / 2^(e·n): the whole CDF is one exact integer
        // over a common power-of-two denominator. Go's big.Rat stays reduced at
        // every step (that is what bounds its sizes); this computes the same
        // real number as a single exact fraction, so the one correctly-rounded
        // conversion at the end is bit-identical.
        Rat pf = Rat::fromF64(p);
        BigInt mp = pf.num;               // p in (0,1): 0 < mp < den
        BigInt mq = pf.den - pf.num;      // 1 - p, positive
        BigInt T = powU64(mq, static_cast<uint64_t>(n));  // pmf_0 · D
        BigInt S = T;
        for (int i = 1; i <= k; i++) {
            // T = T · (n-i+1) · m_p / (i · m_q); exact integer division.
            BigInt num = T * BigInt::fromI64(n - i + 1) * mp;
            BigInt diq = BigInt::fromI64(i) * mq;
            BigInt qv, rem;
            BigInt::divmod(num, diq, qv, rem);
            T = qv;
            S = S + T;
        }
        BigInt D = powU64(pf.den, static_cast<uint64_t>(n));  // 2^(e·n)
        r.num = std::move(S);
        r.den = std::move(D);
    }
    if (r.cmp(ratOneMinusEps()) > 0) {
        return 1.0;
    }
    if (r.cmp(ratOne()) > 0) {
        return 1.0;
    }
    return r.toF64();
}

McNemarResult mcnemarExact(int b, int c, double alpha) {
    if (b < 0 || c < 0) {
        throw std::runtime_error("b, c must be >= 0, got " + std::to_string(b) + ", " +
                                 std::to_string(c));
    }
    int n = b + c;
    McNemarResult m;
    m.b = b;
    m.c = c;
    if (n == 0) {
        m.diff = 0.0;
        m.ci = Interval(0.0, 0.0, alpha, "mcnemar-degenerate");
        m.pValue = 1.0;
        return m;
    }
    int x = b;
    if (c < x) {
        x = c;
    }
    double pValue = std::min(1.0, 2 * binomCDF(x, n, 0.5));
    double d = static_cast<double>(b - c) / static_cast<double>(n);
    double bc = static_cast<double>(b - c);
    double varD = (static_cast<double>(n) - bc * bc / static_cast<double>(n)) /
                  (static_cast<double>(n) * static_cast<double>(n));
    if (varD < 0) {
        varD = 0.0;
    }
    double half = zTwoSided(alpha) * std::sqrt(varD);
    m.diff = d;
    m.ci = Interval(d - half, d + half, alpha, "mcnemar-wald");
    m.pValue = pValue;
    return m;
}

namespace {

Interval percentileInterval(std::vector<double> stats, int B, double alpha,
                            std::string method) {
    std::sort(stats.begin(), stats.end());
    int loIdx = static_cast<int>(std::floor(alpha / 2 * static_cast<double>(B))) - 1;
    if (loIdx < 0) {
        loIdx = 0;
    }
    int hiIdx = static_cast<int>(std::ceil((1 - alpha / 2) * static_cast<double>(B))) - 1;
    if (hiIdx > B - 1) {
        hiIdx = B - 1;
    }
    return Interval(stats[static_cast<size_t>(loIdx)], stats[static_cast<size_t>(hiIdx)],
                    alpha, std::move(method));
}

// Exact per-cluster sums as big integers on a common power-of-two scale.
struct ClusterSums {
    BigInt scale;
    std::vector<BigInt> sums;
    std::vector<int> counts;
    bool ok = false;
};

ClusterSums exactClusterSums(const std::vector<std::vector<double>>& clusters) {
    ClusterSums out;
    std::vector<double> flat;
    for (const auto& cl : clusters) {
        flat.insert(flat.end(), cl.begin(), cl.end());
    }
    ScaledInts si = exactScaledInts(flat);
    if (!si.ok) {
        return out;
    }
    out.scale = si.scale;
    size_t pos = 0;
    for (const auto& cl : clusters) {
        BigInt s;
        for (size_t i = 0; i < cl.size(); i++) {
            s = s + si.ints[pos + i];
        }
        out.sums.push_back(std::move(s));
        out.counts.push_back(static_cast<int>(cl.size()));
        pos += cl.size();
    }
    out.ok = true;
    return out;
}

}  // namespace

Interval clusterBootstrapCI(const std::vector<std::vector<double>>& clusters,
                            const std::function<double(const std::vector<double>&)>& stat,
                            int B, double alpha, int seed) {
    if (B < 2) {
        throw std::runtime_error("B must be >= 2, got " + std::to_string(B));
    }
    if (clusters.empty()) {
        throw std::runtime_error("no clusters given");
    }
    PyRandom rng;
    rng.seedInt(static_cast<int64_t>(seed));
    int k = static_cast<int>(clusters.size());
    if (!stat) {
        ClusterSums cs = exactClusterSums(clusters);
        if (cs.ok) {
            std::vector<double> stats;
            stats.reserve(static_cast<size_t>(B));
            for (int i = 0; i < B; i++) {
                BigInt total;
                int cnt = 0;
                for (int j = 0; j < k; j++) {
                    int idx = rng.randrange(k);
                    total = total + cs.sums[static_cast<size_t>(idx)];
                    cnt += cs.counts[static_cast<size_t>(idx)];
                }
                if (cnt == 0) {
                    throw std::runtime_error("mean requires at least one data point");
                }
                BigInt denom = cs.scale * BigInt::fromI64(cnt);
                stats.push_back(Rat{total, denom}.toF64());
            }
            return percentileInterval(std::move(stats), B, alpha, "cluster-bootstrap");
        }
        // fall through to the generic exactMean path
        std::vector<double> stats;
        stats.reserve(static_cast<size_t>(B));
        for (int i = 0; i < B; i++) {
            std::vector<const std::vector<double>*> sample;
            for (int j = 0; j < k; j++) {
                sample.push_back(&clusters[static_cast<size_t>(rng.randrange(k))]);
            }
            std::vector<double> flat;
            for (const auto* cl : sample) {
                flat.insert(flat.end(), cl->begin(), cl->end());
            }
            stats.push_back(exactMean(flat));
        }
        return percentileInterval(std::move(stats), B, alpha, "cluster-bootstrap");
    }
    std::vector<double> stats;
    stats.reserve(static_cast<size_t>(B));
    for (int i = 0; i < B; i++) {
        std::vector<const std::vector<double>*> sample;
        for (int j = 0; j < k; j++) {
            sample.push_back(&clusters[static_cast<size_t>(rng.randrange(k))]);
        }
        std::vector<double> flat;
        for (const auto* cl : sample) {
            flat.insert(flat.end(), cl->begin(), cl->end());
        }
        stats.push_back(stat(flat));
    }
    return percentileInterval(std::move(stats), B, alpha, "cluster-bootstrap");
}

}  // namespace evalsig

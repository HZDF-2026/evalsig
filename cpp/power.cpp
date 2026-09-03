// power.cpp — see power.h. Port of power.go.

#include "power.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "intervals.h"
#include "pynum.h"

namespace evalsig {

namespace {

void validatePower(double p1, double p2, double alpha, double power) {
    const double ps[2] = {p1, p2};
    for (double p : ps) {
        if (!(p >= 0 && p <= 1)) {
            throw std::runtime_error("proportion out of [0,1]: " + pyReprFloat(p));
        }
    }
    if (!(alpha > 0 && alpha < 1)) {
        throw std::runtime_error("alpha must be in (0,1), got " + pyReprFloat(alpha));
    }
    if (!(power > 0 && power < 1)) {
        throw std::runtime_error("power must be in (0,1), got " + pyReprFloat(power));
    }
}

}  // namespace

PowerPlan nTwoProportions(double p1, double p2, double alpha, double power) {
    validatePower(p1, p2, alpha, power);
    double delta = std::fabs(p1 - p2);
    if (delta == 0) {
        throw std::runtime_error("p1 == p2: no effect to detect, n is undefined");
    }
    double zA = zTwoSided(alpha);
    double zB = normInvCDF(power);
    double pbar = (p1 + p2) / 2;
    double num = zA * std::sqrt(2 * pbar * (1 - pbar)) +
                 zB * std::sqrt(p1 * (1 - p1) + p2 * (1 - p2));
    num = num * num;
    long long n = static_cast<long long>(std::ceil(num / (delta * delta)));
    if (n < 1) {
        n = 1;
    }
    PowerPlan plan;
    plan.nPerVersion = static_cast<int>(n);
    plan.alpha = alpha;
    plan.power = power;
    plan.method = "two-proportion z";
    return plan;
}

PowerPlan nPaired(double expectedDiscordance, double delta, double alpha, double power) {
    validatePower(0.5, 0.5, alpha, power);
    if (!(expectedDiscordance > 0 && expectedDiscordance <= 1)) {
        throw std::runtime_error("discordance must be in (0, 1], got " +
                                 pyReprFloat(expectedDiscordance));
    }
    if (delta <= 0) {
        throw std::runtime_error("delta must be > 0");
    }
    double zA = zTwoSided(alpha);
    double zB = normInvCDF(power);
    long long n = static_cast<long long>(
        std::ceil((zA + zB) * (zA + zB) * expectedDiscordance / (delta * delta)));
    if (n < 1) {
        n = 1;
    }
    PowerPlan plan;
    plan.nPerVersion = static_cast<int>(n);
    plan.alpha = alpha;
    plan.power = power;
    plan.method = "mcnemar";
    plan.note = "paired on the same task set; multiply-unpaired designs need far more";
    return plan;
}

PowerPlan nForCIWidth(double width, double p, double alpha) {
    if (!(width > 0 && width < 1)) {
        throw std::runtime_error("width must be in (0, 1), got " + pyReprFloat(width));
    }
    if (!(p >= 0 && p <= 1)) {
        throw std::runtime_error("p must be in [0, 1], got " + pyReprFloat(p));
    }
    double z = zTwoSided(alpha);
    double half = width / 2;
    long long n = static_cast<long long>(std::ceil(z * z * p * (1 - p) / (half * half)));
    if (n < 1) {
        n = 1;
    }
    PowerPlan plan;
    plan.nPerVersion = static_cast<int>(n);
    plan.alpha = alpha;
    plan.power = 0.0;
    plan.method = "ci-width";
    return plan;
}

}  // namespace evalsig

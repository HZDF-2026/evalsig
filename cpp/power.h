// power.h — power analysis: how many reruns before a decision means anything.
// Port of power.go.
#pragma once

#include <string>

namespace evalsig {

// Sample-size plan with the design it was computed for.
struct PowerPlan {
    int nPerVersion = 0;
    double alpha = 0.0;
    double power = 0.0;
    std::string method;
    std::string note;
};

// Minimum n per version to detect p1 vs p2 (unpaired z-test). Normal
// approximation with pooled-variance form:
// n = (z_{a/2} sqrt(2 p_bar q_bar) + z_b sqrt(p1 q1 + p2 q2))^2 / delta^2.
PowerPlan nTwoProportions(double p1, double p2, double alpha, double power);

// Minimum total N (paired, fixed task set) for McNemar. delta = |p1 - p2| is
// the win-rate difference; expectedDiscordance is the fraction of tasks
// expected to flip between versions (b + c)/N — from a pilot run, or 0.3 as
// a common first guess for agent benchmarks.
// N = (z_{a/2} + z_b)^2 * psi / delta^2 with psi = discordance.
PowerPlan nPaired(double expectedDiscordance, double delta, double alpha, double power);

// Minimum n for a CI half-width <= width/2 at success rate p. Wald-based
// planning formula n = z^2 p(1-p) / (width/2)^2; conservative because it
// ignores Wilson's shrinkage, which is the right direction for planning.
PowerPlan nForCIWidth(double width, double p, double alpha);

}  // namespace evalsig

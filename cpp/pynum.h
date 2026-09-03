// pynum.h — bit-exact ports of the CPython numeric semantics evalsig leans
// on: exact rational arithmetic (statistics.mean / statistics.stdev), MT19937
// (random.Random seeding + getrandbits + _randbelow), AS 241 normal quantile
// (NormalDist.inv_cdf), fdlibm log, erf, round(), and the float repr / '.Ng' /
// '.Nf' / '.0%' format specs.
//
// C++17, no external dependencies. The BigInt/Rat core reproduces Go's
// math/big semantics that the Go port itself uses to match CPython: exact
// rational sums, one correctly-rounded division to double (quotToFloat64
// algorithm: round-half-even).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace evalsig {

// ---------------------------------------------------------------------------
// BigInt: signed arbitrary-precision integer, 32-bit little-endian limbs.
// Only the operations the Rat layer needs are provided.
// ---------------------------------------------------------------------------

class BigInt {
public:
    BigInt() = default;
    static BigInt fromU64(uint64_t v);
    static BigInt fromI64(int64_t v);

    bool isZero() const { return mag.empty(); }
    bool negative() const { return neg && !mag.empty(); }
    void setZero() { mag.clear(); neg = false; }

    int bitLen() const;               // number of significant bits, 0 for zero
    uint64_t low64() const;           // value mod 2^64
    int cmp(const BigInt& o) const;   // -1, 0, 1 (signed compare)
    static int cmpMag(const BigInt& a, const BigInt& b);

    BigInt operator+(const BigInt& o) const;
    BigInt operator-(const BigInt& o) const;
    BigInt operator*(const BigInt& o) const;
    BigInt operator<<(int n) const;  // n >= 0
    BigInt operator>>(int n) const;  // n >= 0, floor semantics

    // Floor division and modulus on non-negative operands.
    static void divmod(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r);

    // Exact power-of-two scaling used by the dyadic rational fast paths.
    static BigInt pow2(int e) { return BigInt::fromU64(1) << e; }

    std::string toString() const;    // decimal, for debugging

private:
    bool neg = false;
    std::vector<uint32_t> mag;  // little-endian, no leading zero limbs

    void trim();
    static void divmodMag(const std::vector<uint32_t>& u,
                           const std::vector<uint32_t>& v,
                           std::vector<uint32_t>& q, std::vector<uint32_t>& r);
};

// ---------------------------------------------------------------------------
// Rat: exact rational, numerator signed / denominator positive (Go big.Rat
// semantics; normalization/gcd is not needed for numerical results).
// ---------------------------------------------------------------------------

struct Rat {
    BigInt num;
    BigInt den{BigInt::fromU64(1)};

    static Rat fromF64(double v);     // exact value of a finite double
    static Rat fromI64(int64_t v);

    Rat add(const Rat& o) const;
    Rat sub(const Rat& o) const;
    Rat mul(const Rat& o) const;
    Rat quo(const Rat& o) const;
    int cmp(const Rat& o) const;      // -1, 0, 1

    // Correctly-rounded nearest double (round-half-even), Go's
    // big.Rat.Float64 / CPython long_true_divide algorithm.
    double toF64() const;
};

// ---------------------------------------------------------------------------
// MT19937 — CPython random.Random for the ops evalsig uses.
// ---------------------------------------------------------------------------

class PyRandom {
public:
    void seedInt(int64_t seed);       // random.Random(n).seed(n)
    uint32_t genrandUint32();
    uint64_t getrandbits(int k);     // 1 <= k <= 64
    uint64_t randbelow(uint64_t n);  // _randbelow(n) rejection loop
    int randrange(int stop) { return static_cast<int>(randbelow(static_cast<uint64_t>(stop))); }

private:
    uint32_t mt[624];
    int index = 624;  // zero-value pyRandom equivalent is never used unseeded

    void initGenrand(uint32_t s);
    void initByArray(const uint32_t* key, size_t keyLen);
};

// ---------------------------------------------------------------------------
// Normal distribution: AS 241 inverse CDF (CPython statistics.NormalDist),
// fdlibm log, erf (Go math.Erf pure algorithm), CDF.
// ---------------------------------------------------------------------------

double fdlibmLog(double x);   // FreeBSD msun __ieee754_log, pure algorithm
double normInvCDF(double p);  // statistics.NormalDist().inv_cdf(p)
double goErf(double x);        // Go math.Erf, pure-Go algorithm (amd64 path)
double normCDF(double x);     // statistics.NormalDist().cdf(x)

// ---------------------------------------------------------------------------
// Python numeric semantics: round(), float repr, format specs.
// ---------------------------------------------------------------------------

int64_t pyRound(double x);                 // round(x): half-to-even to int
std::string pyReprFloat(double v);         // repr(x) / str(x)
std::string pyFormatG(double x, int prec); // format(x, '.Ng')
std::string pyFormatF(double x, int prec); // format(x, '.Nf')
std::string pyPercent0(double x);          // format(x, '.0%')

// Helper for %+.*f call sites (Go fmt "%+.4f").
std::string fmtPlusF(double x, int prec);
// Helper for Go fmt "%-24s" / "%-28s" style padded fields.
std::string fmtPadLeft(const std::string& s, size_t width);

// ---------------------------------------------------------------------------
// Exact statistics: statistics.mean / statistics.stdev for float data — exact
// rational summation, one correctly-rounded division.
// ---------------------------------------------------------------------------

// statistics.mean(data); throws std::runtime_error on empty input.
double exactMean(const std::vector<double>& xs);
// statistics.stdev(data); throws std::runtime_error when n < 2.
double exactStdev(const std::vector<double>& xs);

// Exact dyadic representation of finite floats under a common scale:
// values[i] == ints[i] / scale exactly. ok=false when any value is not finite.
struct ScaledInts {
    BigInt scale;
    std::vector<BigInt> ints;
    bool ok = false;
};
ScaledInts exactScaledInts(const std::vector<double>& values);

}  // namespace evalsig

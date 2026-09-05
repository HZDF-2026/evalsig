// pynum.cpp — see pynum.h. Every algorithm here is a line-by-line port of a
// reference implementation (Go math/big quotToFloat64, CPython _random /
// statistics, FreeBSD msun erf/log) so the produced bits match the Python and
// Go ports of evalsig.
#include "pynum.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cfloat>
#include <utility>

namespace evalsig {

// ---------------------------------------------------------------------------
// BigInt
// ---------------------------------------------------------------------------

BigInt BigInt::fromU64(uint64_t v) {
    BigInt r;
    if (v != 0) {
        r.mag.push_back(static_cast<uint32_t>(v));
        r.mag.push_back(static_cast<uint32_t>(v >> 32));
        r.trim();
    }
    return r;
}

BigInt BigInt::fromI64(int64_t v) {
    BigInt r;
    if (v != 0) {
        r.neg = v < 0;
        uint64_t u = r.neg ? static_cast<uint64_t>(-(v + 1)) + 1 : static_cast<uint64_t>(v);
        r.mag.push_back(static_cast<uint32_t>(u));
        r.mag.push_back(static_cast<uint32_t>(u >> 32));
        r.trim();
    }
    return r;
}

void BigInt::trim() {
    while (!mag.empty() && mag.back() == 0) mag.pop_back();
    if (mag.empty()) neg = false;
}

int BigInt::bitLen() const {
    if (mag.empty()) return 0;
    int top = static_cast<int>(mag.size()) - 1;
    uint32_t h = mag[top];
    int b = 0;
    while (h != 0) {
        h >>= 1;
        b++;
    }
    return top * 32 + b;
}

uint64_t BigInt::low64() const {
    uint64_t lo = mag.empty() ? 0 : mag[0];
    uint64_t hi = mag.size() < 2 ? 0 : mag[1];
    return lo | (hi << 32);
}

int BigInt::cmpMag(const BigInt& a, const BigInt& b) {
    if (a.mag.size() != b.mag.size())
        return a.mag.size() < b.mag.size() ? -1 : 1;
    for (size_t i = a.mag.size(); i-- > 0;) {
        if (a.mag[i] != b.mag[i]) return a.mag[i] < b.mag[i] ? -1 : 1;
    }
    return 0;
}

int BigInt::cmp(const BigInt& o) const {
    bool an = negative(), bn = o.negative();
    if (an != bn) return an ? -1 : 1;
    int c = cmpMag(*this, o);
    return an ? -c : c;
}

BigInt BigInt::operator+(const BigInt& o) const {
    BigInt r;
    if (negative() == o.negative()) {
        r.neg = negative();
        r.mag.resize(std::max(mag.size(), o.mag.size()) + 1, 0);
        uint64_t carry = 0;
        for (size_t i = 0; i < r.mag.size(); i++) {
            uint64_t cur = carry;
            if (i < mag.size()) cur += mag[i];
            if (i < o.mag.size()) cur += o.mag[i];
            r.mag[i] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        r.trim();
        return r;
    }
    int c = cmpMag(*this, o);
    if (c == 0) return BigInt();
    const BigInt& big = c > 0 ? *this : o;
    const BigInt& small = c > 0 ? o : *this;
    r.neg = big.negative();
    r.mag.assign(big.mag.begin(), big.mag.end());
    int64_t borrow = 0;
    for (size_t i = 0; i < r.mag.size(); i++) {
        int64_t cur = static_cast<int64_t>(r.mag[i]) - borrow -
                      (i < small.mag.size() ? small.mag[i] : 0);
        if (cur < 0) {
            cur += (1ll << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        r.mag[i] = static_cast<uint32_t>(cur);
    }
    r.trim();
    return r;
}

BigInt BigInt::operator-(const BigInt& o) const {
    BigInt mo = o;
    if (!mo.mag.empty()) mo.neg = !mo.neg;
    return *this + mo;
}

BigInt BigInt::operator*(const BigInt& o) const {
    BigInt r;
    if (mag.empty() || o.mag.empty()) return r;
    r.neg = negative() != o.negative();
    r.mag.assign(mag.size() + o.mag.size(), 0);
    for (size_t i = 0; i < mag.size(); i++) {
        uint64_t carry = 0;
        for (size_t j = 0; j < o.mag.size(); j++) {
            uint64_t cur = static_cast<uint64_t>(r.mag[i + j]) +
                           static_cast<uint64_t>(mag[i]) * o.mag[j] + carry;
            r.mag[i + j] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        size_t k = i + o.mag.size();
        while (carry != 0) {
            uint64_t cur = static_cast<uint64_t>(r.mag[k]) + carry;
            r.mag[k] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
            k++;
        }
    }
    r.trim();
    return r;
}

BigInt BigInt::operator<<(int n) const {
    if (mag.empty() || n <= 0) return *this;
    int limbShift = n / 32;
    int bitShift = n % 32;
    BigInt r;
    r.neg = negative();
    r.mag.assign(static_cast<size_t>(limbShift), 0);
    if (bitShift == 0) {
        r.mag.insert(r.mag.end(), mag.begin(), mag.end());
    } else {
        uint32_t carry = 0;
        for (uint32_t limb : mag) {
            r.mag.push_back((limb << bitShift) | carry);
            carry = limb >> (32 - bitShift);
        }
        if (carry != 0) r.mag.push_back(carry);
    }
    r.trim();
    return r;
}

BigInt BigInt::operator>>(int n) const {
    // magnitude shift; callers only use this on non-negative values
    if (mag.empty() || n <= 0) return *this;
    int limbShift = n / 32;
    int bitShift = n % 32;
    BigInt r;
    r.neg = negative();
    if (limbShift >= static_cast<int>(mag.size())) return BigInt();
    if (bitShift == 0) {
        r.mag.assign(mag.begin() + limbShift, mag.end());
    } else {
        for (size_t i = static_cast<size_t>(limbShift); i < mag.size(); i++) {
            uint32_t lo = mag[i] >> bitShift;
            uint32_t hi = (i + 1 < mag.size()) ? (mag[i + 1] << (32 - bitShift)) : 0;
            r.mag.push_back(lo | hi);
        }
    }
    r.trim();
    return r;
}

namespace {
int nlz32v(uint32_t x) {
    int n = 0;
    if (x == 0) return 32;
    while ((x & 0x80000000u) == 0) {
        x <<= 1;
        n++;
    }
    return n;
}

int cmpMagVec(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}
}  // namespace

// Knuth algorithm D (32-bit limbs, 64-bit intermediates), following the
// classic Hacker's Delight divmnu formulation.
void BigInt::divmodMag(const std::vector<uint32_t>& u,
                       const std::vector<uint32_t>& v,
                       std::vector<uint32_t>& q, std::vector<uint32_t>& r) {
    if (u.empty() || cmpMagVec(u, v) < 0) {
        q.clear();
        r = u;
        return;
    }
    int n = static_cast<int>(v.size());
    if (n == 1) {
        uint64_t rem = 0;
        q.assign(u.size(), 0);
        for (size_t j = u.size(); j-- > 0;) {
            uint64_t cur = (rem << 32) | u[j];
            q[j] = static_cast<uint32_t>(cur / v[0]);
            rem = cur % v[0];
        }
        if (rem != 0) r.assign(1, static_cast<uint32_t>(rem));
        else r.clear();
        return;
    }
    int m = static_cast<int>(u.size()) - n;
    int s = nlz32v(v[n - 1]);
    std::vector<uint32_t> vn(n), un(m + n + 1, 0);
    if (s > 0) {
        for (int i = n - 1; i > 0; i--) vn[i] = (v[i] << s) | (v[i - 1] >> (32 - s));
        vn[0] = v[0] << s;
        un[m + n] = u[m + n - 1] >> (32 - s);
        for (int i = m + n - 1; i > 0; i--) un[i] = (u[i] << s) | (u[i - 1] >> (32 - s));
        un[0] = u[0] << s;
    } else {
        std::copy(v.begin(), v.end(), vn.begin());
        std::copy(u.begin(), u.end(), un.begin());
    }
    std::vector<uint32_t> qq(m + 1, 0);
    for (int j = m; j >= 0; j--) {
        uint64_t num = (static_cast<uint64_t>(un[j + n]) << 32) | un[j + n - 1];
        uint64_t qhat = num / vn[n - 1];
        uint64_t rhat = num % vn[n - 1];
        while (qhat >= (1ull << 32) ||
               qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2])) {
            qhat--;
            rhat += vn[n - 1];
            if (rhat >= (1ull << 32)) break;
        }
        int64_t borrow = 0;
        uint64_t carry = 0;
        for (int i = 0; i < n; i++) {
            uint64_t p = qhat * vn[i] + carry;
            carry = p >> 32;
            int64_t t = static_cast<int64_t>(un[i + j]) -
                        static_cast<int64_t>(static_cast<uint32_t>(p)) - borrow;
            if (t < 0) {
                t += (1ll << 32);
                borrow = 1;
            } else {
                borrow = 0;
            }
            un[i + j] = static_cast<uint32_t>(t);
        }
        int64_t t = static_cast<int64_t>(un[j + n]) - static_cast<int64_t>(carry) - borrow;
        if (t < 0) {
            // qhat was one too large: add the divisor back
            qhat--;
            uint64_t c = 0;
            for (int i = 0; i < n; i++) {
                uint64_t s2 = static_cast<uint64_t>(un[i + j]) + vn[i] + c;
                un[i + j] = static_cast<uint32_t>(s2);
                c = s2 >> 32;
            }
            un[j + n] = static_cast<uint32_t>(static_cast<uint64_t>(un[j + n]) + c);
        } else {
            un[j + n] = static_cast<uint32_t>(t);
        }
        qq[j] = static_cast<uint32_t>(qhat);
    }
    // remainder = un[0..n) >> s
    r.clear();
    if (s > 0) {
        for (int i = 0; i < n; i++) {
            uint32_t lo = un[i] >> s;
            uint32_t hi = un[i + 1] << (32 - s);  // un has n+m+1 limbs: in range
            r.push_back(lo | hi);
        }
    } else {
        r.assign(un.begin(), un.begin() + n);
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    q.swap(qq);
    while (!q.empty() && q.back() == 0) q.pop_back();
}

void BigInt::divmod(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r) {
    std::vector<uint32_t> qm, rm;
    divmodMag(a.mag, b.mag, qm, rm);
    q.mag = std::move(qm);
    q.neg = false;
    q.trim();
    r.mag = std::move(rm);
    r.neg = false;
    r.trim();
}

std::string BigInt::toString() const {
    if (mag.empty()) return "0";
    std::vector<uint32_t> cur = mag;
    std::string out;
    while (!cur.empty()) {
        uint64_t rem = 0;
        for (size_t i = cur.size(); i-- > 0;) {
            uint64_t x = (rem << 32) | cur[i];
            cur[i] = static_cast<uint32_t>(x / 1000000000u);
            rem = x % 1000000000u;
        }
        while (!cur.empty() && cur.back() == 0) cur.pop_back();
        char buf[16];
        if (cur.empty()) std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(rem));
        else std::snprintf(buf, sizeof buf, "%09llu", static_cast<unsigned long long>(rem));
        out = buf + out;
    }
    if (negative()) out = "-" + out;
    return out;
}

// ---------------------------------------------------------------------------
// Rat
// ---------------------------------------------------------------------------

Rat Rat::fromI64(int64_t v) {
    Rat r;
    r.num = BigInt::fromI64(v);
    return r;
}

Rat Rat::fromF64(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    bool sign = (bits >> 63) != 0;
    int exp = static_cast<int>((bits >> 52) & 0x7ff);
    uint64_t man = bits & 0xfffffffffffffull;
    BigInt num;
    BigInt den = BigInt::fromU64(1);
    if (exp == 0) {
        if (man == 0) return Rat();  // +-0.0 -> exact zero
        num = BigInt::fromU64(man);
        den = BigInt::pow2(1074);
    } else {
        man |= 1ull << 52;
        int shift = exp - 1075;
        if (shift >= 0) {
            num = BigInt::fromU64(man) << shift;
        } else {
            num = BigInt::fromU64(man);
            den = BigInt::pow2(-shift);
        }
    }
    if (sign) num = BigInt() - num;
    Rat r;
    r.num = std::move(num);
    r.den = std::move(den);
    return r;
}

Rat Rat::add(const Rat& o) const {
    Rat r;
    r.num = num * o.den + o.num * den;
    r.den = den * o.den;
    return r;
}

Rat Rat::sub(const Rat& o) const {
    Rat r;
    r.num = num * o.den - o.num * den;
    r.den = den * o.den;
    return r;
}

Rat Rat::mul(const Rat& o) const {
    Rat r;
    r.num = num * o.num;
    r.den = den * o.den;
    return r;
}

Rat Rat::quo(const Rat& o) const {
    // (num/den) / (o.num/o.den) = (num * o.den) / (den * |o.num|), sign folded in.
    BigInt onumAbs = o.num.negative() ? (BigInt() - o.num) : o.num;
    Rat r;
    r.num = num * o.den;
    r.den = den * onumAbs;
    if (o.num.negative()) r.num = BigInt() - r.num;
    return r;
}

int Rat::cmp(const Rat& o) const {
    return (num * o.den).cmp(o.num * den);
}

double Rat::toF64() const {
    // Go math/big quotToFloat64 (rat.go): nearest double, round-half-even.
    const int Msize = 52, Msize1 = 53, Msize2 = 54;
    const int Ebias = 1023, Emin = 1 - Ebias;

    bool neg = num.negative();
    BigInt a = neg ? (BigInt() - num) : num;
    const BigInt& b = den;
    int alen = a.bitLen();
    if (alen == 0) return neg ? -0.0 : 0.0;
    int blen = b.bitLen();
    int exp = alen - blen;
    BigInt a2 = a, b2 = b;
    int shift = Msize2 - exp;
    if (shift > 0) {
        a2 = a2 << shift;
    } else if (shift < 0) {
        b2 = b2 << (-shift);
    }
    BigInt q, r;
    BigInt::divmod(a2, b2, q, r);
    uint64_t mantissa = q.low64();
    bool haveRem = !r.isZero();
    if ((mantissa >> Msize2) == 1) {
        if (mantissa & 1) haveRem = true;
        mantissa >>= 1;
        exp++;
    }
    if (Emin - Msize <= exp && exp <= Emin) {
        int s = Emin - (exp - 1);
        if (s > 0) {
            uint64_t lostbits = mantissa & ((1ull << s) - 1);
            haveRem = haveRem || lostbits != 0;
            mantissa >>= s;
            exp = 2 - Ebias;
        }
    }
    if (mantissa & 1) {
        if (haveRem || (mantissa & 2) != 0) {
            mantissa++;
            if (mantissa >= (1ull << Msize2)) {
                mantissa >>= 1;
                exp++;
            }
        }
    }
    mantissa >>= 1;
    double f = std::ldexp(static_cast<double>(mantissa), exp - Msize1);
    if (neg) f = -f;
    return f;
}

// ---------------------------------------------------------------------------
// MT19937 — port of internal/evalsig/pyrand.go which is itself a port of
// CPython _randommodule.c.
// ---------------------------------------------------------------------------

namespace {
constexpr int mtN = 624;
constexpr int mtM = 397;
constexpr uint32_t mtMatrixA = 0x9908b0df;
constexpr uint32_t mtUpper = 0x80000000u;
constexpr uint32_t mtLower = 0x7fffffffu;
}  // namespace

void PyRandom::initGenrand(uint32_t s) {
    mt[0] = s;
    for (int i = 1; i < mtN; i++) {
        mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    index = mtN;
}

void PyRandom::initByArray(const uint32_t* key, size_t keyLen) {
    initGenrand(19650218u);
    int i = 1, j = 0;
    size_t k = mtN > static_cast<int>(keyLen) ? mtN : keyLen;
    for (; k > 0; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
        i++;
        j++;
        if (i >= mtN) {
            mt[0] = mt[mtN - 1];
            i = 1;
        }
        if (j >= static_cast<int>(keyLen)) j = 0;
    }
    for (k = mtN - 1; k > 0; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
        i++;
        if (i >= mtN) {
            mt[0] = mt[mtN - 1];
            i = 1;
        }
    }
    mt[0] = 0x80000000u;
}

void PyRandom::seedInt(int64_t seed) {
    uint64_t u = static_cast<uint64_t>(seed);
    if (seed < 0) u = static_cast<uint64_t>(-(seed + 1)) + 1;
    uint32_t key[4];
    size_t keyLen = 0;
    if (u == 0) {
        key[0] = 0;
        keyLen = 1;
    } else {
        while (u > 0) {
            key[keyLen++] = static_cast<uint32_t>(u & 0xffffffffull);
            u >>= 32;
        }
    }
    initByArray(key, keyLen);
}

uint32_t PyRandom::genrandUint32() {
    if (index >= mtN) {
        if (index == mtN + 1) initGenrand(5489u);
        uint32_t y;
        const uint32_t mag01[2] = {0, mtMatrixA};
        int kk = 0;
        for (; kk < mtN - mtM; kk++) {
            y = (mt[kk] & mtUpper) | (mt[kk + 1] & mtLower);
            mt[kk] = mt[kk + mtM] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        for (; kk < mtN - 1; kk++) {
            y = (mt[kk] & mtUpper) | (mt[kk + 1] & mtLower);
            mt[kk] = mt[kk + (mtM - mtN)] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        y = (mt[mtN - 1] & mtUpper) | (mt[0] & mtLower);
        mt[mtN - 1] = mt[mtM - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
        index = 0;
    }
    uint32_t y = mt[index];
    index++;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

uint64_t PyRandom::getrandbits(int k) {
    if (k <= 0) throw std::runtime_error("number of bits must be greater than zero");
    if (k <= 32) return static_cast<uint64_t>(genrandUint32() >> (32 - k));
    int words = (k - 1) / 32 + 1;
    uint64_t x = 0;
    int kk = k;
    for (int i = 0; i < words; i++, kk -= 32) {
        uint32_t w = genrandUint32();
        if (kk < 32) w >>= 32 - kk;
        x |= static_cast<uint64_t>(w) << (i * 32);
    }
    return x;
}

uint64_t PyRandom::randbelow(uint64_t n) {
    if (n == 0) return 0;
    int nlz = 0;
    uint64_t x = n;
    while (nlz < 64 && (x & 0x8000000000000000ull) == 0) {
        x <<= 1;
        nlz++;
    }
    int k = 64 - nlz;  // n.bit_length()
    uint64_t v = getrandbits(k);
    while (v >= n) v = getrandbits(k);
    return v;
}

// ---------------------------------------------------------------------------
// fdlibm log — port of Go math/log.go pure path (pynorm.go).
// ---------------------------------------------------------------------------

double fdlibmLog(double x) {
    const double Ln2Hi = 6.93147180369123816490e-01;
    const double Ln2Lo = 1.90821492927058770002e-10;
    const double L1 = 6.666666666666735130e-01;
    const double L2 = 3.999999999940941908e-01;
    const double L3 = 2.857142874366239149e-01;
    const double L4 = 2.222219843214978396e-01;
    const double L5 = 1.818357216161805012e-01;
    const double L6 = 1.531383769920937332e-01;
    const double L7 = 1.479819860511658591e-01;
    if (std::isnan(x) || std::isinf(x)) return x;
    if (x < 0) return std::nan("");
    if (x == 0) return -HUGE_VAL;
    int ki;
    double f1 = std::frexp(x, &ki);
    if (f1 < std::sqrt(2.0) / 2) {
        f1 *= 2;
        ki--;
    }
    double f = f1 - 1;
    double k = static_cast<double>(ki);
    double s = f / (2 + f);
    double s2 = s * s;
    double s4 = s2 * s2;
    double t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)));
    double t2 = s4 * (L2 + s4 * (L4 + s4 * L6));
    double R = t1 + t2;
    double hfsq = 0.5 * f * f;
    return k * Ln2Hi - ((hfsq - (s * (hfsq + R) + k * Ln2Lo)) - f);
}

// ---------------------------------------------------------------------------
// AS 241 inverse normal CDF — port of pynorm.go (CPython statistics.py).
// ---------------------------------------------------------------------------

double normInvCDF(double p) {
    double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        double r = 0.180625 - q * q;
        double num = (((((((2.5090809287301226727e+3 * r +
                           3.3430575583588128105e+4) * r +
                           6.7265770927008700853e+4) * r +
                           4.5921953931549871457e+4) * r +
                           1.3731693765509461125e+4) * r +
                           1.9715909503065514427e+3) * r +
                           1.3314166789178437745e+2) * r +
                           3.3871328727963666080e+0) * q;
        double den = (((((((5.2264952788528545610e+3 * r +
                           2.8729085735721942674e+4) * r +
                           3.9307895800092710610e+4) * r +
                           2.1213794301586595867e+4) * r +
                           5.3941960214247511077e+3) * r +
                           6.8718700749205790830e+2) * r +
                           4.2313330701600911252e+1) * r +
                           1.0);
        return num / den;
    }
    double r = p;
    if (q > 0.0) r = 1.0 - p;
    r = std::sqrt(-fdlibmLog(r));
    if (r <= 5.0) {
        r -= 1.6;
        double num = (((((((7.74545014278341407640e-4 * r +
                           2.27238449892691845833e-2) * r +
                           2.41780725177450611770e-1) * r +
                           1.27045825245236838258e+0) * r +
                           3.64784832476320460504e+0) * r +
                           5.76949722146069140550e+0) * r +
                           4.63033784615654529590e+0) * r +
                           1.42343711074968357734e+0);
        double den = (((((((1.05075007164441684324e-9 * r +
                           5.47593808499534494600e-4) * r +
                           1.51986665636164571966e-2) * r +
                           1.48103976427480074590e-1) * r +
                           6.89767334985100004550e-1) * r +
                           1.67638483018380384940e+0) * r +
                           2.05319162663775882187e+0) * r +
                           1.0);
        double x = num / den;
        if (q < 0.0) x = -x;
        return x;
    }
    r -= 5.0;
    double num = (((((((2.01033439929228813265e-7 * r +
                       2.71155556874348757815e-5) * r +
                       1.24266094738807843860e-3) * r +
                       2.65321895265761230930e-2) * r +
                       2.96560571828504891230e-1) * r +
                       1.78482653991729133580e+0) * r +
                       5.46378491116411436990e+0) * r +
                       6.65790464350110377720e+0);
    double den = (((((((2.04426310338993978564e-15 * r +
                       1.42151175831644588870e-7) * r +
                       1.84631831751005468180e-5) * r +
                       7.86869131145613259100e-4) * r +
                       1.48753612908506148525e-2) * r +
                       1.36929880922735805310e-1) * r +
                       5.9983220655887937690e-1) * r +
                       1.0);
    double x = num / den;
    if (q < 0.0) x = -x;
    return x;
}

// ---------------------------------------------------------------------------
// erf — port of Go math/erf.go's pure algorithm (the amd64 build path:
// haveArchErf is false there, so this exact code is what runs).
// ---------------------------------------------------------------------------

double goErf(double x) {
    const double VeryTiny = 2.848094538889218e-306;
    const double Small = 1.0 / (1 << 28);
    const double erx = 8.45062911510467529297e-01;
    const double efx = 1.28379167095512586316e-01;
    const double efx8 = 1.02703333676410069053e+00;
    const double pp0 = 1.28379167095512558561e-01;
    const double pp1 = -3.25042107247001499370e-01;
    const double pp2 = -2.84817495755985104766e-02;
    const double pp3 = -5.77027029648944159157e-03;
    const double pp4 = -2.37630166566501626084e-05;
    const double qq1 = 3.97917223959155352819e-01;
    const double qq2 = 6.50222499887672944485e-02;
    const double qq3 = 5.08130628187576562776e-03;
    const double qq4 = 1.32494738004321644526e-04;
    const double qq5 = -3.96022827877536812320e-06;
    const double pa0 = -2.36211856075265944077e-03;
    const double pa1 = 4.14856118683748331666e-01;
    const double pa2 = -3.72207876035701323847e-01;
    const double pa3 = 3.18346619901161753674e-01;
    const double pa4 = -1.10894694282396677476e-01;
    const double pa5 = 3.54783043256182359371e-02;
    const double pa6 = -2.16637559486879084300e-03;
    const double qa1 = 1.06420880400844228286e-01;
    const double qa2 = 5.40397917702171048937e-01;
    const double qa3 = 7.18286544141962662868e-02;
    const double qa4 = 1.26171219808761642112e-01;
    const double qa5 = 1.36370839120290507362e-02;
    const double qa6 = 1.19844998467991074170e-02;
    const double ra0 = -9.86494403484714822705e-03;
    const double ra1 = -6.93858572707181764372e-01;
    const double ra2 = -1.05586262253232909814e+01;
    const double ra3 = -6.23753324503260060396e+01;
    const double ra4 = -1.62396669462573470355e+02;
    const double ra5 = -1.84605092906711035994e+02;
    const double ra6 = -8.12874355063065934246e+01;
    const double ra7 = -9.81432934416914548592e+00;
    const double sa1 = 1.96512716674392571292e+01;
    const double sa2 = 1.37657754143519042600e+02;
    const double sa3 = 4.34565877475229228821e+02;
    const double sa4 = 6.45387271733267880336e+02;
    const double sa5 = 4.29008140027567833386e+02;
    const double sa6 = 1.08635005541779435134e+02;
    const double sa7 = 6.57024977031928170135e+00;
    const double sa8 = -6.04244152148580987438e-02;
    const double rb0 = -9.86494292470009928597e-03;
    const double rb1 = -7.99283237680523006574e-01;
    const double rb2 = -1.77579549177547519889e+01;
    const double rb3 = -1.60636384855821916062e+02;
    const double rb4 = -6.37566443368389627722e+02;
    const double rb5 = -1.02509513161107724954e+03;
    const double rb6 = -4.83519191608651397019e+02;
    const double sb1 = 3.03380607434824582924e+01;
    const double sb2 = 3.25792512996573918826e+02;
    const double sb3 = 1.53672958608443695994e+03;
    const double sb4 = 3.19985821950859553908e+03;
    const double sb5 = 2.55305040643316442583e+03;
    const double sb6 = 4.74528541206955367215e+02;
    const double sb7 = -2.24409524465858183362e+01;

    if (std::isnan(x)) return std::nan("");
    if (std::isinf(x)) return x > 0 ? 1.0 : -1.0;
    bool sign = false;
    if (x < 0) {
        x = -x;
        sign = true;
    }
    double temp;
    if (x < 0.84375) {
        if (x < Small) {
            if (x < VeryTiny) {
                temp = 0.125 * (8.0 * x + efx8 * x);
            } else {
                temp = x + efx * x;
            }
        } else {
            double z = x * x;
            double r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            double s = 1 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            double y = r / s;
            temp = x + x * y;
        }
        return sign ? -temp : temp;
    }
    if (x < 1.25) {
        double s = x - 1;
        double P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        double Q = 1 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        if (sign) return -erx - P / Q;
        return erx + P / Q;
    }
    if (x >= 6) {
        return sign ? -1.0 : 1.0;
    }
    double s = 1 / (x * x);
    double R, S;
    if (x < 1 / 0.35) {
        R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
        S = 1 + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
    } else {
        R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
        S = 1 + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
    }
    uint64_t bits;
    std::memcpy(&bits, &x, 8);
    bits &= 0xffffffff00000000ull;
    double z;
    std::memcpy(&z, &bits, 8);
    double r = std::exp(-z * z - 0.5625) * std::exp((z - x) * (z + x) + R / S);
    if (sign) return r / x - 1;
    return 1 - r / x;
}

double normCDF(double x) {
    return 0.5 * (1.0 + goErf(x / std::sqrt(2.0)));
}

// ---------------------------------------------------------------------------
// Python numeric semantics
// ---------------------------------------------------------------------------

int64_t pyRound(double x) {
    double f = std::floor(x);
    double d = x - f;
    if (d > 0.5) return static_cast<int64_t>(f) + 1;
    if (d < 0.5) return static_cast<int64_t>(f);
    if (static_cast<int64_t>(f) % 2 == 0) return static_cast<int64_t>(f);
    return static_cast<int64_t>(f) + 1;
}

std::string pyReprFloat(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    // shortest scientific form via printf + strtod round-trip probe
    char buf[64];
    for (int prec = 1; prec <= 17; prec++) {
        std::snprintf(buf, sizeof buf, "%.*e", prec - 1, v);
        double w = std::strtod(buf, nullptr);
        if (std::memcmp(&w, &v, 8) == 0) break;
    }
    const char* s = buf;
    bool neg = (*s == '-');
    if (neg) s++;
    std::string digits;
    digits += *s++;
    if (*s == '.') {
        s++;
        while (*s != 'e' && *s != '\0') digits += *s++;
    }
    while (*s != 'e' && *s != '\0') s++;
    int exp10 = std::atoi(s + 1);
    if (exp10 >= 16 || exp10 < -4) {
        return std::string(buf);  // printf's form already matches Python's
    }
    std::string out;
    size_t n = digits.size();
    if (static_cast<int>(n) - 1 <= exp10) {
        out = digits + std::string(static_cast<size_t>(exp10 - (static_cast<int>(n) - 1)), '0');
        out += ".0";
    } else if (exp10 >= 0) {
        out = digits.substr(0, static_cast<size_t>(exp10) + 1) + "." +
              digits.substr(static_cast<size_t>(exp10) + 1);
    } else {
        out = "0." + std::string(static_cast<size_t>(-exp10) - 1, '0') + digits;
    }
    if (neg) out = "-" + out;
    return out;
}

std::string pyFormatG(double x, int prec) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", prec, x);
    return std::string(buf);
}

std::string pyFormatF(double x, int prec) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[512];
    std::snprintf(buf, sizeof buf, "%.*f", prec, x);
    return std::string(buf);
}

std::string pyPercent0(double x) {
    double v = x * 100;
    double f = std::floor(v);
    double d = v - f;
    if (d > 0.5 || (d == 0.5 && std::fmod(f, 2) != 0)) {
        f++;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f", f);
    std::string s = buf;
    // Python prints "-0%" for negative zero and negative values that round to
    // zero; guard so a printf-emitted "-0" does not get a second minus.
    if (f == 0 && std::signbit(v) && s[0] != '-') s = "-" + s;
    return s + "%";
}

std::string fmtPlusF(double x, int prec) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%+.*f", prec, x);
    return std::string(buf);
}

std::string fmtPadLeft(const std::string& s, size_t width) {
    if (s.size() >= width) return s;
    return s + std::string(width - s.size(), ' ');
}

double exactMean(const std::vector<double>& xs) {
    if (xs.empty()) {
        throw std::runtime_error("mean requires at least one data point");
    }
    double nonFinite = 0.0;
    for (double v : xs) {
        if (std::isnan(v)) return std::nan("");
        if (std::isinf(v)) {
            if (nonFinite == 0) {
                nonFinite = v;
            } else if (nonFinite != v) {
                return std::nan("");
            }
        }
    }
    if (nonFinite != 0) return nonFinite;
    // Exact rational sum over a common dyadic scale: one integer sum, one
    // correctly-rounded division. The value is identical to summing reduced
    // rationals (Go's big.Rat / CPython's Fraction) and therefore rounds to
    // the same double.
    ScaledInts si = exactScaledInts(xs);
    if (si.ok) {
        BigInt s;
        for (const BigInt& v : si.ints) s = s + v;
        BigInt den = si.scale * BigInt::fromI64(static_cast<int64_t>(xs.size()));
        return Rat{std::move(s), std::move(den)}.toF64();
    }
    Rat total;
    for (double v : xs) total = total.add(Rat::fromF64(v));
    Rat res = total.quo(Rat::fromI64(static_cast<int64_t>(xs.size())));
    return res.toF64();
}

double exactStdev(const std::vector<double>& xs) {
    size_t n = xs.size();
    if (n < 2) {
        throw std::runtime_error("variance requires at least two data points");
    }
    ScaledInts si = exactScaledInts(xs);
    if (si.ok) {
        // v_i = ints[i] / D, mean = S / (n·D) with S = Σ ints. The sum of
        // squared deviations is then exact integer arithmetic over the common
        // denominator n²·D²:
        //   Σ(v_i - mean)² / (n-1) = Σ(n·ints[i] - S)² / (n²·(n-1)·D²).
        // One correctly-rounded conversion, then sqrt — the same sequence
        // (and the same bits) as the reduced-rational reference.
        BigInt S;
        for (const BigInt& v : si.ints) S = S + v;
        BigInt nn = BigInt::fromI64(static_cast<int64_t>(n));
        BigInt ss;
        for (const BigInt& v : si.ints) {
            BigInt dev = nn * v - S;
            ss = ss + dev * dev;
        }
        BigInt den = nn * nn * BigInt::fromI64(static_cast<int64_t>(n - 1)) * si.scale *
                     si.scale;
        double var = Rat{std::move(ss), std::move(den)}.toF64();
        return std::sqrt(var);
    }
    Rat total;
    for (double v : xs) total = total.add(Rat::fromF64(v));
    Rat meanR = total.quo(Rat::fromI64(static_cast<int64_t>(n)));
    Rat ss;
    for (double v : xs) {
        Rat d = Rat::fromF64(v).sub(meanR);
        ss = ss.add(d.mul(d));
    }
    Rat varR = ss.quo(Rat::fromI64(static_cast<int64_t>(n - 1)));
    return std::sqrt(varR.toF64());
}

ScaledInts exactScaledInts(const std::vector<double>& values) {
    ScaledInts out;
    struct Ratio {
        BigInt m;
        int e;
    };
    std::vector<Ratio> ratios;
    ratios.reserve(values.size());
    int eMax = 0;
    for (double v : values) {
        if (std::isinf(v) || std::isnan(v)) {
            return out;
        }
        Rat r = Rat::fromF64(v);
        int e = r.den.bitLen() - 1;
        if (e > eMax) eMax = e;
        ratios.push_back(Ratio{r.num, e});
    }
    out.scale = BigInt::pow2(eMax);
    out.ints.reserve(values.size());
    for (const auto& rt : ratios) {
        out.ints.push_back(rt.m << (eMax - rt.e));
    }
    out.ok = true;
    return out;
}

}  // namespace evalsig

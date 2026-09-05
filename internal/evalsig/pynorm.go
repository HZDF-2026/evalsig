// Bit-exact ports of the CPython 3.10 statistics building blocks evalsig
// depends on: NormalDist.inv_cdf (AS 241, Wichura 1988 — same rational
// approximations, same operation order as CPython Lib/statistics.py) and
// NormalDist.cdf (0.5*(1+erf(x/sqrt(2)))).
package evalsig

import "math"

// pyMul is a non-fusable multiply. CPython rounds x*y to float64 before the
// surrounding + or - sees it; Go permits an implementation to fuse x*y+z into
// a single FMA — gc does exactly that on arm64 and ppc64, across statements
// too — which rounds once instead of twice and changes last-bit results. The
// //go:noinline call boundary forces the product out as a rounded float64
// before any consumer can fuse it, keeping this port bit-exact with CPython
// on every architecture. The C++ port compiles with -ffp-contract=off for
// the same reason.
//
//go:noinline
func pyMul(x, y float64) float64 { return x * y }

// fdlibmLog is the pure-Go fdlibm/FreeBSD msun __ieee754_log, the same
// algorithm as math.Log's non-assembly path. We call it directly instead of
// math.Log because math.Log dispatches to a hand-written assembly variant on
// some architectures (log_amd64.s) whose last-ulp behavior differs from the
// pure-Go one — a same-source-different-bits split we refuse to ship.
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. (From
// FreeBSD /usr/src/lib/msun/src/e_log.c, via Go's math/log.go; see the
// license notice in this repository's README.)
func fdlibmLog(x float64) float64 {
	const (
		Ln2Hi = 6.93147180369123816490e-01 /* 3fe62e42 fee00000 */
		Ln2Lo = 1.90821492927058770002e-10 /* 3dea39ef 35793c76 */
		L1    = 6.666666666666735130e-01   /* 3FE55555 55555593 */
		L2    = 3.999999999940941908e-01   /* 3FD99999 9997FA04 */
		L3    = 2.857142874366239149e-01   /* 3FD24924 94229359 */
		L4    = 2.222219843214978396e-01   /* 3FCC71C5 1D8E78AF */
		L5    = 1.818357216161805012e-01   /* 3FC74664 96CB03DE */
		L6    = 1.531383769920937332e-01   /* 3FC39A09 D078C69F */
		L7    = 1.479819860511658591e-01   /* 3FC2F112 DF3E5244 */
	)
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(-1)
	}
	f1, ki := math.Frexp(x)
	if f1 < math.Sqrt2/2 {
		f1 *= 2
		ki--
	}
	f := f1 - 1
	k := float64(ki)
	s := f / (2 + f)
	s2 := s * s
	s4 := s2 * s2
	t1 := s2 * (L1 + pyMul(s4, L3+pyMul(s4, L5+pyMul(s4, L7))))
	t2 := s4 * (L2 + pyMul(s4, L4+pyMul(s4, L6)))
	R := t1 + t2
	hfsq := 0.5 * f * f
	return pyMul(k, Ln2Hi) - ((hfsq - (pyMul(s, hfsq+R) + pyMul(k, Ln2Lo))) - f)
}

// normInvCDF is NormalDist().inv_cdf(p) for the standard normal distribution,
// bit-exact with CPython's _normal_dist_inv_cdf (statistics.py, AS 241).
func normInvCDF(p float64) float64 {
	q := p - 0.5
	if math.Abs(q) <= 0.425 {
		r := 0.180625 - pyMul(q, q)
		num := 2.5090809287301226727e+3
		num = pyMul(num, r) + 3.3430575583588128105e+4
		num = pyMul(num, r) + 6.7265770927008700853e+4
		num = pyMul(num, r) + 4.5921953931549871457e+4
		num = pyMul(num, r) + 1.3731693765509461125e+4
		num = pyMul(num, r) + 1.9715909503065514427e+3
		num = pyMul(num, r) + 1.3314166789178437745e+2
		num = pyMul(num, r) + 3.3871328727963666080e+0
		den := 5.2264952788528545610e+3
		den = pyMul(den, r) + 2.8729085735721942674e+4
		den = pyMul(den, r) + 3.9307895800092710610e+4
		den = pyMul(den, r) + 2.1213794301586595867e+4
		den = pyMul(den, r) + 5.3941960214247511077e+3
		den = pyMul(den, r) + 6.8718700749205790830e+2
		den = pyMul(den, r) + 4.2313330701600911252e+1
		den = pyMul(den, r) + 1.0
		x := num * q / den
		return x
	}
	r := p
	if q > 0.0 {
		r = 1.0 - p
	}
	r = math.Sqrt(-fdlibmLog(r))
	if r <= 5.0 {
		r -= 1.6
		num := 7.74545014278341407640e-4
		num = pyMul(num, r) + 2.27238449892691845833e-2
		num = pyMul(num, r) + 2.41780725177450611770e-1
		num = pyMul(num, r) + 1.27045825245236838258e+0
		num = pyMul(num, r) + 3.64784832476320460504e+0
		num = pyMul(num, r) + 5.76949722146069140550e+0
		num = pyMul(num, r) + 4.63033784615654529590e+0
		num = pyMul(num, r) + 1.42343711074968357734e+0
		den := 1.05075007164441684324e-9
		den = pyMul(den, r) + 5.47593808499534494600e-4
		den = pyMul(den, r) + 1.51986665636164571966e-2
		den = pyMul(den, r) + 1.48103976427480074590e-1
		den = pyMul(den, r) + 6.89767334985100004550e-1
		den = pyMul(den, r) + 1.67638483018380384940e+0
		den = pyMul(den, r) + 2.05319162663775882187e+0
		den = pyMul(den, r) + 1.0
		x := num / den
		if q < 0.0 {
			x = -x
		}
		return x
	}
	r -= 5.0
	num := 2.01033439929228813265e-7
	num = pyMul(num, r) + 2.71155556874348757815e-5
	num = pyMul(num, r) + 1.24266094738807843860e-3
	num = pyMul(num, r) + 2.65321895265761230930e-2
	num = pyMul(num, r) + 2.96560571828504891230e-1
	num = pyMul(num, r) + 1.78482653991729133580e+0
	num = pyMul(num, r) + 5.46378491116411436990e+0
	num = pyMul(num, r) + 6.65790464350110377720e+0
	den := 2.04426310338993978564e-15
	den = pyMul(den, r) + 1.42151175831644588870e-7
	den = pyMul(den, r) + 1.84631831751005468180e-5
	den = pyMul(den, r) + 7.86869131145613259100e-4
	den = pyMul(den, r) + 1.48753612908506148525e-2
	den = pyMul(den, r) + 1.36929880922735805310e-1
	den = pyMul(den, r) + 5.9983220655887937690e-1
	den = pyMul(den, r) + 1.0
	x := num / den
	if q < 0.0 {
		x = -x
	}
	return x
}

// normCDF is NormalDist().cdf(x) for the standard normal distribution:
// 0.5 * (1.0 + erf(x / sqrt(2))). Go's math.Erf differs from the C runtime's
// erf by at most a few ulp; never enough to flip a %.4g rendering.
func normCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt(2.0)))
}

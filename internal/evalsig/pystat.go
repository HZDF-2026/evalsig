// Bit-exact ports of the CPython numeric semantics evalsig leans on:
// statistics.mean / statistics.stdev (exact rational summation, one final
// correctly-rounded division), round() (half-to-even), float repr, the
// '.Ng' / '.Nf' / '.0%' format specs, math.fsum, and json.dumps(indent=2,
// ensure_ascii=True) with insertion-ordered objects.
//
// The exact-arithmetic paths matter: naive float addition drifts from
// Python's mean by an ulp on adversarial data, and the bootstrap hot path
// compares *bounds*, where an ulp is a visible mismatch.
package evalsig

import (
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf16"
)

// pyRound is Python round(x) with no ndigits: nearest integer, ties to even.
func pyRound(x float64) int64 {
	f := math.Floor(x)
	d := x - f
	if d > 0.5 {
		return int64(f) + 1
	}
	if d < 0.5 {
		return int64(f)
	}
	if int64(f)%2 == 0 {
		return int64(f)
	}
	return int64(f) + 1
}

func ratOf(v float64) *big.Rat {
	return new(big.Rat).SetFloat64(v)
}

// exactMean is statistics.mean(data) for float data: exact rational sum,
// exact division by the count, one correctly-rounded conversion to float.
// Non-finite inputs follow Python's _sum special handling.
func exactMean(xs []float64) float64 {
	if len(xs) == 0 {
		panic("mean requires at least one data point")
	}
	nonFinite := 0.0
	for _, v := range xs {
		if math.IsNaN(v) {
			return math.NaN()
		}
		if math.IsInf(v, 0) {
			if nonFinite == 0 {
				nonFinite = v
			} else if nonFinite != v {
				return math.NaN()
			}
		}
	}
	if nonFinite != 0 {
		return nonFinite
	}
	total := new(big.Rat)
	for _, v := range xs {
		total.Add(total, ratOf(v))
	}
	res := new(big.Rat).Quo(total, new(big.Rat).SetInt64(int64(len(xs))))
	f, _ := res.Float64()
	return f
}

// exactStdev is statistics.stdev(data): exact sum of squared deviations from
// the exact mean, exact division by n-1, correctly-rounded to float, sqrt.
func exactStdev(xs []float64) float64 {
	n := len(xs)
	if n < 2 {
		panic("variance requires at least two data points")
	}
	total := new(big.Rat)
	for _, v := range xs {
		total.Add(total, ratOf(v))
	}
	meanR := new(big.Rat).Quo(total, new(big.Rat).SetInt64(int64(n)))
	ss := new(big.Rat)
	for _, v := range xs {
		d := new(big.Rat).Sub(ratOf(v), meanR)
		ss.Add(ss, new(big.Rat).Mul(d, d))
	}
	varR := new(big.Rat).Quo(ss, new(big.Rat).SetInt64(int64(n-1)))
	vf, _ := varR.Float64()
	return math.Sqrt(vf)
}

// pyReprFloat is Python's repr(x) / str(x) for floats: shortest digits that
// round-trip, exponent form when exp10 < -4 or >= 16, ".0" on integral
// values, "inf"/"nan" spellings.
func pyReprFloat(v float64) string {
	if math.IsNaN(v) {
		return "nan"
	}
	if math.IsInf(v, 1) {
		return "inf"
	}
	if math.IsInf(v, -1) {
		return "-inf"
	}
	e := strconv.FormatFloat(v, 'e', -1, 64)
	i := strings.IndexByte(e, 'e')
	exp, _ := strconv.Atoi(e[i+1:])
	if exp >= 16 || exp < -4 {
		return e
	}
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return s
}

// pyFormatG is Python's format(x, '.Ng') spec.
func pyFormatG(x float64, prec int) string {
	if math.IsNaN(x) {
		return "nan"
	}
	if math.IsInf(x, 1) {
		return "inf"
	}
	if math.IsInf(x, -1) {
		return "-inf"
	}
	return strconv.FormatFloat(x, 'g', prec, 64)
}

// pyFormatF is Python's format(x, '.Nf') spec (used with %+ variants at the
// call site).
func pyFormatF(x float64, prec int) string {
	if math.IsNaN(x) {
		return "nan"
	}
	if math.IsInf(x, 1) {
		return "inf"
	}
	if math.IsInf(x, -1) {
		return "-inf"
	}
	return strconv.FormatFloat(x, 'f', prec, 64)
}

// pyPercent0 is Python's format(x, '.0%'): x*100 rounded to an integer with
// round-half-even, then '%'.
func pyPercent0(x float64) string {
	v := x * 100
	f := math.Floor(v)
	d := v - f
	if d > 0.5 || (d == 0.5 && math.Mod(f, 2) != 0) {
		f++
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	// Python prints "-0%" for negative zero and negative values that round to
	// zero; FormatFloat already emits "-0" for negative zero, so guard against
	// prepending a second minus.
	if f == 0 && math.Signbit(v) && s[0] != '-' {
		s = "-" + s
	}
	return s + "%"
}

// pyJSONDumps is json.dumps(v, indent=2): 2-space indent, ",\n" item
// separator, ": " key separator, ensure_ascii escaping, Python float repr,
// NaN / Infinity literals allowed, insertion-ordered objects.
func pyJSONDumps(v interface{}) string {
	var b strings.Builder
	emitJSON(&b, v, 0)
	return b.String()
}

func emitJSON(b *strings.Builder, v interface{}, depth int) {
	switch t := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case int:
		b.WriteString(strconv.Itoa(t))
	case int64:
		b.WriteString(strconv.FormatInt(t, 10))
	case float64:
		b.WriteString(pyJSONFloat(t))
	case json.Number:
		// A value that round-tripped through JSON: json.loads splits
		// int/float by the presence of '.'/'e', and dumps re-emits ints
		// verbatim and floats via repr.
		if numIsInt(t) {
			b.WriteString(t.String())
		} else {
			f, err := numFloat(t)
			if err != nil {
				b.WriteString(t.String())
			} else {
				b.WriteString(pyJSONFloat(f))
			}
		}
	case string:
		b.WriteString(pyJSONString(t))
	case []interface{}:
		if len(t) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteString("[\n")
		ind := strings.Repeat("  ", depth+1)
		for i, e := range t {
			if i > 0 {
				b.WriteString(",\n")
			}
			b.WriteString(ind)
			emitJSON(b, e, depth+1)
		}
		b.WriteString("\n")
		b.WriteString(strings.Repeat("  ", depth))
		b.WriteString("]")
	case *pyObj:
		if len(t.pairs) == 0 {
			b.WriteString("{}")
			return
		}
		b.WriteString("{\n")
		ind := strings.Repeat("  ", depth+1)
		for i, kv := range t.pairs {
			if i > 0 {
				b.WriteString(",\n")
			}
			b.WriteString(ind)
			b.WriteString(pyJSONString(kv.k))
			b.WriteString(": ")
			emitJSON(b, kv.v, depth+1)
		}
		b.WriteString("\n")
		b.WriteString(strings.Repeat("  ", depth))
		b.WriteString("}")
	default:
		panic("pyJSONDumps: unsupported type")
	}
}

func pyJSONFloat(v float64) string {
	if math.IsNaN(v) {
		return "NaN"
	}
	if math.IsInf(v, 1) {
		return "Infinity"
	}
	if math.IsInf(v, -1) {
		return "-Infinity"
	}
	return pyReprFloat(v)
}

func pyJSONString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 || r >= 0x7f {
				writeEscapedRune(&b, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

func writeEscapedRune(b *strings.Builder, r rune) {
	if r > 0xffff {
		r1, r2 := utf16.EncodeRune(r)
		writeEscapedRune(b, r1)
		writeEscapedRune(b, r2)
		return
	}
	b.WriteString(`\u`)
	s := strconv.FormatInt(int64(r), 16)
	for len(s) < 4 {
		s = "0" + s
	}
	b.WriteString(s)
}

var errMeanEmpty = errors.New("mean requires at least one data point")

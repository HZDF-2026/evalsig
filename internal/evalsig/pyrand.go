// Bit-exact port of CPython's random.Random for the operations evalsig uses:
// integer seeding (init_by_array over the little-endian 32-bit words of |n|),
// getrandbits, and randrange's _randbelow rejection loop. The bootstrap CI is
// deterministic given a seed, and this file is what makes the Go port draw
// the *same* resample indices as CPython, so the two implementations agree
// bit-for-bit, not merely in distribution.
package evalsig

// pyRandom is CPython's _random.Random: MT19937 state plus a next-word index.
type pyRandom struct {
	mt    [624]uint32
	index int
}

const (
	mtN       = 624
	mtM       = 397
	mtMatrixA = 0x9908b0df
	mtUpper   = 0x80000000
	mtLower   = 0x7fffffff
)

func (r *pyRandom) initGenrand(s uint32) {
	r.mt[0] = s
	for i := 1; i < mtN; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	r.index = mtN
}

func (r *pyRandom) initByArray(key []uint32) {
	r.initGenrand(19650218)
	i, j := 1, 0
	k := mtN
	if len(key) > k {
		k = len(key)
	}
	for ; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1]>>30)) * 1664525)) + key[j] + uint32(j)
		i++
		j++
		if i >= mtN {
			r.mt[0] = r.mt[mtN-1]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k = mtN - 1; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1]>>30)) * 1566083941)) - uint32(i)
		i++
		if i >= mtN {
			r.mt[0] = r.mt[mtN-1]
			i = 1
		}
	}
	r.mt[0] = 0x80000000
}

// seedInt reproduces random.Random(n).seed(n) for a non-negative int64:
// the absolute value split into little-endian 32-bit words (at least one).
func (r *pyRandom) seedInt(seed int64) {
	u := uint64(seed)
	if seed < 0 {
		u = uint64(-seed)
	}
	var key []uint32
	if u == 0 {
		key = []uint32{0}
	} else {
		for u > 0 {
			key = append(key, uint32(u&0xffffffff))
			u >>= 32
		}
	}
	r.initByArray(key)
}

func (r *pyRandom) genrandUint32() uint32 {
	if r.index >= mtN {
		if r.index == mtN+1 {
			r.initGenrand(5489)
		}
		var y uint32
		mag01 := [2]uint32{0, mtMatrixA}
		kk := 0
		for ; kk < mtN-mtM; kk++ {
			y = (r.mt[kk] & mtUpper) | (r.mt[kk+1] & mtLower)
			r.mt[kk] = r.mt[kk+mtM] ^ (y >> 1) ^ mag01[y&0x1]
		}
		for ; kk < mtN-1; kk++ {
			y = (r.mt[kk] & mtUpper) | (r.mt[kk+1] & mtLower)
			r.mt[kk] = r.mt[kk+(mtM-mtN)] ^ (y >> 1) ^ mag01[y&0x1]
		}
		y = (r.mt[mtN-1] & mtUpper) | (r.mt[0] & mtLower)
		r.mt[mtN-1] = r.mt[mtM-1] ^ (y >> 1) ^ mag01[y&0x1]
		r.index = 0
	}
	y := r.mt[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// getrandbits reproduces random.Random.getrandbits(k) for 1 <= k <= 64.
func (r *pyRandom) getrandbits(k int) uint64 {
	if k <= 0 {
		panic("number of bits must be greater than zero")
	}
	if k <= 32 {
		return uint64(r.genrandUint32() >> (32 - k))
	}
	words := (k-1)/32 + 1
	var x uint64
	kk := k
	for i := 0; i < words; i, kk = i+1, kk-32 {
		w := r.genrandUint32()
		if kk < 32 {
			w >>= 32 - kk
		}
		x |= uint64(w) << (i * 32)
	}
	return x
}

// randbelow reproduces random.Random._randbelow(n) (the getrandbits variant):
// k = n.bit_length(); draw k bits, reject while >= n.
func (r *pyRandom) randbelow(n uint64) uint64 {
	k := 64 - leadingZeros64(n)
	v := r.getrandbits(k)
	for v >= n {
		v = r.getrandbits(k)
	}
	return v
}

func leadingZeros64(x uint64) int {
	n := 0
	for x&0x8000000000000000 == 0 && n < 64 {
		x <<= 1
		n++
	}
	return n
}

// randrange reproduces random.Random.randrange(stop) for stop >= 1.
func (r *pyRandom) randrange(stop int) int {
	return int(r.randbelow(uint64(stop)))
}

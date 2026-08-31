package evalsig

import (
	"fmt"
	"math"
	"testing"
)

func TestLogProbe(t *testing.T) {
	inputs := []float64{1.0 - 0.99, 0.01, 0.025, 0.05, 0.0069, 0.0025, 0.001, 0.0025,
		0.005, 1.0 - 0.975, 1.0 - 0.95, 1.0 - 0.999, 1e-10, 1e-7, 1e-5, 1e-4, 0.075,
		1.0 - 0.9, 1.0 - 0.925, 1.0 - 0.9999, 1.0 - 0.999999, 1e-10, 0.425,
		0.42500001, 0.45, 0.55, 1.0 - 0.4, 1.0 - 0.6, 1.0 - 0.3, 0.1, 0.2, 0.3, 0.5}
	for _, x := range inputs {
		l := math.Log(x)
		fmt.Printf("%x %s\n", math.Float64bits(l), pyReprFloat(x))
	}
}

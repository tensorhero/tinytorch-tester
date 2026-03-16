package helpers

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// AssertFloatClose checks that actual is within atol of expected.
func AssertFloatClose(actual, expected, atol float64) error {
	if math.Abs(actual-expected) > atol {
		return fmt.Errorf("expected %.6f, got %.6f (atol=%.6f)", expected, actual, atol)
	}
	return nil
}

// AssertFloatSliceClose checks that two float slices are element-wise within atol.
func AssertFloatSliceClose(actual, expected []float64, atol float64) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("length mismatch: got %d, expected %d", len(actual), len(expected))
	}
	for i := range actual {
		if err := AssertFloatClose(actual[i], expected[i], atol); err != nil {
			return fmt.Errorf("index %d: %w", i, err)
		}
	}
	return nil
}

// ParseFloatResult parses a result string into a float64.
func ParseFloatResult(results map[string]string, testName string) (float64, error) {
	s, ok := results[testName]
	if !ok {
		return 0, fmt.Errorf("test %q: no output found", testName)
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, fmt.Errorf("test %q: cannot parse %q as float: %v", testName, s, err)
	}
	return v, nil
}

// AssertFloatResultClose checks that a structured output float result is close to expected.
func AssertFloatResultClose(results map[string]string, testName string, expected, atol float64) error {
	actual, err := ParseFloatResult(results, testName)
	if err != nil {
		return err
	}
	if err := AssertFloatClose(actual, expected, atol); err != nil {
		return fmt.Errorf("test %q: %w", testName, err)
	}
	return nil
}

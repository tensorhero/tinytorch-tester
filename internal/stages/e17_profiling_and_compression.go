package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e17ProfilingAndCompressionTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "profiling-and-compression",
		Timeout:     30 * time.Second,
		TestFunc:    testE17ProfilingAndCompression,
		CompileStep: autoCompileStep("TestE17", "test_e17"),
	}
}

func testE17ProfilingAndCompression(harness *test_case_harness.TestCaseHarness) error {
	logger := harness.Logger
	workDir := harness.SubmissionDir
	lang := harness.DetectedLang

	r := runner.Run(workDir, lang.RunCmd, lang.RunArgs...).
		WithTimeout(15 * time.Second).
		WithLogger(logger).
		Execute().
		Exit(0)

	if err := r.Error(); err != nil {
		return fmt.Errorf("test driver failed: %v", err)
	}

	results := helpers.ParseStructuredOutput(string(r.Result().Stdout))

	// --- Exact match assertions ---

	exactTests := []struct {
		name     string
		expected string
		label    string
	}{
		// Part 1: Profiler — countParams
		{"count_params_single_linear", "15", "Linear(4,3) with bias = 15 params"},
		{"count_params_no_bias", "12", "Linear(4,3) without bias = 12 params"},
		{"count_params_sequential", "58", "Sequential(Linear(4,8), Linear(8,2)) = 58 params"},
		// Part 1: Profiler — countFlops
		{"count_flops_single_linear", "24", "Linear(4,3) FLOPs = 24"},
		{"count_flops_sequential", "17664", "Sequential FLOPs = 17664"},
		{"count_flops_empty", "0", "Empty model FLOPs = 0"},
		// Part 2: Pruner — magnitudePrune
		{"prune_nonzero_count", "3", "50% prune leaves 3 non-zeros"},
		{"prune_big_value_survives", "true", "Large magnitude value survives pruning"},
		{"prune_small_value_zero", "true", "Small magnitude value is pruned to zero"},
		{"prune_zero_sparsity", "true", "Sparsity=0 prunes nothing"},
		// Part 2: Pruner — measureSparsity
		{"sparsity_all_ones", "true", "All-ones model has 0% sparsity"},
		{"sparsity_ignores_bias", "true", "1D bias is excluded from sparsity"},
		{"sparsity_after_prune", "true", "Sparsity >= 25% after 30% pruning"},
	}

	for _, tc := range exactTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Float close assertion (cross-language float precision) ---

	const atol = 1e-4

	floatTests := []struct {
		name     string
		expected float64
		label    string
	}{
		{"sparsity_known_zeros", 33.333333, "4/12 zeros = 33.33% sparsity"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E17 tests passed!", total)
	return nil
}

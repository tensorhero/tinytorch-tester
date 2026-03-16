package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e01TensorClassTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "tensor-class",
		Timeout:     30 * time.Second,
		TestFunc:    testE01TensorClass,
		CompileStep: autoCompileStep("TestE01", "test_e01"),
	}
}

func testE01TensorClass(harness *test_case_harness.TestCaseHarness) error {
	logger := harness.Logger
	workDir := harness.SubmissionDir
	lang := harness.DetectedLang

	// Run test driver
	r := runner.Run(workDir, lang.RunCmd, lang.RunArgs...).
		WithTimeout(10 * time.Second).
		WithLogger(logger).
		Execute().
		Exit(0)

	if err := r.Error(); err != nil {
		return fmt.Errorf("test driver failed: %v", err)
	}

	results := helpers.ParseStructuredOutput(string(r.Result().Stdout))

	// --- Factory methods ---

	factoryTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"zeros_size", "6", "zeros(2,3).size() == 6"},
		{"zeros_ndim", "2", "zeros(2,3).ndim() == 2"},
		{"zeros_shape", "2,3", "zeros(2,3).shape() == [2,3]"},
		{"zeros_toString", "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]", "zeros(2,3) toString"},
		{"ones_size", "12", "ones(3,4).size() == 12"},
		{"ones_shape", "3,4", "ones(3,4).shape() == [3,4]"},
		{"ones_toString", "[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]", "ones(2,3) toString"},
		{"fromArray_1d_toString", "[1.0, 2.0, 3.0]", "fromArray 1D toString"},
		{"fromArray_2d_toString", "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]", "fromArray 2D toString"},
		{"full_toString", "[[7.0, 7.0], [7.0, 7.0]]", "full(7.0, 2,2) toString"},
	}

	for _, tc := range factoryTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Element-wise operations ---

	opTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"add_toString", "[[2.0, 3.0], [4.0, 5.0]]", "add: [[1,2],[3,4]] + ones(2,2)"},
		{"sub_toString", "[[0.0, 1.0], [2.0, 3.0]]", "sub: [[1,2],[3,4]] - ones(2,2)"},
		{"mul_toString", "[[1.0, 2.0], [3.0, 4.0]]", "mul: [[1,2],[3,4]] * ones(2,2)"},
		{"div_toString", "[[1.0, 2.0], [3.0, 4.0]]", "div: [[2,4],[6,8]] / full(2,2,2)"},
	}

	for _, tc := range opTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Matrix multiplication ---

	matmulTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"matMul_shape", "2,2", "matMul (2,3)@(3,2) → shape (2,2)"},
		{"matMul_toString", "[[22.0, 28.0], [49.0, 64.0]]", "matMul values correct"},
	}

	for _, tc := range matmulTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Reduction operations ---

	reductionTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"sum_axis0", "[5.0, 7.0, 9.0]", "sum(axis=0) column sums"},
		{"sum_axis1", "[6.0, 15.0]", "sum(axis=1) row sums"},
		{"mean_axis0_keepDims", "[[2.5, 3.5, 4.5]]", "mean(axis=0, keepDims=true)"},
		{"mean_axis1", "[2.0, 5.0]", "mean(axis=1) row means"},
	}

	for _, tc := range reductionTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Shape operations ---

	shapeTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"reshape_shape", "3,2", "reshape(3,2).shape() == [3,2]"},
		{"reshape_toString", "[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]", "reshape(3,2) data correct"},
		{"transpose_shape", "3,2", "transpose(1,0).shape() == [3,2]"},
		{"transpose_toString", "[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]", "transpose(1,0) data correct"},
	}

	for _, tc := range shapeTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Gradient fields (dormant) ---

	gradTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"requiresGrad", "false", "requiresGrad defaults to false"},
		{"grad_null", "true", "grad defaults to null/None"},
		{"gradFn_null", "true", "gradFn defaults to null/None"},
	}

	for _, tc := range gradTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- randn (shape only, values are random) ---

	randnTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"randn_shape", "4,5", "randn(4,5).shape() == [4,5]"},
		{"randn_size", "20", "randn(4,5).size() == 20"},
	}

	for _, tc := range randnTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(factoryTests) + len(opTests) + len(matmulTests) +
		len(reductionTests) + len(shapeTests) + len(gradTests) + len(randnTests)
	logger.Successf("All %d E01 tests passed!", total)
	return nil
}

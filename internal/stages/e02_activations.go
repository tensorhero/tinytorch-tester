package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e02ActivationsTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "activations",
		Timeout:     30 * time.Second,
		TestFunc:    testE02Activations,
		CompileStep: autoCompileStep("TestE02", "test_e02"),
	}
}

func testE02Activations(harness *test_case_harness.TestCaseHarness) error {
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

	const atol = 1e-4

	// --- ReLU ---

	reluExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"relu_basic", "[0.0, 0.0, 1.0, 2.0]", "relu([-1,0,1,2]) → [0,0,1,2]"},
		{"relu_all_negative", "[0.0, 0.0, 0.0]", "relu([-3,-2,-1]) → [0,0,0]"},
		{"relu_shape", "2,2", "relu preserves 2D shape"},
	}

	for _, tc := range reluExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Sigmoid ---

	sigmoidFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"sigmoid_zero", 0.5, "sigmoid(0) = 0.5"},
		{"sigmoid_neg2", 0.119203, "sigmoid(-2) ≈ 0.1192"},
		{"sigmoid_pos2", 0.880797, "sigmoid(2) ≈ 0.8808"},
		{"sigmoid_symmetry_sum", 1.0, "sigmoid(-5) + sigmoid(5) ≈ 1.0"},
	}

	for _, tc := range sigmoidFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Tanh ---

	tanhFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"tanh_zero", 0.0, "tanh(0) = 0.0"},
		{"tanh_pos1", 0.761594, "tanh(1) ≈ 0.7616"},
		{"tanh_neg1", -0.761594, "tanh(-1) ≈ -0.7616"},
		{"tanh_antisymmetry_sum", 0.0, "tanh(-3) + tanh(3) ≈ 0.0"},
	}

	for _, tc := range tanhFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- GELU ---

	geluFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"gelu_zero", 0.0, "gelu(0) = 0.0"},
		{"gelu_neg1", -0.158808, "gelu(-1) ≈ -0.1588"},
		{"gelu_pos1", 0.841192, "gelu(1) ≈ 0.8412"},
		{"gelu_pos3", 2.996363, "gelu(3) ≈ 2.9964 (large pos preserved)"},
	}

	for _, tc := range geluFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Softmax ---

	softmaxFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"softmax_val0", 0.090031, "softmax([1,2,3])[0] ≈ 0.0900"},
		{"softmax_val1", 0.244728, "softmax([1,2,3])[1] ≈ 0.2447"},
		{"softmax_val2", 0.665241, "softmax([1,2,3])[2] ≈ 0.6652"},
		{"softmax_sum", 1.0, "softmax sum ≈ 1.0"},
		{"softmax_uniform_val", 0.333333, "softmax([1,1,1])[0] ≈ 1/3"},
		{"softmax_stability_sum", 1.0, "softmax([1000,1001,1002]) sum ≈ 1.0 (no overflow)"},
		{"softmax_stability_val2", 0.665241, "softmax([1000,1001,1002])[2] ≈ 0.6652"},
	}

	for _, tc := range softmaxFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Softmax 2D ---

	softmax2dExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"softmax_2d_shape", "2,3", "softmax 2D shape preserved"},
	}

	for _, tc := range softmax2dExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	softmax2dFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"softmax_2d_row0_sum", 1.0, "softmax 2D row 0 sum ≈ 1.0"},
		{"softmax_2d_row1_val", 0.333333, "softmax 2D uniform row ≈ 1/3"},
	}

	for _, tc := range softmax2dFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(reluExact) + len(sigmoidFloat) + len(tanhFloat) +
		len(geluFloat) + len(softmaxFloat) + len(softmax2dExact) + len(softmax2dFloat)
	logger.Successf("All %d E02 tests passed!", total)
	return nil
}

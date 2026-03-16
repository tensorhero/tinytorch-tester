package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e04LossFunctionsTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "loss-functions",
		Timeout:     30 * time.Second,
		TestFunc:    testE04LossFunctions,
		CompileStep: autoCompileStep("TestE04", "test_e04"),
	}
}

func testE04LossFunctions(harness *test_case_harness.TestCaseHarness) error {
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

	const atol = 1e-3 // wider tolerance for chained float32 loss computations

	// --- MSE ---

	mseFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"mse_zero_loss", 0.0, "MSE perfect prediction → 0"},
		{"mse_basic", 1.0, "MSE pred=[1,0] target=[0,1] → 1.0"},
		{"mse_known", 0.375, "MSE known diff → 0.375"},
		{"mse_2d", 0.375, "MSE 2D same data → 0.375"},
	}

	for _, tc := range mseFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	mseExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"mse_shape", "1", "MSE output shape [1]"},
	}

	for _, tc := range mseExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- logSoftmax ---

	lsFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"logsoftmax_val0", -2.407606, "logSoftmax([1,2,3])[0] ≈ -2.4076"},
		{"logsoftmax_sum_exp", 1.0, "sum(exp(logSoftmax)) ≈ 1.0"},
		{"logsoftmax_uniform", -1.098612, "logSoftmax uniform → log(1/3)"},
		{"logsoftmax_stability", 1.0, "logSoftmax numerical stability"},
	}

	for _, tc := range lsFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	lsExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"logsoftmax_shape", "2,3", "logSoftmax 2D shape preserved"},
		{"logsoftmax_negative", "true", "logSoftmax values all ≤ 0"},
	}

	for _, tc := range lsExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- CrossEntropy ---

	ceFloat := []struct {
		name     string
		expected float64
		label    string
	}{
		{"ce_perfect", 0.0, "CE perfect prediction → ≈ 0"},
		{"ce_uniform", 1.098612, "CE uniform logits → log(3)"},
		{"ce_known", 0.417030, "CE known logits → 0.4170"},
		{"ce_batch", 0.318561, "CE batch of 2 → 0.3186"},
		{"ce_wrong", 2.239394, "CE wrong prediction → high loss"},
	}

	for _, tc := range ceFloat {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	ceExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"ce_shape", "1", "CE output shape [1]"},
	}

	for _, tc := range ceExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(mseFloat) + len(mseExact) + len(lsFloat) + len(lsExact) + len(ceFloat) + len(ceExact)
	logger.Successf("All %d E04 tests passed!", total)
	return nil
}

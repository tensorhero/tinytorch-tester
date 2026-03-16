package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e08OptimizersTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "optimizers",
		Timeout:     30 * time.Second,
		TestFunc:    testE08Optimizers,
		CompileStep: autoCompileStep("TestE08", "test_e08"),
	}
}

func testE08Optimizers(harness *test_case_harness.TestCaseHarness) error {
	logger := harness.Logger
	workDir := harness.SubmissionDir
	lang := harness.DetectedLang

	r := runner.Run(workDir, lang.RunCmd, lang.RunArgs...).
		WithTimeout(10 * time.Second).
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
		{"sgd_zero_grad", "true", "zeroGrad clears gradient to null"},
		{"sgd_no_grad_skip", "true", "SGD skips params without gradient"},
		{"adamw_smaller_than_adam", "true", "AdamW param < Adam param (weight decay)"},
	}

	for _, tc := range exactTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Float close assertions ---

	const atol = 1e-4

	floatTests := []struct {
		name     string
		expected float64
		label    string
	}{
		// SGD basic step
		{"sgd_step_param0", 1.5, "SGD step: param[0] = 2.0 - 0.5*1.0 = 1.5"},
		{"sgd_step_param1", 3.0, "SGD step: param[1] = 4.0 - 0.5*2.0 = 3.0"},
		// SGD momentum
		{"sgd_momentum_step1", 0.8, "SGD momentum step1: 1.0 - 0.1*2.0 = 0.8"},
		{"sgd_momentum_step2", 0.42, "SGD momentum step2: 0.8 - 0.1*3.8 = 0.42"},
		// Adam
		{"adam_step1", 2.99, "Adam step1: 3.0 - 0.01 ≈ 2.99"},
		{"adam_step2", 2.98, "Adam step2: 2.99 - 0.01 ≈ 2.98"},
		// AdamW
		{"adamw_step1", 1.98, "AdamW step1: weight decay + Adam update ≈ 1.98"},
		// Multiple parameters
		{"multi_params_p1_0", 0.9, "multi params p1[0] = 1.0 - 0.2*0.5 = 0.9"},
		{"multi_params_p1_1", 1.9, "multi params p1[1] = 2.0 - 0.2*0.5 = 1.9"},
		{"multi_params_p2_0", 2.8, "multi params p2[0] = 3.0 - 0.2*1.0 = 2.8"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E08 tests passed!", total)
	return nil
}

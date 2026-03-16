package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e03LinearLayerTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "linear-layer",
		Timeout:     30 * time.Second,
		TestFunc:    testE03LinearLayer,
		CompileStep: autoCompileStep("TestE03", "test_e03"),
	}
}

func testE03LinearLayer(harness *test_case_harness.TestCaseHarness) error {
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

	const atol = 1e-2 // wider tolerance for init variance

	// --- Linear basics ---

	linearExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"linear_weight_shape", "2,3", "Linear(3,2).weight shape = [2,3]"},
		{"linear_bias_shape", "2", "Linear(3,2).bias shape = [2]"},
		{"linear_bias_init", "[0.0, 0.0]", "bias initialized to zeros"},
		{"linear_params_count", "2", "Linear(3,2) has 2 parameters (weight+bias)"},
		{"linear_no_bias_params", "1", "Linear(3,2,false) has 1 parameter (weight only)"},
		{"linear_no_bias_null", "true", "Linear(3,2,false).bias is null"},
	}

	for _, tc := range linearExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Linear forward ---

	forwardExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"linear_forward_shape", "2,2", "forward output shape [2,2]"},
		{"linear_forward_toString", "[[11.0, 22.0], [14.0, 25.0]]", "y = x @ W.T + b"},
		{"linear_forward_no_bias", "[[1.0, 2.0]]", "forward without bias"},
	}

	for _, tc := range forwardExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- LeCun init variance (float check) ---

	if err := helpers.AssertFloatResultClose(results, "linear_init_variance", 0.001, atol); err != nil {
		return err
	}
	logger.Successf("✓ LeCun init variance ≈ 1/1000")

	// --- Dropout ---

	dropoutExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"dropout_eval", "[[1.0, 2.0], [3.0, 4.0]]", "dropout eval mode = identity"},
		{"dropout_p0_train", "[[1.0, 2.0], [3.0, 4.0]]", "dropout p=0 train = identity"},
		{"dropout_p1_train", "[[0.0, 0.0], [0.0, 0.0]]", "dropout p=1 train = zeros"},
		{"dropout_shape", "2,2", "dropout preserves shape"},
		{"dropout_no_params", "0", "dropout has no parameters"},
	}

	for _, tc := range dropoutExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Sequential ---

	seqExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"sequential_forward_shape", "1,1", "sequential forward shape [1,1]"},
		{"sequential_forward", "[[3.0]]", "sequential forward chains layers"},
		{"sequential_params_count", "4", "sequential collects 4 parameters"},
		{"sequential_children_count", "2", "sequential has 2 children"},
	}

	for _, tc := range seqExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Train / Eval mode ---

	modeExact := []struct {
		name     string
		expected string
		label    string
	}{
		{"training_default", "true", "training defaults to true"},
		{"eval_sets_false", "false", "eval() sets training=false"},
		{"train_sets_true", "true", "train() sets training=true"},
		{"eval_recursive", "false", "eval() propagates to children"},
		{"train_recursive", "true", "train() propagates to children"},
	}

	for _, tc := range modeExact {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(linearExact) + len(forwardExact) + 1 + len(dropoutExact) + len(seqExact) + len(modeExact)
	logger.Successf("All %d E03 tests passed!", total)
	return nil
}

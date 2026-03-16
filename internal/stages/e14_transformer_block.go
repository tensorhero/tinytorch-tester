package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e14TransformerBlockTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "transformer-block",
		Timeout:     30 * time.Second,
		TestFunc:    testE14TransformerBlock,
		CompileStep: autoCompileStep("TestE14", "test_e14"),
	}
}

func testE14TransformerBlock(harness *test_case_harness.TestCaseHarness) error {
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
		// Part 1: LayerNorm
		{"ln_output_shape", "1,4", "LayerNorm output shape = 1,4"},
		{"ln_params_count", "2", "LayerNorm parameters count = 2"},
		// Part 2: MLP
		{"mlp_output_shape", "1,3,8", "MLP output shape = 1,3,8"},
		{"mlp_params_count", "4", "MLP parameters count = 4"},
		{"mlp_children_count", "2", "MLP children count = 2"},
		// Part 3: TransformerBlock
		{"block_output_shape", "1,3,8", "Block output shape = 1,3,8"},
		{"block_params_count", "16", "Block parameters count = 16"},
		{"block_children_count", "4", "Block children count = 4"},
		{"block_residual_shape", "1,3,8", "Block residual shape preserved = 1,3,8"},
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
		// Part 1: LayerNorm output correctness
		{"ln_output_value_00", -1.341635, "LayerNorm output[0,0] ≈ -1.3416"},
		{"ln_output_value_03", 1.341635, "LayerNorm output[0,3] ≈ 1.3416"},
		{"ln_output_mean", 0.0, "LayerNorm output row mean ≈ 0"},
		// Part 1b: LayerNorm backward
		{"ln_backward_beta_grad_0", 1.0, "LayerNorm beta.grad[0] = 1.0"},
		{"ln_backward_gamma_grad_0", -1.341635, "LayerNorm gamma.grad[0] ≈ -1.3416"},
		// Part 1c: Uniform input edge case
		{"ln_uniform_output_00", 0.0, "LayerNorm uniform input → 0.0"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E14 tests passed!", total)
	return nil
}

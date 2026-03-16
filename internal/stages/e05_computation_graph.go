package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e05ComputationGraphTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "computation-graph",
		Timeout:     30 * time.Second,
		TestFunc:    testE05ComputationGraph,
		CompileStep: autoCompileStep("TestE05", "test_e05"),
	}
}

func testE05ComputationGraph(harness *test_case_harness.TestCaseHarness) error {
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

	// --- Graph recording (exact match) ---

	exactTests := []struct {
		name     string
		expected string
		label    string
	}{
		{"graph_add_has_fn", "true", "add records gradFn"},
		{"graph_sub_has_fn", "true", "sub records gradFn"},
		{"graph_mul_has_fn", "true", "mul records gradFn"},
		{"graph_div_has_fn", "true", "div records gradFn"},
		{"graph_no_grad", "true", "no gradFn when requiresGrad=false"},
		{"graph_requires_grad", "true", "result.requiresGrad propagated"},
		{"graph_inputs_count", "2", "gradFn.inputs() has 2 entries"},
	}

	for _, tc := range exactTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	// --- Forward values + backward gradients (float close) ---

	const atol = 1e-4

	floatTests := []struct {
		name     string
		expected float64
		label    string
	}{
		// Forward values (x=[2,3], y=[4,5])
		{"forward_add_0", 6.0, "add forward: 2+4=6"},
		{"forward_add_1", 8.0, "add forward: 3+5=8"},
		{"forward_sub_0", -2.0, "sub forward: 2-4=-2"},
		{"forward_mul_0", 8.0, "mul forward: 2*4=8"},
		{"forward_div_0", 0.5, "div forward: 2/4=0.5"},

		// AddBackward: grad_a = gradOut, grad_b = gradOut
		{"backward_add_da", 1.0, "add backward da=1"},
		{"backward_add_db", 1.0, "add backward db=1"},

		// SubBackward: grad_a = gradOut, grad_b = -gradOut
		{"backward_sub_da", 1.0, "sub backward da=1"},
		{"backward_sub_db", -1.0, "sub backward db=-1"},

		// MulBackward: grad_a = gradOut*y, grad_b = gradOut*x
		{"backward_mul_da", 4.0, "mul backward da=y[0]=4"},
		{"backward_mul_db", 2.0, "mul backward db=x[0]=2"},

		// DivBackward: grad_a = gradOut/y, grad_b = -gradOut*x/y²
		{"backward_div_da", 0.25, "div backward da=1/4=0.25"},
		{"backward_div_db", -0.125, "div backward db=-2/16=-0.125"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E05 tests passed!", total)
	return nil
}

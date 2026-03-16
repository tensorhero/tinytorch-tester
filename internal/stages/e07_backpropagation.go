package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e07BackpropagationTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "backpropagation",
		Timeout:     30 * time.Second,
		TestFunc:    testE07Backpropagation,
		CompileStep: autoCompileStep("TestE07", "test_e07"),
	}
}

func testE07Backpropagation(harness *test_case_harness.TestCaseHarness) error {
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
		// Topological sort
		{"topo_size", "3", "topo sort has 3 nodes"},
		{"topo_root_last", "true", "root is last in topo order"},
		{"topo_x_before_y", "true", "x comes before y in topo order"},
		{"topo_w_before_y", "true", "w comes before y in topo order"},
		// Multi-op grad existence
		{"multi_x_grad_exists", "true", "x.grad exists after backward"},
		{"multi_w_grad_exists", "true", "w.grad exists after backward"},
		{"multi_b_grad_exists", "true", "b.grad exists after backward"},
		// Grad shapes
		{"multi_x_grad_shape", "2,2", "x.grad shape is [2,2]"},
		{"multi_w_grad_shape", "2,2", "w.grad shape is [2,2]"},
		{"multi_b_grad_shape", "2", "b.grad shape is [2] (broadcast reduced)"},
		// Broadcast grad shape
		{"broadcast_b_grad_shape", "3", "broadcast b.grad shape is [3]"},
		// noGrad
		{"nograd_no_fn", "true", "noGrad result has no gradFn"},
		{"nograd_restored", "true", "gradEnabled restored after noGrad"},
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
		// Simple backward: c = a * b
		{"simple_backward_a_grad", 4.0, "a.grad = b = 4"},
		{"simple_backward_b_grad", 3.0, "b.grad = a = 3"},
		// Multi-op loss value
		{"multi_loss_value", 2.6, "multi-op loss = 2.6"},
		// Multi-op b.grad (broadcast reduced)
		{"multi_b_grad_0", 1.0, "b.grad[0] = 1.0 (relu pass-through)"},
		{"multi_b_grad_1", 0.0, "b.grad[1] = 0.0 (relu blocked)"},
		// Broadcast gradient
		{"broadcast_b_grad_0", 1.0, "broadcast b.grad[0] = 1.0"},
		{"broadcast_b_grad_1", 1.0, "broadcast b.grad[1] = 1.0"},
		{"broadcast_b_grad_2", 1.0, "broadcast b.grad[2] = 1.0"},
		// Gradient accumulation (x + x)
		{"accum_grad_0", 2.0, "accum grad[0] = 2 (used twice)"},
		{"accum_grad_1", 2.0, "accum grad[1] = 2 (used twice)"},
		{"accum_grad_2", 2.0, "accum grad[2] = 2 (used twice)"},
		// noGrad result values
		{"nograd_result_0", 4.0, "noGrad result[0] = 4"},
		{"nograd_result_1", 6.0, "noGrad result[1] = 6"},
		// Finite difference verification: d/dx sum(x^2) = 2x
		{"fd_grad_0", 2.0, "fd grad[0] = 2*1 = 2"},
		{"fd_grad_1", 4.0, "fd grad[1] = 2*2 = 4"},
		{"fd_grad_2", 6.0, "fd grad[2] = 2*3 = 6"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E07 tests passed!", total)
	return nil
}

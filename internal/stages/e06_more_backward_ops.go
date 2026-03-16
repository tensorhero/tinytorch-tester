package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e06MoreBackwardOpsTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "more-backward-ops",
		Timeout:     30 * time.Second,
		TestFunc:    testE06MoreBackwardOps,
		CompileStep: autoCompileStep("TestE06", "test_e06"),
	}
}

func testE06MoreBackwardOps(harness *test_case_harness.TestCaseHarness) error {
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
		{"graph_matmul_has_fn", "true", "matMul records gradFn"},
		{"graph_sum_has_fn", "true", "sum records gradFn"},
		{"graph_mean_has_fn", "true", "mean records gradFn"},
		{"graph_reshape_has_fn", "true", "reshape records gradFn"},
		{"graph_transpose_has_fn", "true", "transpose records gradFn"},
		{"graph_exp_has_fn", "true", "exp records gradFn"},
		{"graph_log_has_fn", "true", "log records gradFn"},
		{"graph_relu_has_fn", "true", "relu records gradFn"},
		{"graph_sigmoid_has_fn", "true", "sigmoid records gradFn"},
		{"graph_tanh_has_fn", "true", "tanh records gradFn"},
		{"graph_gelu_has_fn", "true", "gelu records gradFn"},
		{"graph_ce_has_fn", "true", "crossEntropy records gradFn"},
		{"graph_mse_has_fn", "true", "mse result has gradFn (via Tensor ops)"},
		{"forward_reshape_shape", "3,2", "reshape forward shape"},
		{"forward_transpose_shape", "3,2", "transpose forward shape"},
		{"backward_sum_shape", "2,3", "sum backward shape"},
		{"backward_reshape_shape", "2,3", "reshape backward shape"},
		{"backward_transpose_shape", "2,3", "transpose backward shape"},
		{"forward_ce_shape", "1", "crossEntropy forward shape"},
		{"backward_ce_shape", "2,3", "crossEntropy backward shape"},
		{"forward_mse_shape", "1", "mse forward shape"},
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
		// matMul forward
		{"forward_matmul_00", 19.0, "matMul forward [0,0]=19"},
		{"forward_matmul_11", 50.0, "matMul forward [1,1]=50"},
		// matMul backward
		{"backward_matmul_da_00", 11.0, "matMul backward da[0,0]=11"},
		{"backward_matmul_da_01", 15.0, "matMul backward da[0,1]=15"},
		{"backward_matmul_db_00", 4.0, "matMul backward db[0,0]=4"},
		{"backward_matmul_db_10", 6.0, "matMul backward db[1,0]=6"},
		// sum forward + backward
		{"forward_sum_0", 6.0, "sum forward [0]=6"},
		{"forward_sum_1", 15.0, "sum forward [1]=15"},
		{"backward_sum_00", 1.0, "sum backward [0,0]=1"},
		// mean forward + backward
		{"forward_mean_0", 3.0, "mean forward [0]=3"},
		{"backward_mean_00", 0.5, "mean backward [0,0]=0.5"},
		// transpose forward
		{"forward_transpose_00", 1.0, "transpose forward [0,0]=1"},
		{"forward_transpose_10", 2.0, "transpose forward [1,0]=2"},
		// exp forward + backward
		{"forward_exp_0", 1.0, "exp forward exp(0)=1"},
		{"forward_exp_1", 2.718282, "exp forward exp(1)=e"},
		{"backward_exp_0", 1.0, "exp backward grad*exp(0)=1"},
		{"backward_exp_1", 2.718282, "exp backward grad*exp(1)=e"},
		// log forward + backward
		{"forward_log_0", 0.0, "log forward log(1)=0"},
		{"forward_log_1", 1.0, "log forward log(e)=1"},
		{"backward_log_0", 1.0, "log backward grad/1=1"},
		{"backward_log_1", 0.367879, "log backward grad/e≈0.3679"},
		// relu forward + backward
		{"forward_relu_0", 0.0, "relu forward relu(-1)=0"},
		{"forward_relu_2", 1.0, "relu forward relu(1)=1"},
		{"backward_relu_0", 0.0, "relu backward x=-1 → 0"},
		{"backward_relu_2", 1.0, "relu backward x=1 → 1"},
		// sigmoid forward + backward
		{"forward_sigmoid_1", 0.5, "sigmoid forward sig(0)=0.5"},
		{"backward_sigmoid_1", 0.25, "sigmoid backward sig'(0)=0.25"},
		// tanh forward + backward
		{"forward_tanh_1", 0.0, "tanh forward tanh(0)=0"},
		{"backward_tanh_1", 1.0, "tanh backward tanh'(0)=1"},
		// gelu forward + backward
		{"forward_gelu_1", 0.0, "gelu forward gelu(0)=0"},
		{"backward_gelu_1", 0.5, "gelu backward gelu'(0)=0.5"},
		// crossEntropy backward (row sum ≈ 0)
		{"backward_ce_row_sum", 0.0, "crossEntropy backward row sum≈0"},
		// dropout forward + backward
		{"forward_dropout_0", 2.0, "dropout forward [0]*2=2"},
		{"forward_dropout_1", 0.0, "dropout forward [1]*0=0"},
		{"backward_dropout_0", 2.0, "dropout backward grad*mask=2"},
		{"backward_dropout_1", 0.0, "dropout backward grad*mask=0"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E06 tests passed!", total)
	return nil
}

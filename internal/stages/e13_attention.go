package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e13AttentionTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "attention",
		Timeout:     30 * time.Second,
		TestFunc:    testE13Attention,
		CompileStep: autoCompileStep("TestE13", "test_e13"),
	}
}

func testE13Attention(harness *test_case_harness.TestCaseHarness) error {
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
		// Part 1: Causal Mask
		{"causal_mask_shape", "4,4", "Causal mask shape = 4,4"},
		{"causal_mask_self_visible", "0.000000", "Causal mask: token sees itself (0.0)"},
		{"causal_mask_future_blocked", "1.000000", "Causal mask: future blocked (1.0)"},
		{"causal_mask_past_visible", "0.000000", "Causal mask: past visible (0.0)"},
		{"causal_mask_last_self", "0.000000", "Causal mask: last token sees itself (0.0)"},
		// Part 2: SDPA weight sum
		{"sdpa_weight_sum_shape", "3,2", "SDPA output shape = 3,2"},
		// Part 4: Multi-Head Attention
		{"mha_output_shape", "1,3,8", "MHA output shape = 1,3,8"},
		{"mha_params_count", "8", "MHA parameters count = 8"},
		{"mha_children_count", "4", "MHA children count = 4"},
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
		// Part 2: SDPA weight sum (V=ones → output=1.0 means softmax rows sum to 1)
		{"sdpa_weight_sum_value", 1.0, "SDPA: weight row sum = 1.0 (no mask)"},
		{"sdpa_causal_weight_sum", 1.0, "SDPA: weight row sum = 1.0 (causal)"},
		// Part 3: SDPA correctness — no mask (uniform attention)
		{"sdpa_no_mask_value_00", 5.0, "SDPA no mask: output[0,0] = 5.0 (uniform avg)"},
		{"sdpa_no_mask_value_01", 6.0, "SDPA no mask: output[0,1] = 6.0 (uniform avg)"},
		// Part 3: SDPA correctness — causal mask
		{"sdpa_causal_row0_0", 1.0, "SDPA causal: row 0 only sees itself → 1.0"},
		{"sdpa_causal_row0_1", 2.0, "SDPA causal: row 0 col 1 → 2.0"},
		{"sdpa_causal_row1_0", 3.0, "SDPA causal: row 1 avg(V[0],V[1]) → 3.0"},
		{"sdpa_causal_last_same", 5.0, "SDPA causal: last row = no mask → 5.0"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E13 tests passed!", total)
	return nil
}

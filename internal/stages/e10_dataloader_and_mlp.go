package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e10DataLoaderAndMLPTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "dataloader-and-mlp",
		Timeout:     30 * time.Second,
		TestFunc:    testE10DataLoaderAndMLP,
		CompileStep: autoCompileStep("TestE10", "test_e10"),
	}
}

func testE10DataLoaderAndMLP(harness *test_case_harness.TestCaseHarness) error {
	logger := harness.Logger
	workDir := harness.SubmissionDir
	lang := harness.DetectedLang

	r := runner.Run(workDir, lang.RunCmd, lang.RunArgs...).
		WithTimeout(30 * time.Second).
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
		{"dataset_size", "4", "TensorDataset.size() = 4"},
		{"num_batches_even", "2", "numBatches = 2 (4 samples / batchSize 2)"},
		{"num_batches_uneven", "3", "numBatches = 3 (5 samples / batchSize 2, ceil)"},
		{"batch_count", "2", "DataLoader yields 2 batches"},
		{"batch0_data_rows", "2", "first batch data has 2 rows"},
		{"batch0_data_cols", "2", "first batch data has 2 columns"},
		{"batch0_label_rows", "2", "first batch labels has 2 rows"},
		{"batch0_label_cols", "1", "first batch labels has 1 column"},
		{"total_samples", "4", "all 4 samples iterated"},
		{"last_batch_size", "1", "last batch has 1 sample (5 mod 2)"},
		{"xor_loss_decreases", "true", "XOR training loss decreases over 200 epochs"},
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
		// dataset get
		{"get1_data0", 0.0, "get(1).data[0] = 0.0"},
		{"get1_data1", 1.0, "get(1).data[1] = 1.0"},
		{"get1_label0", 1.0, "get(1).labels[0] = 1.0"},
		{"get2_data0", 1.0, "get(2).data[0] = 1.0"},
		{"get2_data1", 0.0, "get(2).data[1] = 0.0"},
		{"get2_label0", 1.0, "get(2).labels[0] = 1.0"},
		// total coverage
		{"label_sum", 10.0, "sum of all labels = 1+2+3+4 = 10"},
		// no-shuffle order
		{"noshuffle_b0_d00", 10.0, "batch 0 row 0 col 0 = 10.0"},
		{"noshuffle_b0_d10", 30.0, "batch 0 row 1 col 0 = 30.0"},
		{"noshuffle_b1_d00", 50.0, "batch 1 row 0 col 0 = 50.0"},
		{"noshuffle_b1_d10", 70.0, "batch 1 row 1 col 0 = 70.0"},
		// XOR accuracy
		{"xor_accuracy", 1.0, "XOR accuracy = 100% 🎉"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E10 tests passed! 🎉 Milestone 1: MLP trains on XOR!", total)
	return nil
}

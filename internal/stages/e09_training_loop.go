package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e09TrainingLoopTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "training-loop",
		Timeout:     30 * time.Second,
		TestFunc:    testE09TrainingLoop,
		CompileStep: autoCompileStep("TestE09", "test_e09"),
	}
}

func testE09TrainingLoop(harness *test_case_harness.TestCaseHarness) error {
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
		{"train_step_positive", "true", "trainStep returns positive loss"},
		{"train_step_decreasing", "true", "loss decreases over 10 training steps"},
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
		// Cosine schedule
		{"cosine_at_start", 0.1, "cosine lr at step 0 = maxLr"},
		{"cosine_at_middle", 0.0505, "cosine lr at step 50 = midpoint"},
		{"cosine_at_end", 0.001, "cosine lr at step 100 = minLr"},
		{"cosine_at_quarter", 0.085502, "cosine lr at step 25 ≈ 0.0855"},
		// Gradient clipping
		{"clip_returns_norm", 5.0, "clipGradNorm returns original norm = 5.0"},
		{"clip_after_grad0", 1.5, "clipped grad[0] = 3.0 * 2.5/5.0 = 1.5"},
		{"clip_after_grad1", 2.0, "clipped grad[1] = 4.0 * 2.5/5.0 = 2.0"},
		{"clip_no_clip_grad0", 1.0, "grad unchanged when norm < maxNorm"},
		// Accuracy
		{"accuracy_perfect", 1.0, "accuracy = 1.0 when all predictions correct"},
		{"accuracy_partial", 0.5, "accuracy = 0.5 when half predictions correct"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E09 tests passed!", total)
	return nil
}

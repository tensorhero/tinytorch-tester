package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e15GptAndGenerateTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "gpt-and-generate",
		Timeout:     60 * time.Second,
		TestFunc:    testE15GptAndGenerate,
		CompileStep: autoCompileStep("TestE15", "test_e15"),
	}
}

func testE15GptAndGenerate(harness *test_case_harness.TestCaseHarness) error {
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
		// Part 1: Forward pass
		{"forward_shape", "4,12", "Forward shape = [4, 12]"},
		{"forward_ndim", "2", "Forward ndim = 2"},
		{"forward_single_shape", "1,12", "Forward single token shape = [1, 12]"},
		// Part 2: Parameters and children
		{"params_positive", "true", "Parameters count > 0"},
		{"children_count", "2", "Children count = 2 (numLayers)"},
		{"params_count", "37", "Total parameters count = 37"},
		// Part 3: Generate
		{"generate_length", "8", "Generate length = 8 (prompt 3 + generated 5)"},
		{"generate_deterministic", "true", "Greedy generation is deterministic"},
		{"generate_starts_with_prompt", "true", "Generated text starts with prompt"},
		{"generate_temp_length", "5", "Generate with temperature: length = 5"},
		{"generate_short_prompt_length", "5", "Generate with short prompt: length = 5"},
		{"generate_valid_vocab", "true", "All generated chars in vocabulary"},
	}

	for _, tc := range exactTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests)
	logger.Successf("All %d E15 tests passed!", total)
	return nil
}

package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e12EmbeddingsTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "embeddings",
		Timeout:     30 * time.Second,
		TestFunc:    testE12Embeddings,
		CompileStep: autoCompileStep("TestE12", "test_e12"),
	}
}

func testE12Embeddings(harness *test_case_harness.TestCaseHarness) error {
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
		// Embedding lookup & properties
		{"embedding_shape", "3,3", "Embedding forward([0,2,3]) shape = 3,3"},
		{"embedding_lookup_r0c0", "7.000000", "Embedding forward([2])[0,0] = 7.0"},
		{"embedding_lookup_r0c2", "9.000000", "Embedding forward([2])[0,2] = 9.0"},
		{"embedding_repeated", "4.000000", "Embedding forward([1,1]) rows identical"},
		{"embedding_params_count", "1", "Embedding parameters() count = 1"},
		{"embedding_weight_shape", "4,3", "Embedding weight shape = 4,3"},
		// Embedding backward
		{"embedding_grad_shape", "4,3", "Embedding gradient shape = 4,3"},
		{"embedding_grad_zero", "0.000000", "Embedding grad[0,0] = 0.0 (not looked up)"},
		// Sinusoidal PE
		{"sinusoidal_shape", "5,8", "Sinusoidal PE output shape = 5,8"},
		{"sinusoidal_no_params", "0", "Sinusoidal PE has 0 parameters"},
		// Learned PE
		{"learned_shape", "5,8", "Learned PE output shape = 5,8"},
		{"learned_params_count", "1", "Learned PE has 1 parameter"},
		{"learned_weight_shape", "100,8", "Learned PE weight shape = 100,8"},
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
		// Embedding backward
		{"embedding_grad_single", 1.0, "Embedding grad[2,0] = 1.0 (single lookup)"},
		{"embedding_grad_accumulate", 2.0, "Embedding grad[1,0] = 2.0 (accumulated)"},
		// Sinusoidal PE values (input=ones, so output = 1.0 + PE)
		{"sinusoidal_pe_0_0", 1.0, "Sinusoidal PE: 1.0 + sin(0) = 1.0"},
		{"sinusoidal_pe_0_1", 2.0, "Sinusoidal PE: 1.0 + cos(0) = 2.0"},
		{"sinusoidal_pe_1_0", 1.841471, "Sinusoidal PE: 1.0 + sin(1) ≈ 1.841471"},
	}

	for _, tc := range floatTests {
		if err := helpers.AssertFloatResultClose(results, tc.name, tc.expected, atol); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests) + len(floatTests)
	logger.Successf("All %d E12 tests passed!", total)
	return nil
}

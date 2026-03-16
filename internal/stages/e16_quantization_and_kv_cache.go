package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e16QuantizationAndKVCacheTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "quantization-and-kv-cache",
		Timeout:     30 * time.Second,
		TestFunc:    testE16QuantizationAndKVCache,
		CompileStep: autoCompileStep("TestE16", "test_e16"),
	}
}

func testE16QuantizationAndKVCache(harness *test_case_harness.TestCaseHarness) error {
	logger := harness.Logger
	workDir := harness.SubmissionDir
	lang := harness.DetectedLang

	r := runner.Run(workDir, lang.RunCmd, lang.RunArgs...).
		WithTimeout(15 * time.Second).
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
		// Part 1: Quantizer
		{"quantize_shape", "3,4", "Quantize preserves shape (3,4)"},
		{"quantize_range_min", "true", "Quantized values >= 0"},
		{"quantize_range_max", "true", "Quantized values <= 255"},
		{"dequantize_shape", "3,4", "Dequantize shape matches original (3,4)"},
		{"dequantize_close", "true", "Roundtrip error < 1% of range"},
		{"quantized_matmul_shape", "2,2", "Quantized matmul shape (2,2)"},
		{"quantized_matmul_close", "true", "Quantized matmul error < 5%"},
		// Part 2: KVCache
		{"kv_cache_initial_len", "0", "Initial cache length = 0"},
		{"kv_cache_update_len", "3", "Cache length after first update = 3"},
		{"kv_cache_keys_shape", "2,3,4", "Keys shape after first update (2,3,4)"},
		{"kv_cache_multi_update_len", "5", "Cache length after two updates = 5"},
		{"kv_cache_multi_update_shape", "2,5,4", "Keys shape after two updates (2,5,4)"},
		{"kv_cache_values_correct", "true", "Cached values preserved correctly"},
		{"kv_cache_reset_len", "0", "Cache length after reset = 0"},
	}

	for _, tc := range exactTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests)
	logger.Successf("All %d E16 tests passed!", total)
	return nil
}

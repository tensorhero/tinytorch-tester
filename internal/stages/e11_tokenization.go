package stages

import (
	"fmt"
	"time"

	"github.com/tensorhero/tester-utils/runner"
	"github.com/tensorhero/tester-utils/test_case_harness"
	"github.com/tensorhero/tester-utils/tester_definition"
	"github.com/tensorhero/tinytorch-tester/internal/helpers"
)

func e11TokenizationTestCase() tester_definition.TestCase {
	return tester_definition.TestCase{
		Slug:        "tokenization",
		Timeout:     30 * time.Second,
		TestFunc:    testE11Tokenization,
		CompileStep: autoCompileStep("TestE11", "test_e11"),
	}
}

func testE11Tokenization(harness *test_case_harness.TestCaseHarness) error {
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
		// CharTokenizer
		{"char_vocab_size", "4", "CharTokenizer vocabSize('hello') = 4"},
		{"char_encode", "[1,0,2,2,3]", "CharTokenizer encode('hello') = [1,0,2,2,3]"},
		{"char_decode", "hello", "CharTokenizer decode([1,0,2,2,3]) = 'hello'"},
		{"char_roundtrip", "the quick brown fox", "CharTokenizer roundtrip: decode(encode(text)) == text"},
		{"char_space_vocab_size", "3", "CharTokenizer vocabSize('a b') = 3 (includes space)"},
		{"char_space_encode", "[1,0,2]", "CharTokenizer encode('a b') = [1,0,2]"},
		{"char_encode_length", "6", "CharTokenizer encode length = input length"},
		// BPETokenizer
		{"bpe_vocab_size", "6", "BPETokenizer vocabSize = 6 after training"},
		{"bpe_encode", "[5,1,3,5,1,0,2]", "BPETokenizer encode('aaabdaaabac') = [5,1,3,5,1,0,2]"},
		{"bpe_decode", "aaabdaaabac", "BPETokenizer decode → 'aaabdaaabac'"},
		{"bpe_roundtrip", "aaabdaaabac", "BPETokenizer roundtrip: decode(encode(text)) == text"},
		{"bpe_compression", "true", "BPE produces shorter sequences than CharTokenizer"},
		{"bpe_no_merge_size", "4", "BPE with vocabSize=base → no merges"},
		{"bpe_no_merge_len", "11", "BPE no merge → same length as char encoding"},
		{"bpe_long_roundtrip", "abababababcdcdcdcd", "BPE roundtrip on longer text"},
		{"bpe_long_vocab_size", "8", "BPE long text reaches target vocabSize=8"},
		{"bpe_unseen_roundtrip", "abab", "BPE roundtrip on unseen text ordering"},
	}

	for _, tc := range exactTests {
		if err := helpers.AssertEqual(results, tc.name, tc.expected); err != nil {
			return err
		}
		logger.Successf("✓ %s", tc.label)
	}

	total := len(exactTests)
	logger.Successf("All %d E11 tests passed!", total)
	return nil
}

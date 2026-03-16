package helpers

import (
	"fmt"
	"strings"
)

// ParseStructuredOutput parses TEST:name / RESULT:value pairs from stdout.
// Returns a map from test name to result string.
func ParseStructuredOutput(output string) map[string]string {
	results := make(map[string]string)
	lines := strings.Split(output, "\n")
	var currentTest string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "TEST:") {
			currentTest = strings.TrimPrefix(line, "TEST:")
		} else if strings.HasPrefix(line, "RESULT:") && currentTest != "" {
			results[currentTest] = strings.TrimPrefix(line, "RESULT:")
			currentTest = ""
		}
	}
	return results
}

// AssertEqual checks that the structured output for testName matches expected exactly.
func AssertEqual(results map[string]string, testName, expected string) error {
	actual, ok := results[testName]
	if !ok {
		return fmt.Errorf("test %q: no output found (test driver may have crashed before this test)", testName)
	}
	if actual != expected {
		return fmt.Errorf("test %q: expected %q, got %q", testName, expected, actual)
	}
	return nil
}

package stages

import (
	"github.com/tensorhero/tester-utils/tester_definition"
)

// GetDefinition returns the TesterDefinition for the tinytorch course.
func GetDefinition() tester_definition.TesterDefinition {
	return tester_definition.TesterDefinition{
		TestCases: []tester_definition.TestCase{
			// Phase 1: Forward Pass
			e01TensorClassTestCase(),
			e02ActivationsTestCase(),
			e03LinearLayerTestCase(),
			e04LossFunctionsTestCase(),
			// Phase 2: Autograd
			e05ComputationGraphTestCase(),
			e06MoreBackwardOpsTestCase(),
			e07BackpropagationTestCase(),
			// Phase 3: Training
			e08OptimizersTestCase(),
			e09TrainingLoopTestCase(),
			e10DataLoaderAndMLPTestCase(),
			// Phase 4: Transformer
			e11TokenizationTestCase(),
			e12EmbeddingsTestCase(),
			e13AttentionTestCase(),
			e14TransformerBlockTestCase(),
			e15GptAndGenerateTestCase(),
			// Phase 5: Optimization
			e16QuantizationAndKVCacheTestCase(),
			e17ProfilingAndCompressionTestCase(),
		},
	}
}

// javaRule creates a LanguageRule for Java auto-detection.
// testDriver is the class name (e.g. "TestE01").
func javaRule(testDriver string) tester_definition.LanguageRule {
	return tester_definition.LanguageRule{
		DetectFile: "src/main/java/dev/tensorhero/tinytorch/Tensor.java",
		Language:   "java",
		Source:     "src/main/java/dev/tensorhero/tinytorch/Tensor.java",
		Flags: []string{
			"-encoding", "UTF-8",
			"src/main/java/dev/tensorhero/tinynum/NDArray.java",
			"src/main/java/dev/tensorhero/tinynum/Slice.java",
			"src/main/java/dev/tensorhero/tinynum/DType.java",
			"src/main/java/dev/tensorhero/tinytorch/Function.java",
			"src/main/java/dev/tensorhero/tinytorch/Activations.java",
			"src/main/java/dev/tensorhero/tinytorch/Layer.java",
			"src/main/java/dev/tensorhero/tinytorch/Linear.java",
			"src/main/java/dev/tensorhero/tinytorch/Dropout.java",
			"src/main/java/dev/tensorhero/tinytorch/Sequential.java",
			"src/main/java/dev/tensorhero/tinytorch/Losses.java",
			"src/main/java/dev/tensorhero/tinytorch/AddBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/SubBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/MulBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/DivBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/MatMulBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/SumBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/MeanBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/ExpBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/LogBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/ReshapeBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/TransposeBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/ReLUBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/SigmoidBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/TanhBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/GELUBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/CrossEntropyBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/DropoutBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/Optimizer.java",
			"src/main/java/dev/tensorhero/tinytorch/SGD.java",
			"src/main/java/dev/tensorhero/tinytorch/Adam.java",
			"src/main/java/dev/tensorhero/tinytorch/AdamW.java",
			"src/main/java/dev/tensorhero/tinytorch/Trainer.java",
			"src/main/java/dev/tensorhero/tinytorch/DataLoader.java",
			"src/main/java/dev/tensorhero/tinytorch/Tokenizer.java",
			"src/main/java/dev/tensorhero/tinytorch/Embedding.java",
			"src/main/java/dev/tensorhero/tinytorch/EmbeddingBackward.java",
			"src/main/java/dev/tensorhero/tinytorch/PositionalEncoding.java",
			"src/main/java/dev/tensorhero/tinytorch/Attention.java",
			"src/main/java/dev/tensorhero/tinytorch/TransformerBlock.java",
			"src/main/java/dev/tensorhero/tinytorch/GPT.java",
			"src/main/java/dev/tensorhero/tinytorch/Quantizer.java",
			"src/main/java/dev/tensorhero/tinytorch/KVCache.java",
			"src/main/java/dev/tensorhero/tinytorch/Profiler.java",
			"src/main/java/dev/tensorhero/tinytorch/Pruner.java",
			"tests/" + testDriver + ".java",
		},
		RunCmd:  "java",
		RunArgs: []string{"-cp", ".", testDriver},
	}
}

// pythonRule creates a LanguageRule for Python auto-detection.
// testDriver is the module name without extension (e.g. "test_e01").
func pythonRule(testDriver string) tester_definition.LanguageRule {
	return tester_definition.LanguageRule{
		DetectFile: "tinytorch/tensor.py",
		Language:   "python",
		Source:     "tinytorch/tensor.py",
		RunCmd:     "python3",
		RunArgs:    []string{"tests/" + testDriver + ".py"},
	}
}

// autoCompileStep returns a CompileStep with auto-detection for Java/Python.
func autoCompileStep(javaDriver, pythonDriver string) *tester_definition.CompileStep {
	return &tester_definition.CompileStep{
		Language: "auto",
		AutoDetect: []tester_definition.LanguageRule{
			javaRule(javaDriver),
			pythonRule(pythonDriver),
		},
	}
}

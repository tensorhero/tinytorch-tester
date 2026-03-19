#!/bin/bash
# 批量测试所有 stage 的 solution（Java + Python）
# 用法: ./scripts/test-all-solutions.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTER_DIR="$(dirname "$SCRIPT_DIR")"
SOLUTION_DIR="${TESTER_DIR}/../tinytorch-solution"
STARTER_DIR="${TESTER_DIR}/../tinytorch-starter"

# 构建 tester
cd "$TESTER_DIR"
go build -o tinytorch-tester .

# Stage 列表（按课程顺序）
STAGES=(
    "tensor-class"
    "activations"
    "linear-layer"
    "loss-functions"
    "computation-graph"
    "more-backward-ops"
    "backpropagation"
    "optimizers"
    "training-loop"
    "dataloader-and-mlp"
)

# 语言列表
LANGUAGES=("java" "python")

PASSED=0
FAILED=0
SKIPPED=0
TOTAL_TIME=0

echo "=========================================="
echo "  TinyTorch Solution Tester"
echo "=========================================="
echo ""

for lang in "${LANGUAGES[@]}"; do
    echo "--- Language: ${lang} ---"
    echo ""

    sol_dir="${SOLUTION_DIR}/${lang}"

    if [ ! -d "$sol_dir" ]; then
        echo "⏭️  [${lang}] SKIPPED - solution directory not found"
        ((SKIPPED += ${#STAGES[@]}))
        echo ""
        continue
    fi

    # Ensure test drivers are present in solution dir
    if [ "$lang" = "java" ]; then
        mkdir -p "${sol_dir}/tests"
        cp -f "${STARTER_DIR}/java/tests/"*.java "${sol_dir}/tests/" 2>/dev/null || true
    elif [ "$lang" = "python" ]; then
        mkdir -p "${sol_dir}/tests"
        cp -f "${STARTER_DIR}/python/tests/"*.py "${sol_dir}/tests/" 2>/dev/null || true
    fi

    for stage in "${STAGES[@]}"; do
        printf "🧪 [%-20s %6s] Testing... " "$stage" "$lang"

        start_time=$(python3 -c 'import time; print(time.time())')

        if ./tinytorch-tester -d="$sol_dir" -s="$stage" > /dev/null 2>&1; then
            end_time=$(python3 -c 'import time; print(time.time())')
            elapsed=$(python3 -c "print(f'{$end_time - $start_time:.2f}')")
            echo "✅ PASSED (${elapsed}s)"
            ((PASSED++))
        else
            end_time=$(python3 -c 'import time; print(time.time())')
            elapsed=$(python3 -c "print(f'{$end_time - $start_time:.2f}')")
            echo "❌ FAILED (${elapsed}s)"
            ((FAILED++))
        fi

        TOTAL_TIME=$(python3 -c "print(f'{$TOTAL_TIME + $elapsed:.2f}')")
    done

    echo ""
done

echo "=========================================="
echo "  Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "  Total time: ${TOTAL_TIME}s"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi

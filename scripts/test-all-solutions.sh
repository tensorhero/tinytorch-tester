#!/bin/bash
# 批量测试所有 stage 的 solution（Java + Python）
# 用法: ./scripts/test-all-solutions.sh
#
# 分支模型：solution 仓库每种语言一个分支（java / python），
# 脚本通过 git worktree 将各分支 checkout 到临时目录中测试。

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

    # 使用 git worktree 将语言分支 checkout 到临时目录
    worktree_dir="${SOLUTION_DIR}/.worktree-${lang}"
    if [ -d "$worktree_dir" ]; then
        git -C "$SOLUTION_DIR" worktree remove --force "$worktree_dir" 2>/dev/null || rm -rf "$worktree_dir"
    fi
    git -C "$SOLUTION_DIR" worktree add "$worktree_dir" "$lang" 2>/dev/null

    sol_dir="$worktree_dir"

    if [ ! -d "$sol_dir" ]; then
        echo "⏭️  [${lang}] SKIPPED - branch not found"
        ((SKIPPED += ${#STAGES[@]}))
        echo ""
        continue
    fi

    # Ensure test drivers are present in solution dir (copy from starter branch)
    starter_worktree="${STARTER_DIR}/.worktree-${lang}"
    if [ -d "$starter_worktree" ]; then
        git -C "$STARTER_DIR" worktree remove --force "$starter_worktree" 2>/dev/null || rm -rf "$starter_worktree"
    fi
    git -C "$STARTER_DIR" worktree add "$starter_worktree" "$lang" 2>/dev/null

    if [ "$lang" = "java" ]; then
        mkdir -p "${sol_dir}/tests"
        cp -f "${starter_worktree}/tests/"*.java "${sol_dir}/tests/" 2>/dev/null || true
    elif [ "$lang" = "python" ]; then
        mkdir -p "${sol_dir}/tests"
        cp -f "${starter_worktree}/tests/"*.py "${sol_dir}/tests/" 2>/dev/null || true
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

# Cleanup worktrees
for lang in "${LANGUAGES[@]}"; do
    git -C "$SOLUTION_DIR" worktree remove --force "${SOLUTION_DIR}/.worktree-${lang}" 2>/dev/null || true
    git -C "$STARTER_DIR" worktree remove --force "${STARTER_DIR}/.worktree-${lang}" 2>/dev/null || true
done

echo "=========================================="
echo "  Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "  Total time: ${TOTAL_TIME}s"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi

#!/usr/bin/env bash
set -e

# ===== 用户手动控制区（只改这里） =====
TAG_NAME="尝试对meld进行调参前"   
# 示例：
# TAG_NAME="backup-2025-12-23"
# TAG_NAME="pre-ijcai-2026"
# 不想打 tag 就留空
# ======================================

cd "$(dirname "$0")"

echo "[backup] collecting .py and .sh files..."

FILES=$(find . -type f \( -name "*.py" -o -name "*.sh" \))

if [ -z "$FILES" ]; then
    echo "[backup] no .py or .sh files found"
    exit 0
fi

git add $FILES

if git diff --cached --quiet; then
    echo "[backup] no changes to commit"
    exit 0
fi

COMMIT_MSG="backup $(date '+%Y-%m-%d %H:%M:%S')"

echo "[backup] committing..."
git commit -m "$COMMIT_MSG"

# ===== 可选：打 tag =====
if [ -n "$TAG_NAME" ]; then
    if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
        echo "[backup] tag '$TAG_NAME' already exists, skip tagging"
    else
        echo "[backup] tagging commit as '$TAG_NAME'"
        git tag -a "$TAG_NAME" -m "$COMMIT_MSG"
    fi
fi
# =======================

echo "[backup] pushing commits..."
git push origin main

if [ -n "$TAG_NAME" ]; then
    echo "[backup] pushing tag '$TAG_NAME'..."
    git push origin "$TAG_NAME"
fi

echo "[backup] done"

#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "[backup] collecting .py and .sh files..."

# 只 add .py 和 .sh
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

echo "[backup] committing..."
git commit -m "backup $(date '+%Y-%m-%d %H:%M:%S')"

echo "[backup] pushing..."
git push origin main

echo "[backup] done"

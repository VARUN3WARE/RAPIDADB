#!/bin/bash
# Format C++/CUDA and Python code

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Formatting C++/CUDA files..."
find "$ROOT_DIR/csrc" -name "*.cpp" -o -name "*.cu" -o -name "*.h" | \
    xargs clang-format -i --style=file 2>/dev/null || echo "  clang-format not found, skipping"

echo "Formatting Python files..."
black "$ROOT_DIR/python" "$ROOT_DIR/tests/python" "$ROOT_DIR/benchmarks" "$ROOT_DIR/examples" \
    --line-length 100 2>/dev/null || echo "  black not found, skipping"

isort "$ROOT_DIR/python" "$ROOT_DIR/tests/python" "$ROOT_DIR/benchmarks" "$ROOT_DIR/examples" \
    --profile black --line-length 100 2>/dev/null || echo "  isort not found, skipping"

echo "Done!"

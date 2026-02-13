#!/bin/bash
# Profile RapidaDB kernels with NVIDIA Nsight tools

set -e

SCRIPT=${1:-"benchmarks/bench_distance.py"}
OUTPUT_DIR="profiles"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== RapidaDB Profiling ==="
echo "Script: $SCRIPT"
echo "Output: $OUTPUT_DIR/"

# System-level profiling
echo ""
echo "--- Nsight Systems ---"
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="$OUTPUT_DIR/nsys_${TIMESTAMP}" \
    --force-overwrite=true \
    python "$SCRIPT" 2>/dev/null || echo "nsys not available"

# Kernel-level profiling
echo ""
echo "--- Nsight Compute ---"
ncu --set full \
    --target-processes all \
    --output="$OUTPUT_DIR/ncu_${TIMESTAMP}" \
    python "$SCRIPT" 2>/dev/null || echo "ncu not available"

echo ""
echo "Profiling complete. Results in $OUTPUT_DIR/"

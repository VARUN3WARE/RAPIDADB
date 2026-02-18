#!/bin/bash
# Profile RapidaDB kernels with Nsight Compute

set -e

echo "Profiling RapidaDB with Nsight Compute..."
echo "=========================================="

# Profile L2 distance kernel
echo ""
echo "1. Profiling L2 distance kernel (768D)..."
ncu --set full --target-processes all \
    --kernel-regex l2_distance \
    --launch-count 1 \
    python -c "
import torch
import rapidadb._C as _C
queries = torch.randn(32, 768, device='cuda')
database = torch.randn(100000, 768, device='cuda')
_C.l2_distance(queries, database)
" > nsight_l2_distance.txt 2>&1

echo "Results saved to: nsight_l2_distance.txt"

# Profile top-k kernel
echo ""
echo "2. Profiling top-k kernel..."
ncu --set full --target-processes all \
    --kernel-regex topk \
    --launch-count 1 \
    python -c "
import torch
import rapidadb._C as _C
distances = torch.randn(32, 100000, device='cuda')
_C.topk(distances, 10, largest=False)
" > nsight_topk.txt 2>&1

echo "Results saved to: nsight_topk.txt"

echo ""
echo "✅ Profiling complete!"
echo "View detailed metrics in nsight_*.txt files"

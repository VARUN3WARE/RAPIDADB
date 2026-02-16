#!/bin/bash
# Quick benchmark runner for RapidaDB

set -e

echo " RapidaDB Benchmark Suite"
echo "()()()()()()()()()()()()()()()"
echo ""

# Activate virtual environment
source ../.venv/bin/activate

# Default parameters
N=${N:-100000}
DIM=${DIM:-768}
K=${K:-10}
QUERIES=${QUERIES:-1000}

echo "Configuration:"
echo "  Database size: $N vectors"
echo "  Dimensions: $DIM"
echo "  Top-k: $K"
echo "  Queries: $QUERIES"
echo ""

# Quick test
if [ "$1" == "quick" ]; then
    echo "⚡ Running quick test (10K vectors)..."
    python src/bench_compare.py --n 10000 --dim 768 --k 10 --n-queries 100
    exit 0
fi

# Full comparison
if [ "$1" == "full" ]; then
    echo "🔍 Running full comparison vs competitors..."
    python src/bench_full_comparison.py --n $N --dim $DIM --k $K --n-queries $QUERIES --output benchmark_results_latest.json
    exit 0
fi

# Distance kernel benchmark
if [ "$1" == "distance" ]; then
    echo " Running distance kernel benchmark..."
    python src/bench_distance.py --n $N --dim $DIM --batch 1
    python src/bench_distance.py --n $N --dim $DIM --batch 32
    python src/bench_distance.py --n $N --dim $DIM --batch 128
    exit 0
fi

# Scale test
if [ "$1" == "scale" ]; then
    echo "📈 Running scale test..."
    for n in 10000 50000 100000 500000 1000000; do
        echo ""
        echo "Testing with $n vectors..."
        python src/bench_full_comparison.py --n $n --dim 768 --k 10 --n-queries 500 --output benchmark_${n}.json
    done
    exit 0
fi

# Default: show help
echo "Usage: ./run_benchmarks.sh [command]"
echo ""
echo "Commands:"
echo "  quick      - Quick test with 10K vectors"
echo "  full       - Full comparison vs all competitors"
echo "  distance   - Benchmark distance kernels only"
echo "  scale      - Scale test (10K to 1M vectors)"
echo ""
echo "Environment variables:"
echo "  N=<num>       - Number of database vectors (default: 100000)"
echo "  DIM=<num>     - Vector dimensions (default: 768)"
echo "  K=<num>       - Top-k results (default: 10)"
echo "  QUERIES=<num> - Number of queries (default: 1000)"
echo ""
echo "Examples:"
echo "  ./run_benchmarks.sh quick"
echo "  ./run_benchmarks.sh full"
echo "  N=1000000 ./run_benchmarks.sh full"

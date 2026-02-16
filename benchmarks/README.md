# RapidaDB Benchmarks

Quick reference for benchmarking RapidaDB against competitors.

## Quick Start

### Install Dependencies

```bash
cd benchmarks
source ../.venv/bin/activate
pip install -r requirements.txt
```

### Run Benchmarks

**Quick test (10K vectors, ~30s):**

```bash
./run_benchmarks.sh quick
```

**Standard test (100K vectors, ~2min):**

```bash
./run_benchmarks.sh standard
```

**All benchmarks:**

```bash
./run_benchmarks.sh all
```

### Manual Runs

```bash
cd src

# Quick comparison
python bench_compare.py --n 10000 --dim 768 --k 10 --n-queries 500

# Full comparison (all databases)
python bench_full_comparison.py --n 100000 --dim 768 --k 10 --n-queries 1000

# Distance kernel only
python bench_distance.py --n 100000 --dim 768 --batch 32

# Profile kernels
python profile_kernels.py --all
```

## Current Status

**100K vectors × 768D benchmark:**

| Database      | Throughput @ batch=128 | Latency (p50) | Recall@10 |
| ------------- | ---------------------- | ------------- | --------- |
| **RapidaDB**  | 379 QPS                | 6.91ms        | 100%      |
| PyTorch GPU   | **19,951 QPS**         | **1.78ms**    | 100%      |
| Hnswlib (CPU) | 8,988 QPS              | 0.32ms        | 4.3%      |
| Annoy (CPU)   | 6,051 QPS              | 0.17ms        | 0.9%      |
| FAISS (CPU)   | 145 QPS                | 14.24ms       | 100%      |

**Critical Issue:** RapidaDB is 52x slower than PyTorch baseline - needs kernel optimization!

## Structure

```
benchmarks/
├── README.md                    ← Quick reference (this file)
├── BENCHMARKING_REPORT.md       ← Full documentation
├── requirements.txt             ← Python dependencies
├── run_benchmarks.sh            ← Automated runner
├── src/                         ← Benchmark scripts
│   ├── bench_compare.py         ← Quick comparison
│   ├── bench_full_comparison.py ← Full suite
│   ├── bench_competitors.py     ← Competitor wrappers
│   ├── bench_distance.py        ← Distance kernel benchmark
│   ├── datasets.py              ← Dataset loaders
│   └── profile_kernels.py       ← Profiling tools
└── results/                     ← Benchmark outputs (JSON)
    ├── benchmark_results_50k.json
    └── benchmark_results_100k.json
```

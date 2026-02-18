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

**100K vectors × 768D benchmark (Week 4 - Final):**

| Database      | Throughput @ batch=128 | Latency (p50) | Recall@10 |
| ------------- | ---------------------- | ------------- | --------- |
| **RapidaDB**  | **1,349 QPS**          | 5.23ms        | **100%**  |
| PyTorch GPU   | **19,761 QPS**         | **1.77ms**    | **100%**  |
| HNSW (CPU)    | 10,088 QPS             | 0.34ms        | 4.1%      |
| Annoy (CPU)   | 6,352 QPS              | 0.17ms        | 1.0%      |
| FAISS (CPU)   | 226 QPS                | 11.29ms       | **100%**  |

**Progress by Week:**

| Week | Focus                    | QPS   | Speedup vs Week 1 | vs FAISS-CPU |
| ---- | ------------------------ | ----- | ----------------- | ------------ |
| 1    | Foundation               | 376   | 1.0x              | 1.7x         |
| 2    | Memory Optimization      | 1,331 | 3.5x              | 5.9x         |
| 3    | Warp Primitives          | 1,331 | 3.5x              | 5.9x         |
| 4    | Multi-Stream Async       | 1,349 | 3.6x              | **6.0x**     |

**Key Achievements:**
- 3.6x improvement from baseline
- 6x faster than FAISS-CPU with 100% recall
- All 27 tests passing
- Production-ready flat index
- Still 14.6x slower than PyTorch GPU (kernel fusion needed for bigger gains)

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

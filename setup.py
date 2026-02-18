import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ─── Absolute include dir (setuptools requires only sources to be relative) ───
_ROOT = os.path.dirname(os.path.abspath(__file__))
_INCLUDE_DIR = os.path.join(_ROOT, "csrc", "include")

# ─── Source Files (relative paths) ────────────────────────────
kernel_sources = [
    os.path.join("csrc", "kernels", "distance.cu"),
    os.path.join("csrc", "kernels", "topk.cu"),
    os.path.join("csrc", "kernels", "topk_warp.cu"),
]

index_sources = [
    os.path.join("csrc", "index", "flat_index.cpp"),
]

core_sources = [
    os.path.join("csrc", "core", "stream_pool.cpp"),
]

binding_sources = [
    os.path.join("csrc", "bindings", "torch_extension.cpp"),
]

all_sources = kernel_sources + index_sources + core_sources + binding_sources

# ─── Compiler Flags ──────────────────────────────────────────
cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-std=c++17",
]

# Debug mode
if os.environ.get("RAPIDADB_DEBUG", "0") == "1":
    cxx_flags = ["-g", "-O0", "-std=c++17"]
    nvcc_flags = ["-G", "-g", "-lineinfo", "-std=c++17"]

# ─── Extension Module ────────────────────────────────────────
ext_modules = [
    CUDAExtension(
        name="rapidadb._C",
        sources=all_sources,
        include_dirs=[_INCLUDE_DIR],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]

# ─── Setup ────────────────────────────────────────────────────
setup(
    name="rapidadb",
    version="0.1.0",
    description="GPU-Native Vector Database for Production RAG & Multi-Agent Systems",
    author="RapidaDB Contributors",
    python_requires=">=3.10",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-benchmark",
            "black",
            "isort",
            "flake8",
        ],
        "bench": [
            "faiss-gpu",
            "matplotlib",
            "pandas",
        ],
    },
)

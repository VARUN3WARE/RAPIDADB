cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(RapidaDB VERSION 0.1.0 LANGUAGES CXX CUDA)

# ─── C++ Standard ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── Build Type ───────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─── CUDA Architecture ───────────────────────────────────────
# Default to common architectures; override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# ─── Find Packages ───────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# ─── Compiler Flags ──────────────────────────────────────────
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
endif()

# ─── Include Directories ─────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/csrc/include
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# ─── Source Files ─────────────────────────────────────────────
set(KERNEL_SOURCES
    csrc/kernels/distance.cu
    csrc/kernels/topk.cu
    csrc/kernels/topk_warp.cu
)

set(INDEX_SOURCES
    csrc/index/flat_index.cpp
)

set(BINDING_SOURCES
    csrc/bindings/torch_extension.cpp
)

# ─── RapidaDB Core Library ───────────────────────────────────
add_library(rapidadb_core STATIC
    ${KERNEL_SOURCES}
    ${INDEX_SOURCES}
)

target_link_libraries(rapidadb_core PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

set_target_properties(rapidadb_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ─── PyTorch Extension (Shared Library) ──────────────────────
add_library(rapidadb_C SHARED
    ${BINDING_SOURCES}
)

target_link_libraries(rapidadb_C PRIVATE
    rapidadb_core
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

set_target_properties(rapidadb_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "rapidadb_C"
)

# ─── Tests (optional) ────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ─── Install ─────────────────────────────────────────────────
install(TARGETS rapidadb_core rapidadb_C
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY csrc/include/rapidadb
    DESTINATION include
)

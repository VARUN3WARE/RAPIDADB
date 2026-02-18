// ─────────────────────────────────────────────────────────────
// RapidaDB — PyTorch C++ Extension Bindings
//
// Exposes CUDA kernels and index classes to Python via pybind11.
// This is the single entry point loaded by `torch.utils.cpp_extension`.
// ─────────────────────────────────────────────────────────────

#include <torch/extension.h>
#include "rapidadb/core/types.h"
#include "rapidadb/kernels/distance.h"
#include "rapidadb/kernels/topk.h"
#include "rapidadb/index/flat_index.h"

using namespace rapidadb;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RapidaDB: GPU-Native Vector Database";

    // ─── Enums ──────────────────────────────────────────────
    py::enum_<Metric>(m, "Metric")
        .value("COSINE", Metric::COSINE)
        .value("L2", Metric::L2)
        .value("DOT_PRODUCT", Metric::DOT_PRODUCT);

    py::enum_<Precision>(m, "Precision")
        .value("FP32", Precision::FP32)
        .value("FP16", Precision::FP16)
        .value("BF16", Precision::BF16)
        .value("INT8", Precision::INT8);

    // ─── Distance Kernels ───────────────────────────────────
    // Auto-selects optimized kernels when dim <= 128
    
    m.def("cosine_similarity", &cosine_similarity_cuda,
          "Cosine similarity (CUDA)",
          py::arg("queries"), py::arg("database"));

    m.def("l2_distance", &l2_distance_cuda,
          "Squared L2 distance (CUDA)",
          py::arg("queries"), py::arg("database"));

    m.def("dot_product", &dot_product_cuda,
          "Dot product (CUDA)",
          py::arg("queries"), py::arg("database"));

    // ─── Top-K ──────────────────────────────────────────────
    m.def("topk", [](const torch::Tensor& distances, int k, bool largest) {
        auto result = topk_auto(distances, k, largest);
        return std::make_tuple(result.distances, result.indices);
    }, "Top-K selection (auto-select best algorithm)",
       py::arg("distances"), py::arg("k"), py::arg("largest") = true);
    
    m.def("topk_thrust", [](const torch::Tensor& distances, int k, bool largest) {
        auto result = topk_thrust(distances, k, largest);
        return std::make_tuple(result.distances, result.indices);
    }, "Top-K selection using Thrust/CUB",
       py::arg("distances"), py::arg("k"), py::arg("largest") = true);
    
    m.def("topk_warp_heap", [](const torch::Tensor& distances, int k, bool largest) {
        auto result = topk_warp_heap(distances, k, largest);
        return std::make_tuple(result.distances, result.indices);
    }, "Top-K selection using warp-level heap (k <= 128)",
       py::arg("distances"), py::arg("k"), py::arg("largest") = true);

    // ─── FlatIndex ──────────────────────────────────────────
    py::class_<FlatIndex>(m, "FlatIndex")
        .def(py::init<int, Metric>(),
             py::arg("dim"), py::arg("metric") = Metric::COSINE)
        .def("add", [](FlatIndex& self, const torch::Tensor& vectors,
                        const py::object& ids_obj) {
            torch::Tensor ids;
            if (!ids_obj.is_none()) {
                ids = ids_obj.cast<torch::Tensor>();
            }
            self.add(vectors, ids);
        }, py::arg("vectors"), py::arg("ids") = py::none())
        .def("search", [](const FlatIndex& self,
                          const torch::Tensor& queries, int k) {
            auto result = self.search(queries, k);
            return std::make_tuple(result.distances, result.indices);
        }, py::arg("queries"), py::arg("k"))
        .def("reset", &FlatIndex::reset)
        .def("size", &FlatIndex::size)
        .def("dim", &FlatIndex::dim)
        .def("__len__", &FlatIndex::size)
        .def("__repr__", [](const FlatIndex& self) {
            return "FlatIndex(dim=" + std::to_string(self.dim()) +
                   ", size=" + std::to_string(self.size()) + ")";
        });
}

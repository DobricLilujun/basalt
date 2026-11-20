// src/prefill/fp4_dense_arm_stub.cpp - the dense NVFP4 products (sm_120a block-scaled mma) do not exist on
// NVIDIA Thor (sm_110); kernels::nvfp4p::fits refuses the format on aarch64 builds so these never run.  The
// arithmetic matches fp4_dense.cu so sizing code keeps working.
#include "basalt/prefill/fp4_dense.hpp"

#include <cstdlib>

namespace basalt::prefill::fp4_dense {

namespace {
constexpr int64_t kBT = 256;   // the product's token tile, as in fp4_dense.cu
int64_t padded(int64_t tokens) { return (tokens + kBT - 1) / kBT * kBT; }
}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / 2) + (size_t) padded(rows) * (size_t) (cols / 16);
}
void quantize(const float*, int64_t, int64_t, int64_t, void*, float*, void*) { std::abort(); }
void quantize_amax(const uint16_t*, const float*, int64_t, int64_t, int64_t, void*, float*, void*) { std::abort(); }
void rms_norm_quantize(const float*, const float*, float, int64_t, int64_t, void*, float*, float*, void*) { std::abort(); }
void rms_norm_quantize(const uint16_t*, const float*, float, int64_t, int64_t, void*, float*, uint16_t*, void*) { std::abort(); }
void product(const void*, const float*, int64_t, int64_t, const void*, const float*, int64_t, float*, int64_t, bool,
             void*) { std::abort(); }
void product_bf16(const void*, const float*, int64_t, int64_t, const void*, const float*, int64_t, uint16_t*, int64_t,
                  void*) { std::abort(); }
void product_add_bf16(const void*, const float*, int64_t, int64_t, const void*, const float*, int64_t, uint16_t*, int64_t,
                      void*) { std::abort(); }
void swiglu(const void*, const void*, const float*, const float*, int64_t, int64_t, const void*, const float*, int64_t,
            uint16_t*, int64_t, void*, float*) { std::abort(); }

}  // namespace basalt::prefill::fp4_dense

// src/models/qwen4exp/prefill/fp4/moe_fp4_arm_stub.cpp - the FP4 block-scaled tensor-core kernels do not
// exist on NVIDIA Thor (the instruction is sm_120a-only). The prompt path selects them only for
// NVFP4-native layers, and that selection is off on aarch64 (experts.hpp), so these bodies satisfy the
// link, never run.
#include "basalt/models/qwen4exp/prefill/moe_fp4.hpp"

#include <cstdlib>

namespace basalt::prefill::fp4 {

size_t a4_bytes(int64_t rows, int64_t cols, int experts) {
    // pure arithmetic: keep it real, the buffer sizing code uses it before any kernel
    const int64_t seg = (rows + experts - 1) / experts;
    const int64_t padded = ((seg + kTileTokens - 1) / kTileTokens) * kTileTokens;
    return (size_t) experts * (padded * (cols / 2) + padded * (cols / 16));
}
size_t scale_count(int64_t rows, int experts) { return (size_t) rows + experts; }
size_t token_bytes(int64_t tokens, int64_t cols) { return (size_t) tokens * (cols / 2 + cols / 16); }
void plan(const int32_t*, int, int32_t*, void*) { std::abort(); }
void rebase(const int32_t*, int, int32_t*, void*) { std::abort(); }
void quantize(const float*, const int32_t*, const int32_t*, const int32_t*, int, void*, float*, int64_t, int64_t,
              int64_t, void*) { std::abort(); }
void quantize_tokens(const float*, int64_t, int64_t, int64_t, void*, float*, void*) { std::abort(); }
void run(const Product&, void*) { std::abort(); }
void gather_tails(const Product&, size_t, float*, void*) { std::abort(); }
void run_experts(const Group&, int, int, void*) { std::abort(); }
int run_length(const Group&, int, int64_t) { std::abort(); }
int run_group(const Group&, int64_t, void*) { std::abort(); }
void cut_pieces(const int32_t*, int, int64_t, Pieces&) { std::abort(); }

}  // namespace basalt::prefill::fp4

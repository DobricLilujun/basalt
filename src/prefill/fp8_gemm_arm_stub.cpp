// src/prefill/fp8_gemm_arm_stub.cpp - the W8A8 products use the sm_120a FP8 tensor-core path, which NVIDIA
// Thor (sm_110) does not run; the engine keeps dense matrices in their ordinary formats on aarch64, so these
// bodies satisfy the link.  The byte arithmetic matches fp8_gemm.cu so sizing code keeps working.
#include "basalt/prefill/fp8_gemm.hpp"

#include <cstdlib>

namespace basalt::prefill::fp8 {

size_t scale_off_(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols + 15) & ~(size_t) 15; }
size_t act_bytes(int64_t rows, int64_t cols) { return scale_off_(rows, cols) + (size_t) rows * sizeof(float) + 16; }
float* act_scales(void* act, int64_t rows, int64_t cols) { return (float*) ((uint8_t*) act + scale_off_(rows, cols)); }

void quantize(const float*, int64_t, int64_t, int64_t, void*, void*) { std::abort(); }
void rms_norm_quantize(const float*, const float*, float, int64_t, int64_t, void*, float*, void*) { std::abort(); }
void rms_norm_quantize(const uint16_t*, const float*, float, int64_t, int64_t, void*, uint16_t*, void*) { std::abort(); }
void gemm(const uint8_t*, int64_t, int64_t, const float*, const void*, int64_t, void*, int64_t, Epilogue, void*) { std::abort(); }
void gemm_segmented(const Segment*, int, int64_t, const void*, int64_t, Epilogue, void*) { std::abort(); }

}  // namespace basalt::prefill::fp8

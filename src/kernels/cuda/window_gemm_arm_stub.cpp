// src/kernels/cuda/window_gemm_arm_stub.cpp - the window gemm's products are built with the sm_120a
// tensor-core instructions (the Q8_1 planes feed them directly); NVIDIA Thor (sm_110) does not run this build.
// The engine's dense stage keeps its ordinary mmvq/MMQ products on aarch64, so these bodies satisfy the link.
// The byte arithmetic matches window_gemm.cu so sizing code keeps working.
#include "basalt/kernels/window_gemm.hpp"

#include <cstdlib>

namespace basalt::kernels::wgemm {

void q8_planes(const void*, int64_t, int64_t, void*, void*) { std::abort(); }
std::size_t act_bytes(int64_t K) { return (size_t) kRows * K * sizeof(uint16_t); }
void pack(const float*, int64_t, int64_t, int, void*, void*) { std::abort(); }
void norm_pack(const float*, const float*, float, int64_t, int, void*, float*, void*) { std::abort(); }
void norm_pack(const uint16_t*, const float*, float, int64_t, int, void*, float*, void*) { std::abort(); }
void plan_device(int device) { (void) device; }
void workspace_need(const Call&, std::size_t& partial_floats, std::size_t& tickets) { partial_floats = 0; tickets = 0; }
void gemm(const Call&, const void*, int, const Workspace&, void*) { std::abort(); }

}  // namespace basalt::kernels::wgemm

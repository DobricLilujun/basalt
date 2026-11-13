#include "basalt/ggml/ggml.h"
// src/kernels/cpu/arm/quantize_helpers.cpp - placeholder for aarch64-specific activation quantizers.
// The quantizers the engine needs on this host come from the plain-C ggml build
// (src/kernels/cpu/ggml/ggml_cpu_dots_arm.c); this TU exists so the arm source list has a home for
// future NEON helpers without a CMake change.
namespace basalt::kernels::cpu {
}  // namespace basalt::kernels::cpu

// The AVX trait tables do not exist on this host; parity programs that name them link against these and
// fall back through ggml_get_type_traits_cpu.
extern "C" const struct ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx2(enum ggml_type) { return nullptr; }
extern "C" const struct ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx512(enum ggml_type) { return nullptr; }

// Picks the AVX-512 or the AVX2 build of the extracted ggml dot products for this CPU (see ggml_cpu_dots.c).
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"

extern "C" {
#if defined(__aarch64__)
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_arm(ggml_type type);
#else
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx2(ggml_type type);
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx512(ggml_type type);
#endif

const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type type) {
#if defined(__aarch64__)
    return ggml_get_type_traits_cpu_arm(type);   // the plain-C build: every type has a dot product
#else
    namespace cpu = basalt::kernels::cpu;
    static const bool avx512 = cpu::cpu_avx512_ok();
    static const bool avx2 = cpu::cpu_avx2_ok();
    if (avx512) return ggml_get_type_traits_cpu_avx512(type);
    if (avx2) return ggml_get_type_traits_cpu_avx2(type);
    return nullptr;
#endif
}
}

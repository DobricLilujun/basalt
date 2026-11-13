// include/basalt/platform/compiler.hpp - the compiler- and CPU-specific spellings the host code needs, behind
// one name each, so no source file asks which compiler it is built with.
//
// On x86-64 every function is inline and maps to the one instruction or builtin each compiler gives for it.
// On aarch64 (NVIDIA Thor hosts) the AVX spellings are x86-only: the intrinsics include and the CPUID spellings
// are guarded, the probes constant-fold to false, and the portable spellings below replace them.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define BASALT_CPU_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define BASALT_CPU_X86 0
#endif

// Marks a function that uses AVX2, FMA and F16C intrinsics in a translation unit built without those flags. MSVC
// accepts the intrinsics in any function and needs no marking.
#if defined(_MSC_VER)
#define BASALT_TARGET_AVX2_FMA_F16C
#else
#define BASALT_TARGET_AVX2_FMA_F16C __attribute__((target("avx2,fma,f16c")))
#endif

namespace basalt::platform {

inline constexpr bool kCpuX86 = BASALT_CPU_X86 != 0;

// MSVC can retain temporary stack copies for vector aggregate assignments even after inlining.
// Field-wise assignment exposes the values to register allocation.
#if defined(_MSC_VER) && !defined(__clang__)
inline constexpr bool kPreferMemberwiseVectorAssignment = true;
#else
inline constexpr bool kPreferMemberwiseVectorAssignment = false;
#endif

/// CPUID leaf `leaf`, subleaf `sub`: r[0..3] = EAX, EBX, ECX, EDX. x86 only; callers reach it only behind
/// the x86 feature probes.
inline void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if !BASALT_CPU_X86
    r[0] = r[1] = r[2] = r[3] = 0;
    (void) leaf; (void) sub;
    return;
#else
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
#endif
}

/// XCR0, which says which register state the OS saves (read only after CPUID reports OSXSAVE).
inline unsigned long long xgetbv0() {
#if !BASALT_CPU_X86
    return 0;
#elif defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
}

/// The index of the lowest set bit of a nonzero mask.
inline int ctz32(unsigned m) {
#if defined(_MSC_VER)
    unsigned long i = 0;
    _BitScanForward(&i, m);
    return (int) i;
#else
    return __builtin_ctz(m);
#endif
}

/// An IEEE half (its bits) to float.
inline float f16_round_bits(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

/// A store fence: orders the stores before it ahead of the one after it.
inline void store_fence() {
#if BASALT_CPU_X86
    _mm_sfence();
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/// One iteration of a spin wait: yields the pipeline to the core's SMT sibling. A no-op off x86.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#endif
}

/// The fp32 -> half -> fp32 round trip, round to nearest even. Where the compiler has a half type it does the
/// conversion; MSVC has none, and its F16C intrinsic pair is the same instruction with the same rounding.
inline float f16_round(float x) {
#if defined(_MSC_VER)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
#else
    return (float) (_Float16) x;
#endif
}

/// Whether this CPU, and the OS saving its AVX-512 state, can run AVX512_BF16 with AVX512BW.
inline bool cpu_has_avx512_bf16() {
#if !BASALT_CPU_X86
    return false;
#elif defined(_MSC_VER)
    // The same readings GCC's run-time probe takes: the OS has to save the AVX-512 state, and leaf 7 subleaf 0
    // names AVX512BW (EBX bit 30) and AVX512_BF16 (ECX bit 5).
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7u) return false;
    cpuid(1, 0, r);
    if (!((r[2] >> 27) & 1u)) return false;                 // OSXSAVE
    if ((xgetbv0() & 0xE6) != 0xE6) return false;           // the OS saves the AVX-512 state
    cpuid(7, 0, r);
    return ((r[1] >> 30) & 1u) != 0 && ((r[2] >> 5) & 1u) != 0;   // AVX512BW, AVX512_BF16
#else
    return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512bw");
#endif
}

}  // namespace basalt::platform

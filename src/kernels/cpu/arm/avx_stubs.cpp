// src/kernels/cpu/arm/avx_stubs.cpp - link-compatible stubs for the AVX kernel entry points.
// The AVX kernels are not built on aarch64; every caller reaches them only behind cpu_avx*_ok(), which is
// always false there, so these bodies exist to satisfy the linker, never to run.
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"

#include <cstdlib>

namespace basalt::kernels::cpu {

void quantize_q8_K_avx512(const float* x, void* dst, int64_t k)  { std::abort(); }
bool iq512_supported(int ggml_type) noexcept { return false; }
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
bool iq1s_512_fits(int n) noexcept { return false; }
void iq1s_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq1s_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
bool iq_exact_512_supported(int ggml_type) noexcept { return false; }
void iq_exact_512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq_exact_512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void quantize_q8_K_avx2(const float* x, void* dst, int64_t k)  { std::abort(); }
bool iq256_supported(int ggml_type) noexcept { return false; }
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void iq256_gu_rows_base(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq256_gu_rows_vnni(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq256_rows_base(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void iq256_rows_vnni(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void iq256_gu_rows_mode_base(int mode, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq256_gu_rows_mode_vnni(int mode, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void iq256_rows_mode_base(int mode, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void iq256_rows_mode_vnni(int mode, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
bool iq4nl256_fits(int n) noexcept { return false; }
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void iq4nl256_down_rows_base(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void iq4nl256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1)  { std::abort(); }
bool kq256_supported(int ggml_type) noexcept { return false; }
void kq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void kq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out)  { std::abort(); }
bool nvfp4_512_fits(int n_embd, int n_ff) noexcept { return false; }
void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void nvfp4_512_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
bool nvfp4_256_fits(int n_embd, int n_ff) noexcept { return false; }
void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void nvfp4_256_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void nvfp4_256_gu_rows_base(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void nvfp4_256_gu_rows_vnni(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1)  { std::abort(); }
void nvfp4_256_down_rows_base(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }
void nvfp4_256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1)  { std::abort(); }

// declared next to its use in src and tests; the AVX1 build of the bf16 router dot
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) { std::abort(); }

}  // namespace basalt::kernels::cpu

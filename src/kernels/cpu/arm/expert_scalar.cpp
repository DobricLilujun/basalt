// src/kernels/cpu/arm/expert_scalar.cpp - the Q2_0 expert kernel, scalar form (aarch64).
//
// Same contract as the AVX kernels in src/kernels/cpu/expert.cpp and q2_avx2*.cpp:
//   act_quant: per 32-value chunk, s = max|x|/127 (0 when max|x| is 0), inv = 1/s,
//   q_j = clamp(trunc(x_j*inv + copysign(0.5, x_j*inv)), -127, 127), plus sum and hx = s*sum.
//   rows: each weight row is `nblocks` 18-byte blocks (fp16 d, then 16 code bytes = 64 codes c in [0,3]);
//   out[t] = sum_b d_b * ( sum_chunks xs_c*(sum_{j in c} c_j*q_j) - sum_chunks xs_c*hx_c ).
// The values match the AVX path's contract; the float accumulation order differs.
#include "basalt/kernels/cpu/expert.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

// fp16 -> fp32, round-to-nearest-even, no F16C needed.
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

void act_quant_scalar(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fabs(xb[j]) > amax ? std::fabs(xb[j]) : amax;
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int sum = 0;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            a.q[k * QKA + j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

// tokens [0, nt) share the row; each token has its own quantized activation.
void q2g_row_multi_scalar(const uint8_t* row, const ActQ* const* a, int nt, int nblocks, float* res) {
    for (int t = 0; t < nt; ++t) res[t] = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float dw = h2f(blk);
        const uint8_t* codes = blk + 2;
        for (int t = 0; t < nt; ++t) {
            const ActQ* act = a[t];
            float acc = 0.f, corr = 0.f;
            for (int ci = 0; ci < 2; ++ci) {           // two 32-activation chunks per 64-weight block
                const int c = 2 * b + ci;
                float part = 0.f;
                for (int j = 0; j < QKA; ++j) {
                    const uint8_t cbyte = codes[ci * 16 + (j >> 2)];
                    const uint8_t code = (cbyte >> (2 * (j & 3))) & 3u;
                    part += (float) code * (float) act->q[c * QKA + j];
                }
                acc += act->scale[c] * part;
                corr += act->hx[c];
            }
            res[t] += dw * (acc - corr);
        }
    }
}

}  // namespace

CpuFeatures cpu_features() { return CpuFeatures{}; }   // no AVX-512 on aarch64

const char* CpuFeatures::reason() const { return "no AVX-512 (aarch64)"; }

void act_quant_q8_1(const float* x, int n, ActQ& a) { act_quant_scalar(x, n, a); }
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_scalar(x, n, a); }

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    float res[MAXT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi_scalar(w + (size_t) r * row_bytes, a, nt, nblocks, res);
        for (int t = 0; t < nt; ++t) out[t][r] = res[t];
    }
}
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx2_base(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                    float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx2_vnni(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                    float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

}  // namespace basalt::kernels::cpu

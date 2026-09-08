// Launch80 W4A16 GEMV - Stage 2, the kill gate.
//
// Purpose is NOT to ship a 4-bit kernel (Intel's is fine). It is to prove we can
// write a wall-class GEMV at all, on a format where a trusted reference exists, so
// that when 3-bit misses we know it is the format and not our code. Same machinery,
// one variable.
//
// Three things this does differently from research/r8/sycl/w4a16_v3.cpp (281 GB/s):
//
//  1. vec<int32,4> weight loads. v3 read 4 bytes per lane - correctly coalesced but
//     4x too narrow. Stage 1 measured this lever at +22% on the read path alone.
//
//  2. The scale is one register load per 32 weights, from a [N, K/128] column-major
//     layout. v3 did sc[(w/16)*N + n], where lanes in one instruction touch
//     addresses 34816 bytes apart - an 8-cache-line gather per instruction.
//
//  3. Dequant via half2 instead of ~6 scalar ops per weight. The weights are
//     REPACKED OFFLINE so that weight 2t sits at bit 4t and weight 2t+1 at bit
//     16+4t. Then (w >> 4t) & 0x000F000F | 0x64006400, reinterpreted as half2, is
//     literally the pair (1024+w_2t, 1024+w_2t+1) - 0x6400 is fp16 1024.0 and the
//     low mantissa bits carry the nibble. One hsub2 removes 1024+8 (the symmetric
//     zero point), one hfma2 multiplies by x and accumulates. Three ops for two
//     weights, and the x pair (x[2t], x[2t+1]) is naturally adjacent so no shuffle
//     is needed.
//
// Submits on torch's CURRENT XPU stream. A private sycl::queue is never recorded by
// XPU graph capture and replays stale output with no error - measured in
// research/r8/graphprobe.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

using half_t = sycl::half;
using half2_t = sycl::vec<half_t, 2>;
using v4_t = sycl::vec<int32_t, 4>;
using h8_t = sycl::vec<half_t, 8>;

template <int SG, int SGS, int NCOL> class W4Kernel;

template <int SG, int SGS, int NCOL>
static void launch_w4(sycl::queue &q, const half_t *x, const int32_t *qw,
                      const half_t *sc, half_t *out, int K, int N) {
    const int W  = K / 8;          // int32 words per column
    const int NV = W / 4;          // vec4 units per column; 32 weights each
    const int NG = K / 128;        // scale groups per column
    const size_t cols = (size_t)SGS * NCOL;
    const size_t wg = SG * SGS, nwg = (N + cols - 1) / cols;

    q.submit([&](sycl::handler &h) {
        h.parallel_for<W4Kernel<SG, SGS, NCOL>>(
            sycl::nd_range<1>(nwg * wg, wg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int tid = it.get_local_id(0), lane = tid % SG;
                const size_t n0 = it.get_group(0) * cols + (size_t)(tid / SG) * NCOL;
                if (n0 >= (size_t)N) return;
                const half2_t zp{(half_t)1032.0f, (half_t)1032.0f};   // 1024 + 8
                float acc[NCOL];
#pragma unroll
                for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;

                for (int u = lane; u < NV; u += SG) {
                    // x is identical for every output column, so hoist its four h8
                    // loads out of the column loop and reuse them from registers.
                    // This is the change that took the 3-bit kernel from 1.02x to
                    // 1.17x; the 4-bit kernel never had it.
                    h8_t xv[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        xv[e] = *reinterpret_cast<const h8_t *>(x + (size_t)u * 32 + e * 8);
#pragma unroll
                    for (int c = 0; c < NCOL; ++c) {
                        const size_t n = n0 + c;
                        if (n >= (size_t)N) continue;
                        const v4_t wv = *reinterpret_cast<const v4_t *>(
                            qw + n * (size_t)W + (size_t)u * 4);
                        half2_t s2{(half_t)0.f, (half_t)0.f};
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const uint32_t w = (uint32_t)wv[e];
#pragma unroll
                            for (int t = 0; t < 4; ++t) {
                                const uint32_t bits =
                                    ((w >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                                s2 += (sycl::bit_cast<half2_t>(bits) - zp)
                                      * half2_t{xv[e][2 * t], xv[e][2 * t + 1]};
                            }
                        }
                        acc[c] += ((float)s2[0] + (float)s2[1])
                                  * (float)sc[n * (size_t)NG + u / 4];
                    }
                }
#pragma unroll
                for (int c = 0; c < NCOL; ++c) {
                    const float r = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
                    if (lane == 0 && n0 + c < (size_t)N) out[n0 + c] = (half_t)r;
                }
            });
    });
}

at::Tensor gemv_w4(const at::Tensor &x, const at::Tensor &qw, const at::Tensor &sc,
                   int64_t ncol) {
    TORCH_CHECK(x.is_xpu() && qw.is_xpu() && sc.is_xpu(), "all tensors must be XPU");
    TORCH_CHECK(qw.dim() == 2 && sc.dim() == 2, "qw [N,K/8], sc [N,K/128]");
    TORCH_CHECK(qw.is_contiguous() && sc.is_contiguous(), "qw and sc must be contiguous");
    const int N = (int)qw.size(0), K = (int)qw.size(1) * 8;
    TORCH_CHECK(x.numel() == K, "x must have K elements");
    TORCH_CHECK((K % 1024) == 0, "K must be a multiple of 1024");

    auto out = at::empty({1, N}, x.options());
    sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
    const auto *xp  = reinterpret_cast<const half_t *>(x.data_ptr<at::Half>());
    const auto *scp = reinterpret_cast<const half_t *>(sc.data_ptr<at::Half>());
    auto *op        = reinterpret_cast<half_t *>(out.data_ptr<at::Half>());
    auto *qp = qw.data_ptr<int32_t>();
    switch (ncol) {
        case 1:  launch_w4<32, 4, 1>(q, xp, qp, scp, op, K, N); break;
        case 4:  launch_w4<32, 4, 4>(q, xp, qp, scp, op, K, N); break;
        case 8:  launch_w4<32, 4, 8>(q, xp, qp, scp, op, K, N); break;
        default: launch_w4<32, 4, 2>(q, xp, qp, scp, op, K, N); break;
    }
    return out;
}

TORCH_LIBRARY(p608, m) {
    m.def("gemv_w4(Tensor x, Tensor qw, Tensor sc, int ncol=2) -> Tensor");
}
TORCH_LIBRARY_IMPL(p608, XPU, m) { m.impl("gemv_w4", gemv_w4); }

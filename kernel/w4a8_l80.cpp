// Launch80 W4A8 GEMV - closing the ALU gap with dp4a.
//
// Why this exists. The 4-bit kernel runs at 537.9 GB/s where the SAME access pattern
// reads at 579.4 read-only, so ~7% of the time is dequant ALU standing between us and
// the memory system. At fp16 that dequant is ~5 ops per 2 weights (shift, and, or,
// subtract, fma) and there is no obvious way below it.
//
// At 4 bits the nibbles line up with byte lanes, which changes everything:
//     a = (w      ) & 0x0F0F0F0F   -> weights 0,2,4,6 as four uint8 lanes
//     b = (w >> 4 ) & 0x0F0F0F0F   -> weights 1,3,5,7
// and dp4a (sycl::ext::oneapi::dot_acc) does four MACs per instruction. Five ops per
// EIGHT weights instead of five per two - a 4x reduction in dequant work.
//
// Two further wins fall out of it:
//   * int8 activations are half the bytes and half the load instructions of fp16.
//   * The zero point becomes FREE and EXACT. sum((q-8)x) = sum(qx) - 8*sum(x), and
//     with int32 accumulation there is no precision cost to folding it - which is
//     exactly the objection that killed this trick for the fp16 path, where the
//     accumulator would have held 1024*sum(x) against a much smaller signal.
//
// The cost is that activations are quantized to int8. Intel ships the same tradeoff
// as int4_gemm_w4a8, so it is an accepted one, but it IS a quality change and needs a
// gate - unlike the fp16 kernel, which is bit-comparable to the vendor's.
//
// Activations arrive pre-interleaved: for each 32-bit weight word, one int32 holds
// x at the even nibble positions and the next holds the odd ones, so each dp4a takes
// one aligned load and no shuffle.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>

using half_t = sycl::half;
using v4_t = sycl::vec<int32_t, 4>;
namespace dp = sycl::ext::oneapi;

template <int SG, int SGS, int NCOL> class W4A8Kernel;

template <int SG, int SGS, int NCOL>
static void launch(sycl::queue &q, const int32_t *xq, const int32_t *sumx,
                   const int32_t *qw, const half_t *sc, float xscale,
                   half_t *out, int K, int N) {
    const int W  = K / 8;        // int32 weight words per column
    const int NV = W / 4;        // vec4 units per column; 32 weights each
    const int NG = K / 128;      // scale groups per column
    const size_t cols = (size_t)SGS * NCOL;
    const size_t wg = SG * SGS, nwg = (N + cols - 1) / cols;

    q.submit([&](sycl::handler &h) {
        h.parallel_for<W4A8Kernel<SG, SGS, NCOL>>(
            sycl::nd_range<1>(nwg * wg, wg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int tid = it.get_local_id(0), lane = tid % SG;
                const size_t n0 = it.get_group(0) * cols + (size_t)(tid / SG) * NCOL;
                if (n0 >= (size_t)N) return;
                float acc[NCOL];
#pragma unroll
                for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;

                for (int u = lane; u < NV; u += SG) {
                    // 32 weights need 32 int8 activations = 8 int32 = two vec4 loads,
                    // shared across all NCOL columns.
                    const v4_t xa = *reinterpret_cast<const v4_t *>(xq + (size_t)u * 8);
                    const v4_t xb = *reinterpret_cast<const v4_t *>(xq + (size_t)u * 8 + 4);
                    const int32_t sx = sumx[u];
#pragma unroll
                    for (int c = 0; c < NCOL; ++c) {
                        const size_t n = n0 + c;
                        if (n >= (size_t)N) continue;
                        const v4_t wv = *reinterpret_cast<const v4_t *>(
                            qw + n * (size_t)W + (size_t)u * 4);
                        int32_t a = 0;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const uint32_t w = (uint32_t)wv[e];
                            // even-position nibbles, then odd; four MACs each
                            // weights are UNSIGNED nibbles (0..15), activations are
                            // SIGNED int8. The (uint32, int32) overload is the correct
                            // one; casting x to uint32 picks unsigned x unsigned and
                            // turns every negative activation into a large positive.
                            const int32_t xe = (e < 2) ? xa[2 * e] : xb[2 * (e - 2)];
                            const int32_t xo = (e < 2) ? xa[2 * e + 1] : xb[2 * (e - 2) + 1];
                            a = dp::dot_acc(w & 0x0F0F0F0Fu, xe, a);
                            a = dp::dot_acc((w >> 4) & 0x0F0F0F0Fu, xo, a);
                        }
                        // zero point folded exactly: sum((q-8)x) = sum(qx) - 8*sum(x)
                        acc[c] += (float)(a - 8 * sx) * (float)sc[n * (size_t)NG + u / 4];
                    }
                }
#pragma unroll
                for (int c = 0; c < NCOL; ++c) {
                    const float r = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
                    if (lane == 0 && n0 + c < (size_t)N)
                        out[n0 + c] = (half_t)(r * xscale);
                }
            });
    });
}

at::Tensor gemv_w4a8(const at::Tensor &xq, const at::Tensor &sumx, const at::Tensor &qw,
                     const at::Tensor &sc, double xscale, int64_t ncol) {
    TORCH_CHECK(xq.is_xpu() && qw.is_xpu() && sc.is_xpu() && sumx.is_xpu(), "XPU only");
    TORCH_CHECK(qw.is_contiguous() && sc.is_contiguous(), "qw/sc contiguous");
    const int N = (int)qw.size(0), K = (int)qw.size(1) * 8;
    auto out = at::empty({1, N},
                         at::TensorOptions().dtype(at::kHalf).device(qw.device()));
    sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
    const auto *scp = reinterpret_cast<const half_t *>(sc.data_ptr<at::Half>());
    auto *op = reinterpret_cast<half_t *>(out.data_ptr<at::Half>());
    const auto *xp = xq.data_ptr<int32_t>();
    const auto *sp = sumx.data_ptr<int32_t>();
    auto *qp = qw.data_ptr<int32_t>();
    switch (ncol) {
        case 1: launch<32, 1, 1>(q, xp, sp, qp, scp, (float)xscale, op, K, N); break;
        case 4: launch<32, 1, 4>(q, xp, sp, qp, scp, (float)xscale, op, K, N); break;
        default: launch<32, 1, 2>(q, xp, sp, qp, scp, (float)xscale, op, K, N); break;
    }
    return out;
}

TORCH_LIBRARY(p608a8, m) {
    m.def("gemv_w4a8(Tensor xq, Tensor sumx, Tensor qw, Tensor sc, float xscale, "
          "int ncol=2) -> Tensor");
}
TORCH_LIBRARY_IMPL(p608a8, XPU, m) { m.impl("gemv_w4a8", gemv_w4a8); }

// Launch80 W3A16 GEMV for Battlemage. Stage 4.
//
// Nothing ships a fast 3-bit weight-only GEMV: Marlin and Machete reject
// bits != 4,8; BitBLAS has no INT3 row; Intel's ARK lists INT4/INT8 only for
// Battlemage; bestla has int2/int4. This is new.
//
// Structure comes from measurement, not intuition (docs/13):
//   * THREE PLANES. Word 0 of every 32-weight block lives in plane 0, word 1 in
//     plane 1, word 2 in plane 2, each plane contiguous down the column. One vec4
//     from each plane gives a lane 4 whole blocks = 128 weights = exactly one scale
//     group = exactly 256 bytes of x. Three coalesced streams; read-only this hits
//     591 GB/s, 100.7% of the measured wall.
//   * COLUMN-MAJOR, sub-group per column. The k-major alternative - which makes
//     x[k] a broadcast and removes the cross-lane reduction - measured 32% SLOWER,
//     because its three words sit ~68 KB apart and that destroys DRAM page locality.
//   * NO grf_size<256> (costs 15% here; a GEMV needs threads, not registers) and
//     NO prefetch (the hardware prefetcher already handles a sequential stream),
//     even though oneDNN's JIT picks grf256 for this problem.
//
// Dequant is the half2 trick, which is why the format is shaped the way it is:
// (w >> s) & 0x00070007 | 0x64006400 reinterpreted as half2 IS (1024+lo, 1024+hi),
// since 0x6400 is fp16 1024.0 and the low mantissa bits carry the 3-bit field. One
// subtract of 1028 (1024 + the zero point 4), one hfma2 against the naturally
// adjacent x pair. 30 of every 32 weights decode this way; the last 2 come from the
// 6 bits that s in {0,3,6,9,12} cannot reach (bit 15 and 31 of each word), so the
// format wastes nothing.
//
// Submits on torch's current XPU stream - a private queue is silently dropped by
// XPU graph capture and replays stale output (measured, research/r8/graphprobe).
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

using half_t = sycl::half;
using half2_t = sycl::vec<half_t, 2>;
using v4_t = sycl::vec<int32_t, 4>;
template <int BPU> using vb_t = sycl::vec<int32_t, BPU>;
using h8_t = sycl::vec<half_t, 8>;

template <int SG, int SGS, bool STRAG, int BPU, int NCOL> class W3Kernel;

template <int SG, int SGS, bool STRAG, int BPU, int NCOL>
static void launch_w3(sycl::queue &q, const half_t *x, const int32_t *qw,
                      const half_t *sc, half_t *out, int K, int N) {
    const int NBLK = K / 32;       // 32-weight blocks per column = words per plane
    const int NU   = NBLK / BPU;   // lane-iterations per column
    const size_t W = (size_t)3 * NBLK;
    const size_t cols = (size_t)SGS * NCOL;
    const size_t wg = SG * SGS, nwg = (N + cols - 1) / cols;

    q.submit([&](sycl::handler &h) {
        h.parallel_for<W3Kernel<SG, SGS, STRAG, BPU, NCOL>>(
            sycl::nd_range<1>(nwg * wg, wg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int tid = it.get_local_id(0), lane = tid % SG;
                const size_t n0 = it.get_group(0) * cols + (size_t)(tid / SG) * NCOL;
                if (n0 >= (size_t)N) return;

                // x is identical for every output column, so with NCOL columns per
                // sub-group each block's 4 h8 loads are issued ONCE and reused from
                // registers across all NCOL of them. At BPU=1 only 4 h8 are live at
                // a time (16 GRF), which is why this fits where the BPU=4 attempt
                // spilled and collapsed to 39 GB/s.
                const half2_t zp{(half_t)1028.0f, (half_t)1028.0f};
                const int NSG = NBLK / 4;
                float acc[NCOL];
#pragma unroll
                for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;

                for (int u = lane; u < NU; u += SG) {
                    const size_t k0 = (size_t)u * 32;
                    h8_t xv[4];
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        xv[t] = *reinterpret_cast<const h8_t *>(x + k0 + t * 8);
#pragma unroll
                    for (int c = 0; c < NCOL; ++c) {
                        const size_t n = n0 + c;
                        if (n >= (size_t)N) continue;
                        const int32_t *col = qw + n * W;
                        const uint32_t w0 = (uint32_t)col[u];
                        const uint32_t w1 = (uint32_t)col[NBLK + u];
                        const uint32_t w2 = (uint32_t)col[2 * NBLK + u];
                        half2_t s2{(half_t)0.f, (half_t)0.f};
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            const uint32_t wm = (m == 0) ? w0 : ((m == 1) ? w1 : w2);
#pragma unroll
                            for (int t = 0; t < 5; ++t) {
                                const int pp = m * 5 + t, sh = 3 * t, xi = 2 * pp;
                                const uint32_t bits =
                                    ((wm >> sh) & 0x00070007u) | 0x64006400u;
                                s2 += (sycl::bit_cast<half2_t>(bits) - zp)
                                      * half2_t{xv[xi >> 3][xi & 7],
                                                xv[xi >> 3][(xi & 7) + 1]};
                            }
                        }
                        if constexpr (STRAG) {
                            const uint32_t um = (w0 & 0x80008000u)
                                              | ((w1 & 0x80008000u) >> 1)
                                              | ((w2 & 0x80008000u) >> 2);
                            const uint32_t tb = ((um >> 13) & 0x00070007u) | 0x64006400u;
                            s2 += (sycl::bit_cast<half2_t>(tb) - zp)
                                  * half2_t{xv[3][6], xv[3][7]};
                        }
                        acc[c] += ((float)s2[0] + (float)s2[1])
                                  * (float)sc[n * (size_t)NSG + u / 4];
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

// nostrag=true skips weights 30/31 entirely: numerically WRONG, used only to
// bound how much of the remaining gap the straggler path still owns.
at::Tensor gemv_w3(const at::Tensor &x, const at::Tensor &qw, const at::Tensor &sc,
                   bool nostrag, int64_t ncol) {
    // NCOL=2 output columns per sub-group is the default. It amortises each block's
    // four h8 activation loads over two columns, worth 1.02x -> 1.17x. NCOL=8 spills
    // and collapses to 0.77x, so this is a real optimum and not a monotonic knob.

    TORCH_CHECK(x.is_xpu() && qw.is_xpu() && sc.is_xpu(), "all tensors must be XPU");
    TORCH_CHECK(qw.is_contiguous() && sc.is_contiguous(), "qw and sc must be contiguous");
    const int N = (int)qw.size(0);
    const int K = (int)(qw.size(1) / 3) * 32;
    TORCH_CHECK(x.numel() == K, "x must have K elements");
    TORCH_CHECK((K % 128) == 0, "K must be a multiple of 128");
    auto out = at::empty({1, N}, x.options());
    sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
    const auto *xp  = reinterpret_cast<const half_t *>(x.data_ptr<at::Half>());
    const auto *scp = reinterpret_cast<const half_t *>(sc.data_ptr<at::Half>());
    auto *op        = reinterpret_cast<half_t *>(out.data_ptr<at::Half>());
    auto *qp        = qw.data_ptr<int32_t>();
#define DISPATCH(NCOL_)                                                            \
    do {                                                                          \
        if (nostrag) launch_w3<32, 4, false, 1, NCOL_>(q, xp, qp, scp, op, K, N); \
        else         launch_w3<32, 4, true,  1, NCOL_>(q, xp, qp, scp, op, K, N); \
    } while (0)
    switch (ncol) {
        case 2: DISPATCH(2); break;
        case 4: DISPATCH(4); break;
        case 8: DISPATCH(8); break;
        default: DISPATCH(1); break;
    }
#undef DISPATCH
    return out;
}

TORCH_LIBRARY(p608w3, m) { m.def("gemv_w3(Tensor x, Tensor qw, Tensor sc, bool nostrag=False, int ncol=2) -> Tensor"); }
TORCH_LIBRARY_IMPL(p608w3, XPU, m) { m.impl("gemv_w3", gemv_w3); }

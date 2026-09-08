// Stage 1 - can the 3-bit byte layout even be READ at the memory wall?
//
// No dequant, no activations, no math that matters: just stream exactly the bytes
// a W3A16 GEMV would stream (3 bits/weight + one fp16 scale per 128) and XOR them
// into an accumulator so nothing is optimised away. If the layout cannot be read
// near 587 GB/s here, no amount of dequant cleverness recovers it later, and this
// is the cheapest possible place to find that out.
//
// The 3-bit format is ExLlamaV2's: 32 weights in EXACTLY 3 int32 words, so there
// is no padding waste (10-per-word would throw away 6.25% of the bandwidth, which
// is unaffordable in a kernel that is nothing but bandwidth).
//
// The open question this settles is ORIENTATION, because the layout and the thread
// mapping are one decision, not two:
//
//   A  NT / column-major   qw[n*W + w]      one sub-group per output column,
//                                           lanes split K. What our int4 kernel
//                                           does today. Every column re-reads all
//                                           of x; scales for a column are strided.
//
//   B  k-major             qw[w*N + n]      ONE LANE per output column, the whole
//                                           sub-group walking K together. Weight
//                                           reads coalesce across n, the scale read
//                                           coalesces for free, x[k] becomes a
//                                           broadcast instead of a re-read, and the
//                                           cross-lane reduction disappears entirely.
//
// B fixes the scale-gather defect by construction. Measured, not assumed.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

using half_t = sycl::half;
constexpr int GROUP = 128;      // weights per scale
constexpr double WALL = 587.0;

struct Bufs {
    std::vector<int32_t*> qw;
    std::vector<half_t*>  sc;
    int nbuf;
};

static Bufs alloc_rot(sycl::queue &q, size_t words, size_t nscale, double footprint_mb) {
    Bufs b;
    size_t bytes = words * 4 + nscale * 2;
    b.nbuf = std::max(2, (int)(footprint_mb * 1e6 / bytes + 0.5));
    std::vector<int32_t> hq(words);
    std::vector<half_t>  hs(nscale);
    std::srand(608);
    for (auto &v : hq) v = (int32_t)(((uint32_t)std::rand() << 17) ^ (uint32_t)std::rand());
    for (auto &v : hs) v = (half_t)(0.01f);
    for (int i = 0; i < b.nbuf; ++i) {
        b.qw.push_back(sycl::malloc_device<int32_t>(words, q));
        b.sc.push_back(sycl::malloc_device<half_t>(nscale, q));
        q.memcpy(b.qw.back(), hq.data(), words * 4).wait();
        q.memcpy(b.sc.back(), hs.data(), nscale * 2).wait();
    }
    return b;
}
static void free_rot(sycl::queue &q, Bufs &b) {
    for (int i = 0; i < b.nbuf; ++i) { sycl::free(b.qw[i], q); sycl::free(b.sc[i], q); }
}

// ---------------------------------------------------------------------------
// A: NT / column-major, one sub-group per column, lanes split K.
template <int SG, int SGS, int UNROLL> class KA;
template <int SG, int SGS, int UNROLL>
double runA(sycl::queue &q, int K, int N, int reps) {
    const int W = 3 * (K / 32);            // int32 words per output column
    const int NG = K / GROUP;
    Bufs b = alloc_rot(q, (size_t)W * N, (size_t)NG * N, 512.0);
    auto *o = sycl::malloc_device<uint32_t>(N, q);
    const size_t wg = SG * SGS, nwg = (N + SGS - 1) / SGS;

    auto go = [&](int bi) {
        const int32_t *qw = b.qw[bi % b.nbuf];
        const half_t  *sc = b.sc[bi % b.nbuf];
        return q.submit([&](sycl::handler &h) {
            h.parallel_for<KA<SG, SGS, UNROLL>>(
                sycl::nd_range<1>(nwg * wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    auto sg = it.get_sub_group();
                    const int tid = it.get_local_id(0), lane = tid % SG;
                    const size_t n = it.get_group(0) * SGS + tid / SG;
                    if (n >= (size_t)N) return;
                    const int32_t *col = qw + n * (size_t)W;
                    uint32_t acc = 0;
                    for (int base = 0; base < W; base += SG * UNROLL) {
#pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            int ww = base + u * SG + lane;
                            if (ww < W) acc ^= (uint32_t)col[ww];
                        }
                    }
                    // scales laid out column-contiguous so this is not a gather
                    for (int g = lane; g < NG; g += SG)
                        acc ^= sycl::bit_cast<uint16_t>(sc[n * (size_t)NG + g]);
                    acc = sycl::reduce_over_group(sg, acc, sycl::bit_xor<uint32_t>());
                    if (lane == 0) o[n] = acc;
                });
        });
    };
    for (int i = 0; i < 5; ++i) go(i).wait();
    double best = 1e30;
    for (int r = 0; r < 8; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) go(i);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count() / reps);
    }
    free_rot(q, b); sycl::free(o, q);
    return best;
}

// ---------------------------------------------------------------------------
// B: k-major, ONE LANE per column. No cross-lane reduction at all.
template <int SG, int SGS, int UNROLL> class KB;
template <int SG, int SGS, int UNROLL>
double runB(sycl::queue &q, int K, int N, int reps) {
    const int NB = K / 32;                 // 32-weight blocks along K
    const int NG = K / GROUP;
    Bufs b = alloc_rot(q, (size_t)3 * NB * N, (size_t)NG * N, 512.0);
    auto *o = sycl::malloc_device<uint32_t>(N, q);
    const size_t wg = SG * SGS, nwg = (N + wg - 1) / wg;

    auto go = [&](int bi) {
        const int32_t *qw = b.qw[bi % b.nbuf];
        const half_t  *sc = b.sc[bi % b.nbuf];
        return q.submit([&](sycl::handler &h) {
            h.parallel_for<KB<SG, SGS, UNROLL>>(
                sycl::nd_range<1>(nwg * wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const size_t n = it.get_global_id(0);
                    if (n >= (size_t)N) return;
                    uint32_t acc = 0;
                    for (int base = 0; base < NB; base += UNROLL) {
#pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            const int bb = base + u;
                            if (bb < NB) {
#pragma unroll
                                for (int j = 0; j < 3; ++j)
                                    acc ^= (uint32_t)qw[(size_t)(3 * bb + j) * N + n];
                            }
                        }
                    }
                    // consecutive lanes -> consecutive n -> coalesced, for free
                    for (int g = 0; g < NG; ++g)
                        acc ^= sycl::bit_cast<uint16_t>(sc[(size_t)g * N + n]);
                    o[n] = acc;
                });
        });
    };
    for (int i = 0; i < 5; ++i) go(i).wait();
    double best = 1e30;
    for (int r = 0; r < 8; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) go(i);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count() / reps);
    }
    free_rot(q, b); sycl::free(o, q);
    return best;
}

int main() {
    // in-order, to match torch-xpu's queue: an out-of-order queue lets the burst
    // overlap and overstates the rate by ~8%.
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    const size_t gmem = dev.get_info<sycl::info::device::global_mem_size>();
    if (gmem < 20ull << 30) {
        std::printf("\nREFUSING TO RUN: %.1f GB device - this is not the B65.\n"
                    "Pass --device /dev/dri/renderD130, not all of /dev/dri.\n", gmem / 1e9);
        return 2;
    }
    const int K = 5120, N = 17408;
    const double bytes = (double)(3 * (K / 32)) * N * 4.0 + (double)(K / GROUP) * N * 2.0;
    std::printf("  3-bit payload %dx%d = %.2f MB   wall %.0f GB/s   GATE >= 550 GB/s\n",
                K, N, bytes / 1e6, WALL);
    std::printf("  (vendor int4 reads 46.0 MB @ 473 GB/s; 3-bit break-even = 358)\n\n");
    std::printf("  %-6s %5s %5s %7s %10s %9s %7s\n",
                "layout", "SG", "SGS", "UNROLL", "us", "GB/s", "%wall");
    std::printf("  ---------------------------------------------------------------\n");
    double bestA = 0, bestB = 0;
#define TA(sg, sgs, un) { double t = runA<sg,sgs,un>(q, K, N, 50); double g = bytes/t/1e9; \
    bestA = std::max(bestA, g); \
    std::printf("  %-6s %5d %5d %7d %10.1f %9.1f %6.1f%%\n", "A NT", sg, sgs, un, t*1e6, g, g/WALL*100); }
#define TB(sg, sgs, un) { double t = runB<sg,sgs,un>(q, K, N, 50); double g = bytes/t/1e9; \
    bestB = std::max(bestB, g); \
    std::printf("  %-6s %5d %5d %7d %10.1f %9.1f %6.1f%%\n", "B kmaj", sg, sgs, un, t*1e6, g, g/WALL*100); }
    TA(32, 4, 2) TA(32, 4, 4) TA(32, 8, 2) TA(32, 8, 4) TA(32, 2, 4)
    std::printf("  ---------------------------------------------------------------\n");
    TB(32, 4, 2) TB(32, 4, 4) TB(32, 8, 2) TB(32, 8, 4) TB(32, 2, 4)
    TB(16, 8, 4) TB(32, 16, 4) TB(32, 8, 8)
    std::printf("  ---------------------------------------------------------------\n");
    std::printf("  best A (NT, sub-group per column): %.1f GB/s (%.0f%% of wall)\n",
                bestA, bestA / WALL * 100);
    std::printf("  best B (k-major, lane per column): %.1f GB/s (%.0f%% of wall)\n",
                bestB, bestB / WALL * 100);
    std::printf("\n  %s\n", std::max(bestA, bestB) >= 550.0
        ? "GATE PASS - the layout reads at the wall; proceed to dequant."
        : "GATE MISS - fix the read path before writing any 3-bit logic.");
    return 0;
}

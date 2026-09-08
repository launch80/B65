// v3: v2 + vector loads. The 8 x-values a lane needs are CONTIGUOUS (kb=w*8),
// so read them as one 16-byte vec<half,8> instead of 8 scalar loads that each
// use 2 bytes of a 16-byte line. Same for accumulating in half2 pairs.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
using half_t = sycl::half;
constexpr int GROUP = 128;

template <int SG, int SGS, int UNROLL>
double run(sycl::queue &q, int K, int N, int reps, double *out_err) {
    const int K8 = K / 8, NG = K / GROUP;
    auto *qw = sycl::malloc_device<int32_t>((size_t)K8 * N, q);
    auto *sc = sycl::malloc_device<half_t>((size_t)NG * N, q);
    auto *x  = sycl::malloc_device<half_t>(K, q);
    auto *o  = sycl::malloc_device<half_t>(N, q);
    std::vector<int32_t> hq((size_t)K8 * N);
    std::vector<half_t> hs((size_t)NG * N), hx(K);
    std::srand(608);
    for (auto &v : hq) v = (int32_t)(((uint32_t)std::rand() << 17) ^ (uint32_t)std::rand());
    for (auto &v : hs) v = (half_t)(0.005f + 0.02f * (std::rand() / (float)RAND_MAX));
    for (auto &v : hx) v = (half_t)(0.05f * (2.f * std::rand() / (float)RAND_MAX - 1.f));
    q.memcpy(qw, hq.data(), hq.size()*4).wait();
    q.memcpy(sc, hs.data(), hs.size()*2).wait();
    q.memcpy(x,  hx.data(), hx.size()*2).wait();

    const size_t wg = SG * SGS, nwg = (N + SGS - 1) / SGS;
    auto go = [&]{ return q.submit([&](sycl::handler &h){
        h.parallel_for(sycl::nd_range<1>(nwg*wg, wg),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int tid = it.get_local_id(0), lane = tid % SG;
            const size_t n = it.get_group(0) * SGS + tid / SG;
            if (n >= (size_t)N) return;
            const int32_t *col = qw + n * (size_t)K8;
            float acc = 0.f;
            // UNROLL independent words per lane per step -> more loads in flight
            for (int base = 0; base < K8; base += SG * UNROLL) {
                int32_t pk[UNROLL];
                int     ww[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    ww[u] = base + u * SG + lane;
                    pk[u] = (ww[u] < K8) ? col[ww[u]] : 0;
                }
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    if (ww[u] >= K8) continue;
                    const float s = (float)sc[(size_t)(ww[u] / 16) * N + n];
                    // one 16-byte load for the 8 contiguous x values this word needs
                    const sycl::vec<half_t,8> xv =
                        *reinterpret_cast<const sycl::vec<half_t,8>*>(x + ww[u]*8);
                    float part = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        part += (float)(((pk[u] >> (4*j)) & 0xF) - 8) * (float)xv[j];
                    acc += part * s;
                }
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) o[n] = (half_t)acc;
          }); }); };

    for (int i = 0; i < 5; ++i) go().wait();
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        go().wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1-t0).count());
    }
    if (out_err) {
        std::vector<half_t> ho(N); q.memcpy(ho.data(), o, N*2).wait();
        double worst = 0;
        for (int n2 = 0; n2 < std::min(N, 32); ++n2) {
            double ref = 0;
            for (int w = 0; w < K8; ++w) {
                int32_t p = hq[(size_t)n2*K8 + w];
                double s = (double)(float)hs[(size_t)(w/16)*N + n2];
                for (int j = 0; j < 8; ++j)
                    ref += (double)(((p >> (4*j)) & 0xF) - 8) * (double)(float)hx[w*8+j] * s;
            }
            worst = std::max(worst, std::fabs((double)(float)ho[n2]-ref)/(std::fabs(ref)+1e-6));
        }
        *out_err = worst;
    }
    sycl::free(qw,q); sycl::free(sc,q); sycl::free(x,q); sycl::free(o,q);
    return best;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int K = 5120, N = 17408;
    const double bytes = (double)(K/8)*N*4.0 + (double)(K/128)*N*2.0;
    const double vendor = 393.8;
    std::printf("  shape %dx%d = %.1f MB   vendor %.0f GB/s\n\n", K, N, bytes/1e6, vendor);
    std::printf("  %6s %6s %8s %10s %9s %8s\n", "SG", "SGS", "UNROLL", "us", "GB/s", "ratio");
    std::printf("  -------------------------------------------------------\n");
    double err = 0, best = 1e30; const char *bl = "";
    static char lbl[64];
#define TRY(sg, sgs, un) { double t = run<sg,sgs,un>(q, K, N, 25, &err); \
    double g = bytes/t/1e9; \
    std::printf("  %6d %6d %8d %9.1f %9.1f %7.2fx%s\n", sg, sgs, un, t*1e6, g, g/vendor, \
                err > 1e-2 ? "  BAD" : ""); \
    if (t < best) { best = t; std::snprintf(lbl,64,"SG=%d SGS=%d UNROLL=%d",sg,sgs,un); bl=lbl; } }
    TRY(32, 2, 4) TRY(32, 4, 2) TRY(32, 4, 4) TRY(32, 4, 8)
    TRY(32, 8, 2) TRY(32, 8, 4) TRY(32, 2, 8) TRY(32, 1, 4)
    std::printf("  -------------------------------------------------------\n");
    std::printf("  best: %s -> %.1f GB/s (%.2fx vendor), verify err %.2e\n",
                bl, bytes/best/1e9, bytes/best/1e9/vendor, err);
    return 0;
}

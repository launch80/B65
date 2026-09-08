// What can this card actually READ? Establish the ceiling before optimising toward it.
//
// The project has been quoting a 587 GB/s "wall" from a streaming benchmark, and
// calling 91% of it "at the wall". But the W3 read skeleton measured 591 on the GEMV
// access pattern - ABOVE that figure - which means the wall number is not a ceiling,
// it is just one measurement. Datasheet is 608.
//
// Four probes, all cache-cold over rotated buffers, in-order queue:
//   1. grid-stride streaming read, vec4        - the classic wall measurement
//   2. same, vec8 (32 B/lane)                  - is the wall load-width limited?
//   3. GEMV pattern, 46 MB, read-only          - the ceiling our int4 kernel chases
//   4. GEMV pattern, multiple buffers in flight - does more MLP help?
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
constexpr double DATASHEET = 608.0, QUOTED_WALL = 587.0;

// Buffers MUST be filled with incompressible data. Arc does lossless memory
// compression, so a memset-to-a-constant buffer never moves the bytes it claims to
// and the benchmark reports rates above the datasheet - which is how this was caught.
static std::vector<int32_t> random_fill(size_t nelem) {
    std::vector<int32_t> h(nelem);
    std::srand(608);
    for (auto &v : h) v = (int32_t)(((uint32_t)std::rand() << 17) ^ (uint32_t)std::rand());
    return h;
}

template <int V, int U> class StreamK;
template <int V, int U>
double stream_read(sycl::queue &q, size_t bytes, int nbuf, int reps) {
    using v_t = sycl::vec<int32_t, V>;
    const size_t nelem = bytes / 4, nvec = nelem / V;
    auto h = random_fill(nelem);
    std::vector<int32_t*> bufs;
    for (int i = 0; i < nbuf; ++i) {
        bufs.push_back(sycl::malloc_device<int32_t>(nelem, q));
        h[i % nelem] ^= 0x5a5a5a5a;              // make each buffer distinct
        q.memcpy(bufs.back(), h.data(), bytes).wait();
    }
    auto *out = sycl::malloc_device<int32_t>(1024, q);
    const size_t wg = 256, total = 1024 * 256;
    auto go = [&](int bi) {
        const int32_t *p = bufs[bi % nbuf];
        return q.submit([&](sycl::handler &h) {
            h.parallel_for<StreamK<V,U>>(sycl::nd_range<1>(total, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    const size_t gid = it.get_global_id(0), stride = total;
                    int32_t acc = 0;
                    for (size_t i = gid; i < nvec; i += stride) {
                        const v_t v = *reinterpret_cast<const v_t*>(p + i * V);
#pragma unroll
                        for (int e = 0; e < V; ++e) acc ^= v[e];
                    }
                    if (acc == 0x7fffffff) out[gid % 1024] = acc;
                });
        });
    };
    for (int i = 0; i < 3; ++i) go(i).wait();
    double best = 1e30;
    for (int r = 0; r < 8; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) go(i);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1-t0).count()/reps);
    }
    for (auto *b : bufs) sycl::free(b, q);
    sycl::free(out, q);
    return bytes / best / 1e9;
}

// The GEMV pattern: one sub-group per output column, marching a contiguous strip.
template <int V, int SGS> class GemvK;
template <int V, int SGS>
double gemv_read(sycl::queue &q, int K, int N, int nbuf, int reps) {
    using v_t = sycl::vec<int32_t, V>;
    const int W = K / 8;                    // int32 per column, 4-bit
    const int NV = W / V;
    const size_t words = (size_t)W * N, bytes = words * 4;
    auto h = random_fill(words);
    std::vector<int32_t*> bufs;
    for (int i = 0; i < nbuf; ++i) {
        bufs.push_back(sycl::malloc_device<int32_t>(words, q));
        h[i % words] ^= 0x5a5a5a5a;
        q.memcpy(bufs.back(), h.data(), bytes).wait();
    }
    auto *out = sycl::malloc_device<int32_t>(N, q);
    const size_t wg = 32 * SGS, nwg = (N + SGS - 1) / SGS;
    auto go = [&](int bi) {
        const int32_t *qw = bufs[bi % nbuf];
        return q.submit([&](sycl::handler &h) {
            h.parallel_for<GemvK<V,SGS>>(sycl::nd_range<1>(nwg*wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    auto sg = it.get_sub_group();
                    const int tid = it.get_local_id(0), lane = tid % 32;
                    const size_t n = it.get_group(0) * SGS + tid / 32;
                    if (n >= (size_t)N) return;
                    const int32_t *col = qw + n * (size_t)W;
                    int32_t acc = 0;
                    for (int u = lane; u < NV; u += 32) {
                        const v_t v = *reinterpret_cast<const v_t*>(col + (size_t)u * V);
#pragma unroll
                        for (int e = 0; e < V; ++e) acc ^= v[e];
                    }
                    acc = sycl::reduce_over_group(sg, acc, sycl::bit_xor<int32_t>());
                    if (lane == 0) out[n] = acc;
                });
        });
    };
    for (int i = 0; i < 3; ++i) go(i).wait();
    double best = 1e30;
    for (int r = 0; r < 8; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) go(i);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1-t0).count()/reps);
    }
    for (auto *b : bufs) sycl::free(b, q);
    sycl::free(out, q);
    return bytes / best / 1e9;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto d = q.get_device();
    std::printf("device: %s\n", d.get_info<sycl::info::device::name>().c_str());
    if (d.get_info<sycl::info::device::global_mem_size>() < (20ull<<30)) {
        std::printf("REFUSING: not the B65\n"); return 2; }
    std::printf("  datasheet %.0f GB/s   project's quoted wall %.0f\n\n", DATASHEET, QUOTED_WALL);
    auto row = [&](const char *nm, double g) {
        std::printf("  %-42s %7.1f GB/s  %5.1f%% of wall  %5.1f%% of datasheet\n",
                    nm, g, g/QUOTED_WALL*100, g/DATASHEET*100);
    };
    std::printf("  --- pure streaming read, 256 MB rotated ---\n");
    row("grid-stride vec4 (16 B/lane)",  stream_read<4,1>(q, 128u<<20, 6, 20));
    row("grid-stride vec8 (32 B/lane)",  stream_read<8,1>(q, 128u<<20, 6, 20));
    row("grid-stride vec2 (8 B/lane)",   stream_read<2,1>(q, 128u<<20, 6, 20));
    row("grid-stride vec1 (4 B/lane)",   stream_read<1,1>(q, 128u<<20, 6, 20));
    std::printf("\n  --- GEMV access pattern, 5120x17408 = 46.0 MB, 11 buffers ---\n");
    row("sub-group per column, vec4, SGS=1", gemv_read<4,1>(q, 5120, 17408, 11, 50));
    row("sub-group per column, vec4, SGS=4", gemv_read<4,4>(q, 5120, 17408, 11, 50));
    row("sub-group per column, vec8, SGS=1", gemv_read<8,1>(q, 5120, 17408, 11, 50));
    row("sub-group per column, vec2, SGS=1", gemv_read<2,1>(q, 5120, 17408, 11, 50));
    std::printf("\n  our int4 GEMV runs at 537.9; vendor at 528.\n");
    return 0;
}

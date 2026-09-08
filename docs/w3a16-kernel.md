# The Launch80 W3A16 kernel — beating Intel's int4 GEMV at 3 bits

## Result

Measured on the B65 (BMG-G31), cache-cold, rotated buffers, in-order queue, both
kernels timed in one process. Vendor is `torch.ops._xpu_C.int4_gemm_w4a16`
(oneDNN v3.12.0 `jit:gemm:any`).

| K × N | 4b MB | 3b MB | rel err | vendor µs | **W3 µs** | **speedup** |
|---|---|---|---|---|---|---|
| 5120×17408 | 46.0 | 34.8 | 6.43e-04 | 87.0 | **74.3** | **1.17×** |
| 17408×5120 | 46.0 | 34.8 | 6.17e-04 | 83.1 | **70.9** | **1.17×** |
| 5120×10240 | 27.0 | 20.5 | 6.45e-04 | 53.0 | **46.2** | **1.15×** |
| 5120×12288 | 32.4 | 24.6 | 6.37e-04 | 63.1 | **54.3** | **1.16×** |
| 6144×5120 | 16.2 | 12.3 | 6.43e-04 | 32.7 | **27.6** | **1.19×** |
| 5120×6144 | 16.2 | 12.3 | 6.50e-04 | 35.7 | **31.5** | **1.14×** |
| **400-linear mix** | | | | **23.82 ms** | **20.41 ms** | **1.17×** |

**468.8 GB/s** on the reference shape against a break-even of 399.7. Body at 0.857×
of the vendor projects end-to-end to **~87.5 tok/s from 77.2**.

Nothing else ships a fast 3-bit weight-only GEMV — Marlin and Machete reject
`bits != 4,8`, BitBLAS has no INT3 row, Intel's ARK lists INT4/INT8 only for
Battlemage, bestla has int2/int4.

## The format

32 weights in exactly 3 int32 words — 96 bits of payload in 96 bits, zero waste.
The shape is dictated by making dequant possible with the half2 trick:
`(w >> s) & 0x00070007 | 0x64006400` reinterpreted as `half2` *is* the pair
`(1024+lo, 1024+hi)`, because `0x6400` is fp16 1024.0 and the low mantissa bits
carry the field. That needs both 3-bit fields at the same offset within their
16-bit half.

Within a 32-bit word the usable shifts are `s ∈ {0,3,6,9,12}` — five pairs, filling
bits 0–14 and 16–30. Bits 15 and 31 are unreachable. So three words carry **30
weights as 15 cheap pairs and leave exactly 6 spare bits, which is exactly the 2
remaining weights**. This is why 10-values-per-word (6.25% wasted bandwidth) is
unnecessary.

The two stragglers are placed so all six spare bits sit at bit 15 or 31, letting one
masked merge collect both:

```
u = (w0 & 0x80008000) | ((w1 & 0x80008000) >> 1) | ((w2 & 0x80008000) >> 2)
(u >> 13) & 0x00070007      // == the half2 pair (weight30, weight31)
```

12 ops instead of the ~25 that building two integers and converting them cost.

## Four decisions, all made by measurement, three of them against my prediction

**1. Column-major beat k-major by 32%** (506 vs 344 GB/s on the read path). k-major
makes `x[k]` a broadcast, coalesces the scale read, and removes the cross-lane
reduction — I expected it to win. It loses because its three words sit ~68 KB apart
and that destroys DRAM page locality. Sequential streaming down a contiguous column
strip beats every structural advantage k-major has.

**2. `grf_size<256>` costs 15%** (576 → 488). oneDNN's JIT selects `grf256` on
nearly every high-scoring candidate for this exact problem, so copying it looked
obvious. It halves the hardware thread count, and a GEMV needs latency hiding far
more than it needs registers. Prefetch was likewise a no-op — the hardware
prefetcher already handles a sequential stream.

**3. Narrow loads with perfect lane balance beat wide loads with poor balance.**
A `vec4` unit covers 4 blocks, so K=5120 gives 40 units against 32 lanes: one full
round, then a round where 24 of 32 lanes idle. Dropping to **one 32-weight block per
lane-iteration** — scalar loads — took 5120×17408 from 0.79× to 1.00×. `NBLK = K/32`
is a multiple of 32 for every real shape, so the balance is exact. The per-shape data
had shown this before the fix: lane efficiency 62% / 75% / 85% tracked speedup
0.81 / 0.84 / 0.87 almost exactly.

**4. Column blocking, which had previously collapsed, is worth 1.02× → 1.17×.**
`x` is identical for every output column, so with NCOL=2 each block's four `h8`
activation loads are issued once and reused from registers. An earlier attempt at
this measured 264 → 71 → 39 GB/s and was written off — but that was at BPU=4, where
staging 256 bytes of x per lane costs ~64 GRF and spills. At BPU=1 only 4 `h8` are
live. **The idea was right and the configuration was wrong.** NCOL=8 still spills
(0.77×), so 2 is a real optimum, not a monotonic knob.

## Correctness

The format is validated by adversarial tests in `w3_format.py`, built around the
fact that a wrong bit layout does **not** crash — against i.i.d. Gaussian weights it
yields plausible values with the right variance, because Gaussians are exchangeable.
Every test destroys that exchangeability:

- **Ordinal values** `q[k,n] = (7k + 3n) % 8` — every value encodes its own position.
- **Bit bijection** — set one bit of one weight at a time and confirm each of the 96
  payload bits is written exactly once. Catches overlapping fields, which random data
  hides almost perfectly.
- **One-hot sweep** over all 32 slots × 7 values, checking nothing else moves.
- **half2 reachability** — for each of the 15 pairs, confirm `(w >> s) & 0x00070007`
  recovers exactly weights 2p and 2p+1.
- An **independently written** unpacker, from the spec rather than by inverting the
  packer.

Against the GPU kernel, `rel err ≈ 6.4e-04` uniformly across all six shapes,
consistent with fp16 rounding (the vendor itself runs `attr-fpmath:f16:true`).

## Integration constraint

The op submits on `c10::xpu::getCurrentXPUStream().queue()`. This is not optional:
`VLLM_XPU_ENABLE_XPU_GRAPH=1` runs decode under real stream capture, and a kernel on
a private `sycl::queue` is never recorded — it replays the warm-up buffer, silently,
with no error. Measured in `research/r8/graphprobe`.

## Files

- `research/r8/l80kernel/w3_l80.cpp` — the kernel, as a torch custom op
- `research/r8/l80kernel/w3_format.py` — packer, independent unpacker, adversarial tests
- `research/r8/l80kernel/w4_l80.cpp` — the 4-bit kernel from the Stage 2 kill gate
  (0.975× vendor), kept because it isolates "can we write a wall-class GEMV" from
  "is 3-bit viable"
- `research/r8/l80kernel/bench_w3.py` — the head-to-head harness

## Remaining headroom

The no-straggler ablation runs at 488.7 vs 468.8, so weights 30/31 still cost ~4%.
The three-plane `vec4` read path measured 591 GB/s, so the kernel is still ~26% above
its own read floor — the gap is dequant ALU (5 ops per 2 weights: shift, and, or,
subtract, fma). Folding the subtract is not available: the fp16 accumulator would
hold `1024·Σx`, which is ~50× the signal and destroys it.

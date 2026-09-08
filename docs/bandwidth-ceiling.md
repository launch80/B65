# How much of the 608 GB/s is actually reachable

I previously called the 4-bit path "exhausted" on the grounds that it runs at 91.6%
of a 587 GB/s wall. That was the wrong frame and the wrong number. This is the full
accounting.

## The wall figure was never a ceiling, and one measurement of it was wrong

The project has quoted a **587 GB/s** wall from a streaming benchmark. Re-measuring
with the GEMV access pattern gives **579.4 GB/s**, and pure grid-stride streaming
gives **579.8** — the two agree to 0.07%, so the GEMV pattern already reaches the
full read rate of the machine. The datasheet is 608, so **95.3% of datasheet is what
this card actually delivers on reads.** The missing 5% is DRAM overhead — refresh,
page misses, command bus — and no kernel design recovers it.

**A caution that cost me a round.** The first version of this benchmark filled buffers
with `memset` to a constant byte and reported **1073 GB/s — 177% of datasheet.** Arc
does lossless memory compression, so a constant-filled buffer never moves the bytes it
claims to. Any read benchmark on this hardware must use incompressible data;
`research/r10/ceiling.cpp` now fills from `rand()`. The earlier 591 GB/s skeleton
figure used random data and stands.

| | GB/s | % of 608 |
|---|---|---|
| pure streaming read, vec4, incompressible | **579.8** | 95.4% |
| **GEMV access pattern, read-only** | **579.4** | **95.3%** |
| our int4 GEMV (W4A16, NCOL=2) | 537.0 | 88.3% |
| Intel `int4_gemm_w4a16` | 528 | 86.8% |
| vec1 (4 B/lane) — why load width matters | 378.4 | 62.2% |

## Where every microsecond goes, on 5120x17408 (46.0 MB)

| | us | note |
|---|---|---|
| pure read floor | **79.3** | 579.4 GB/s; nothing can beat this |
| our kernel, dequant removed | 83.1 | +4.8%: activation loads + cross-lane reduce |
| our kernel, full | **85.6** | +7.2% on top: dequant ALU |
| Intel's kernel | 87.0 | |

So the **maximum any 4-bit GEMV can beat the vendor by is 87.0 / 79.3 = 1.10x**, and
we are at 1.02x. The headroom is real - I was wrong to say exhausted - but it is
bounded at ~10%, and essentially all of it is dequant ALU (7.2%), with a further ~5%
in loads and the reduction that every GEMV must pay.

Removing dequant *entirely* is worth 4.4-7.2% depending on shape.

## dp4a: the obvious route, and why it failed

At 4 bits the nibbles line up with byte lanes, so `w & 0x0F0F0F0F` yields four weights
as int8 lanes in one op and a 4-way integer dot product would do the MACs. That is
~0.6 ALU ops per weight against the fp16 half2 path's 2.5 - a 4x reduction, and with
int32 accumulation the zero point folds exactly (`sum((q-8)x) = sum(qx) - 8 sum(x)`),
which is the objection that killed the same trick in fp16.

Implemented as `research/r8/l80kernel/w4a8_l80.cpp` using
`sycl::ext::oneapi::dot_acc`. It is numerically correct (kernel rel err 3.9e-04) and
**0.82x - slower than the fp16 kernel.**

The reason is in the header source, `sycl/ext/oneapi/dot_product.hpp`:

```c++
int32_t dot_acc(uint32_t pa, int32_t pb, int32_t c) {
  Uu a = *(reinterpret_cast<Uu *>(&pa));
  Us b = *(reinterpret_cast<Us *>(&pb));
  return a.s[0]*b.s[0] + a.s[1]*b.s[1] + a.s[2]*b.s[2] + a.s[3]*b.s[3] + c;
}
```

`dot_acc` is a **portability shim, not a hardware mapping** - four byte extracts, four
multiplies, four adds. So it is an ALU *increase* over half2, which does eight MACs in
four `hfma2` instructions. The measured 0.82x is exactly that.

A native dp4a does exist on this hardware, but only through **ESIMD**
(`sycl/ext/intel/esimd/math.hpp`). That is the one place the portable-SYCL-first
decision explicitly allows dropping to ESIMD: a specific gap that measurement shows
the portable path cannot express.

## What is and is not worth doing

- **Native dp4a via ESIMD** could capture the 4.4-7.2% ALU, taking us from 1.02x to
  roughly 1.08-1.10x. It requires int8 activations, which is a real quality change
  (measured activation-quantization error ~8.5e-03 relative) needing a gate. Intel
  ships the same tradeoff as `int4_gemm_w4a8`, so it is an accepted one, but it is not
  free the way the fp16 kernel is.
- **Nothing else in the kernel.** The access pattern already reads at the machine's
  full rate; load width is the one structural lever and vec4 is already optimal
  (vec1 costs 35%).
- **The real lever remains bytes, not rate.** At 95% of achievable read bandwidth,
  the only large win left is moving less data per token - which is the quantization
  problem, and 3-bit failed it on quality (`docs/15-w3-quantization-verdict.md`).

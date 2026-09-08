# Stage 0b + Stage 1 — graph capture, and can the 3-bit layout be read at the wall?

## Stage 0b — custom SYCL ops under XPU graph capture: PASS

`VLLM_XPU_ENABLE_XPU_GRAPH=1` does not use torch.compile. vLLM shims
`torch.cuda.graph -> torch.xpu.graph` and `CUDAGraph -> torch.xpu.XPUGraph`
(`vllm/v1/worker/xpu_model_runner.py:59-63`), so decode runs under **real stream
capture**. `supports_xpu_graph()` requires torch >= 2.11.0.dev; the image has 2.13.0+xpu.

The risk is therefore not "do custom ops work" but *which queue the kernel is
submitted on*. `research/r8/graphprobe/` defines two ops that differ only in that:

| op | queue | result |
|---|---|---|
| `scale_stream` | `c10::xpu::getCurrentXPUStream().queue()` | **REPLAY OK** |
| `scale_ownqueue` | its own `sycl::queue` | **REPLAY STALE** — returned the warm-up value, no error |

The control is the point. A kernel on a private queue is never recorded by capture,
so replay silently reuses whatever was in the output buffer. Nothing raises. The
model would produce plausible, wrong logits.

**Requirement for the W3A16 op: submit on torch's current XPU stream, and never
call `queue::wait()` inside the op.**

## Stage 1 — the read path: PASS at 591 GB/s

Skeleton only: stream exactly the bytes a W3A16 GEMV would stream (3 bits/weight +
one fp16 scale per 128 = 34.82 MB) and XOR them so nothing is elided. No dequant,
no activations.

### Orientation was decided by measurement, and the intuition was wrong

| layout | thread mapping | GB/s | % wall |
|---|---|---|---|
| **A — NT / column-major** `qw[n*W + w]` | sub-group per column, lanes split K | **506** | 86% |
| B — k-major `qw[w*N + n]` | one lane per column | 344 | 59% |

k-major *looks* better on paper — weight and scale reads both coalesce across `n`,
`x[k]` becomes a broadcast instead of a re-read, and the cross-lane reduction
disappears. It is 32% slower. Each lane must read three words ~68 KB apart, and
that destroys DRAM page locality. Sequential streaming down a contiguous column
strip beats every structural advantage k-major has. **The 5.6x activation-traffic
reduction I expected from k-major is real and irrelevant.**

### The levers, on layout A

| change | GB/s | note |
|---|---|---|
| scalar loads (4 B/lane) — what our int4 kernel does today | 470 | |
| `vec<int32,2>` | 542 | |
| **`vec<int32,4>`** (512 B per sub-group instruction) | **576** | +22% |
| + `joint_prefetch` distance 1 / 2 | 573 / 568 | **no help** |
| + `grf_size<256>` | 488 | **−15%** |
| **three-plane layout, vec4 per plane** | **591** | **100.7% of wall** |

Two vendor-inspired levers refuted:

- **Prefetch does nothing.** The hardware prefetcher already handles a sequential
  column stream; the explicit prefetch only adds instructions.
- **`grf_size<256>` costs 15%.** oneDNN's JIT selects `grf256` on nearly every
  high-scoring candidate for this problem, so it was tempting to copy. But it
  halves the hardware thread count, and a GEMV needs latency hiding far more than
  it needs registers. The vendor picks it for GEMM tiles that actually use them.

### The three-plane layout

Within a column's contiguous strip, word 0 of every 32-weight block goes in plane 0,
word 1 in plane 1, word 2 in plane 2. A lane reads a `vec<int32,4>` from each plane
at the same index and holds **4 complete blocks = 128 weights = exactly one scale
group = exactly 256 bytes of x**. Three perfectly coalesced streams, no straddle
across a load boundary, and the scale becomes one register load per unit instead of
the 8-cache-line gather our int4 kernel does.

This reads at **591 GB/s, 100.7% of the 587 GB/s wall** — the wall figure is a
streaming-read measurement and this slightly exceeds it, so the read path is
effectively perfect and there is nothing left to win there.

## What the ablations say about the rest of the kernel

| variant | GB/s | % of read |
|---|---|---|
| read only | 591 | 100% |
| + dequant, no x | 417 | 71% |
| + dequant + x (vec8 loads) | **264** | 45% |

**264 is below the 358 break-even.** The read path is solved; the kernel is now
**ALU- and load-issue-bound**, exactly where the plan said the risk was.

Two identified causes, both addressable and both our own format's fault:

1. **Dequant is ~6 ops per weight** (shift, mask, int->float, subtract, multiply,
   add). The EXL2 half2 trick — `(w & 0x00070007) | 0x6400` reinterpreted as a
   `half2`, then one `hsub2` and one `hfma2` — does two weights in ~5 ops instead of
   12. It needs the two 3-bit fields at the *same* offset within each 16-bit half,
   which is a packing choice we own.

   The dense 3-word block can accommodate it with **zero waste**: put 5 fields at
   bits 0,3,6,9,12 and 5 at 16,19,22,25,28 of each word (30 fields), leaving bits 15
   and 31 spare in each of the 3 words — 6 spare bits, exactly the 2 remaining
   fields. 30 of 32 weights become half2-aligned and 2 are assembled from the spare
   bits. This is what ExLlamaV2's offline `shuffle_3bit_32` is for, and it is why
   10-values-per-word (6.25% waste) is not necessary.

2. **x costs 16 load instructions per 3 weight loads.** Each lane loads 256 bytes of
   x per 48 bytes of weight. The bytes are free (x is 10 KB, L1-resident) but the
   issue slots are not. Fix by consuming x one 32-weight block at a time (4 loads)
   rather than staging all 128 up front.

### N-blocking: refuted, decisively

| NBLK | GB/s |
|---|---|
| 1 | 264 |
| 2 | 71 |
| 4 | 39 |

Not a small loss — a collapse, from register spilling. Staging 256 bytes of x per
lane already costs ~64 GRF registers; multiplying that by NBLK plus per-column
accumulators and three `vec4`s spills to scratch. The analysis that predicted
N-blocking would not help (activation re-read is ~10% of L1 bandwidth, and the real
constraint is dequant ALU, which N-blocking does nothing for) is confirmed, though
for an additional reason it did not anticipate.

**Keep NBLK=1.** Revisit only if the x-load restructuring in (2) frees enough
registers to make it cheap, and only if measurement then asks for it.

## Status

- Stage 0b gate: **PASS** — and the silent-failure mode is characterised.
- Stage 1 gate (read path >= 550 GB/s): **PASS at 591**, 100.7% of wall.
- Naive full kernel: 264 GB/s vs 358 break-even. The remaining work is entirely
  dequant cost and load-issue pressure, not bandwidth.

# Stage 0 — the measurement floor for W3A16

Every gate in the W3A16 kernel plan is expressed relative to the vendor's rate, so
that rate had to be re-established. Doing so overturned three numbers the project
had been reasoning with, one of which was measured on the wrong GPU.

## The wrong-GPU error

This machine has **two** Battlemage cards plus an iGPU:

| PCI | device | node | cap |
|---|---|---|---|
| `00:02.0` | Arrow Lake-S iGPU | `renderD128` | — |
| `04:00.0` | Battlemage **G21**, Arc B580 (`8086:e20b`) | `renderD129` | 190 W (hwmon6) |
| `86:00.0` | Battlemage **G31**, Arc Pro B65 (`8086:e222`) | `renderD130` | 200 W (hwmon8) |

Running a container with `--device /dev/dri` exposes all three, and
`ZE_AFFINITY_MASK=0` then selects the **B580**. A full benchmark completes and
prints entirely plausible numbers for the wrong hardware — the banner said
`Arc(TM) B580 Graphics` and it is easy to read past.

The B580 measures **431.3 GB/s** cache-cold on 5120×17408 where the B65 measures
**473.6**, a 9% error in the direction of "3-bit looks harder than it is".

`research/r8/bench_floor.py` now refuses to run unless the device reports ≥20 GB,
overridable with `--allow-any-device`. **Always pass `--device /dev/dri/renderD130`,
never `/dev/dri`.** Note that `kernel/README.md` in the public repo currently tells
readers to use `renderD128`, which is the iGPU.

## Cache inflation: real, but not where we assumed

`release/bench_gemv.py` amortised over 50 back-to-back calls **reusing one weight
buffer**, so up to 18 MB of L2 could serve it. `bench_floor.py` rotates over enough
distinct buffers (~512 MB per shape) that nothing can be resident.

| K × N | MB | hot (1 buffer) | **cold (rotated)** | inflation |
|---|---|---|---|---|
| 5120×17408 | 46.0 | 476.5 | **473.6** | 0.6% |
| 17408×5120 | 46.0 | 487.5 | **485.0** | 0.5% |
| 5120×12288 | 32.4 | 484.9 | **461.3** | 5.1% |
| 5120×10240 | 27.0 | 497.4 | **457.5** | 8.7% |
| 6144×5120 | 16.2 | 625.2 | **442.7** | **41.2%** |
| 5120×6144 | 16.2 | 534.7 | **419.9** | **27.3%** |
| 5120×1024 | 2.7 | 364.2 | **293.2** | 24.2% |
| **all 400 linears** | | 493.6 | **465.6 (79% of wall)** | 6.0% |

`docs/10` argued the 46 MB shapes were safe because they exceed any plausible L2.
**That argument was correct** — they inflate by 0.6%. The inflation is entirely in
the shapes at or below the 18 MB L2, exactly as it flagged for the 16.2 MB entry.

## What did not reproduce

`docs/10` records 533 GB/s hot on 5120×17408 and 547 aggregate. On the same shape,
same protocol, same single-buffer reuse, this machine now measures **476.5 hot** and
493.6 aggregate — about 11% lower. Repeated three times the cold figure is
472.7 / 473.4 / 473.4, a spread of **0.15%**, so today's number is not noise.

The 533 is not reproducible and is not used. It does not change any gate, because
every gate is a **ratio** of the vendor rate measured in the same session.

## The gate

```
V  vendor int4, 5120x17408, cache-cold, in-order burst   473.4 GB/s   (81% of wall)
B  3-bit break-even = (34.8/46.0) * V = 0.757 * V        358.4 GB/s
   3-bit at V parity                                     473.4 -> 1.32x on the body
```

| 3-bit rate | body vs today | end-to-end | tok/s from 77.2 |
|---|---|---|---|
| 358 | 1.000 | 1.00× | 77.2 — break-even |
| 400 | 0.895 | 1.09× | 84.5 |
| 450 | 0.796 | 1.20× | 92.7 |
| 473 (vendor parity) | 0.757 | 1.25× | 96.5 |
| 500 | 0.717 | 1.30× | 100.6 |

## Our starting point, measured honestly

`w4a16_v4.cpp` is `v3` with the timing loop replaced — same kernel, correct protocol
(buffer rotation, one sync around 50 launches, **in-order queue**). The in-order
property matters: an out-of-order queue lets the 50 launches overlap and reported
303 GB/s against 281 for the same code.

| | GB/s | vs vendor |
|---|---|---|
| v3 as previously reported (per-call, wrong card) | 212 | — |
| v4 per-call, B65 | 286.7 | 0.73× of vendor per-call |
| **v4 burst, B65, in-order** | **281.3** | **0.59×** |
| vendor burst | 473.4 | 1.00× |
| break-even for 3-bit | 358.4 | 0.757× |

**The gap is 1.68×, not the 2.2× the plan assumed.** Our existing kernel — with two
known defects unfixed (4-byte-per-lane weight loads, an 8-cache-line scale gather)
and none of `grf_size<256>`, AOT, prefetch or split-K applied — needs **+27%** to
reach 3-bit break-even and +68% for vendor parity.

## The vendor kernel is oneDNN JIT — confirmed

`ONEDNN_VERBOSE=all` on a single call:

```
exec,gpu,matmul,jit:gemm:any,undef,
  src:f16::blocked:ab wei:u4::blocked:ba dst:f16::blocked:ab,
  attr-fpmath:f16:true attr-scales:wei:3:f16:128x1 attr-zero-points:wei:0:s8,
  1x5120:5120x17408
```

oneDNN **v3.12.0**, Level Zero, `binary_kernels:enabled`. The systolic path
(`jit:xe_hp:gemm:any`) is *rejected* for this problem —
`src/gpu/intel/gemm/jit_xe_hp_systolic.cpp:85`, "skipping or dispatching to another
implementation" — and it lands on the generic **gemmstone JIT generator**, whose
source is public in `src/gpu/intel/gemm/jit/`.

It scores a catalog of strategies at runtime. The candidates it considers name the
levers directly:

```
F gemm FHS T@4N@4N 16 64 aS64x2+S16@24 aB16x2+S1,32@16 aB wg 32x1 cb4 ks64 ql nb 32x0 ... grf256
F gemm FHS T@4N@4N 32 64 aS16x2     aB16+S1,64@24  aB wg 16x2 cb4 ks32 nb 0x2  ... grf256
F gemm FHS T@8N@8N 32 64 at32+m32@64 am32/16+m16@64 aB wg 8x4 ... sb64 grf256 sys kv afb
```

`grf256` on essentially every high-scoring entry, `ks32`/`ks64` k-slicing
(**split-K**), work-groups of `32x1` / `16x2` / `8x4`, `wei:u4::blocked:ba` (N-major,
the NT layout we already use). This is direct confirmation of three of the plan's
levers and tells us the vendor is not doing anything we cannot express.

## Status

Stage 0 gate **met**: V established at 473.4 GB/s, reproducible to 0.15% (gate was
<3%). `B = 358.4 GB/s`. Branch taken: the "comfortable" one — cache inflation was
not corrupting the reference shape, but the wrong-GPU error was, in our favour.

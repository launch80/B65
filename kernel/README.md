# Launch80 W3A16 — a 3-bit GEMV that beats Intel's 4-bit kernel on Battlemage

**1.17× faster than `int4_gemm_w4a16` across the model's 400 body linears**, at
`rel err 6.4e-04` (fp16 rounding). Measured on an Arc Pro B65 (BMG-G31).

| K × N | vendor µs | **W3 µs** | **speedup** |
|---|---|---|---|
| 5120×17408 | 87.0 | **74.3** | **1.17×** |
| 17408×5120 | 83.1 | **70.9** | **1.17×** |
| 5120×10240 | 53.0 | **46.2** | **1.15×** |
| 5120×12288 | 63.1 | **54.3** | **1.16×** |
| 6144×5120 | 32.7 | **27.6** | **1.19×** |
| 5120×6144 | 35.7 | **31.5** | **1.14×** |
| **400-linear mix** | **23.82 ms** | **20.41 ms** | **1.17×** |

Decode at batch 1 is read-bound — 2 op/byte against a machine balance near 581, so
the XMX array is ~99.7% idle. The only lever of size is moving fewer bytes, and the
quantized body is 82% of every decode step. 3-bit removes 24% of it.

Nothing else ships a fast 3-bit weight-only GEMV: Marlin and Machete reject
`bits != 4,8`, BitBLAS has no INT3 row, Intel's ARK lists INT4/INT8 only for
Battlemage, bestla has int2/int4 only.

> **An earlier version of this file concluded the opposite** — that hand-written
> SYCL could reach only 54% of the vendor and that 3-bit was therefore dead, and it
> recommended filing an upstream request. That conclusion was wrong for two
> measurable reasons: the comparison was taken per-call rather than under the burst
> the served model actually runs, and it was taken **on the wrong GPU** (see
> `docs/measurement-floor.md`). The recommendation is withdrawn.

## The format

32 weights in exactly 3 int32 words — 96 bits of payload in 96 bits, zero waste.

The shape is dictated by making dequant cheap. `(w >> s) & 0x00070007 | 0x64006400`
reinterpreted as `half2` *is* the pair `(1024+lo, 1024+hi)`, because `0x6400` is
fp16 1024.0 and the low mantissa bits carry the field. One `hsub2` removes the
offset and zero point, one `hfma2` multiplies by x. That needs both 3-bit fields at
the same offset within their 16-bit half, which allows only `s ∈ {0,3,6,9,12}`:
five pairs per word, so 30 weights across three words — leaving exactly 6 spare bits,
which is exactly the 2 remaining weights. This is why the common
10-values-per-32-bit-word layout (6.25% wasted bandwidth) is unnecessary.

The two stragglers are placed so all six spare bits sit at bit 15 or bit 31, so one
masked merge collects both into half2 shape:

```c
u = (w0 & 0x80008000) | ((w1 & 0x80008000) >> 1) | ((w2 & 0x80008000) >> 2);
(u >> 13) & 0x00070007        // == the pair (weight30, weight31)
```

## Files

| file | what |
|---|---|
| `w3_l80.cpp` | the kernel, as a torch custom op |
| `w3_format.py` | packer, an independently written unpacker, adversarial tests |
| `w4_l80.cpp` | the same machinery at 4 bits (0.975× vendor) — isolates "can we write a wall-class GEMV" from "is 3-bit viable" |
| `bench_w3.py` / `bench_w4.py` | head-to-head against the vendor op, one process, rotated buffers |
| `bench_floor.py` | establishes the vendor reference cache-cold; refuses to run on the wrong device |
| `layout_study.cpp` | the orientation experiment: column-major vs k-major |
| `graphprobe/` | proves a custom SYCL op survives XPU graph capture, and that a private queue silently does not |
| `sycl_w4a16_gemv.cpp`, `triton_w4a16_gemv.py` | the original de-risking pass, kept as history |

## Running it

**Pass the B65's render node explicitly.** This machine has two Battlemage cards and
an iGPU; `--device /dev/dri` plus `ZE_AFFINITY_MASK=0` silently selects the wrong
one and prints entirely plausible numbers. Find yours with
`readlink -f /dev/dri/by-path/pci-<BDF>-render`.

```sh
docker build -t p608-sycl -f Dockerfile.dpcpp .

BDF=0000:86:00.0                                     # your B65
R=$(readlink -f /dev/dri/by-path/pci-$BDF-render)
BP=$(mktemp -d); ln -s "../$(basename $R)" "$BP/pci-$BDF-render"

docker run --rm --device "$R" -v "$BP:/dev/dri/by-path:ro" --ipc=host \
  -v "$PWD:/w" -e ZE_AFFINITY_MASK=0 --entrypoint bash p608-sycl \
  -lc 'bash /w/build.sh && python /w/bench_w3.py'
```

`python w3_format.py` runs the format's correctness suite on CPU alone.

## Correctness

A wrong bit layout does not crash. Against i.i.d. Gaussian weights it yields
plausible values with the right variance, because Gaussians are exchangeable —
permuting them changes nothing you can see. Every test in `w3_format.py` is built to
destroy that exchangeability: ordinal values that encode their own position, a
bit-bijection check that each of the 96 payload bits is written exactly once, a
one-hot sweep over all 32 slots, a half2-reachability check on all 15 pairs, and an
unpacker written from the spec rather than by inverting the packer.

## Integration constraint

The op submits on `c10::xpu::getCurrentXPUStream().queue()`. This is not optional.
`VLLM_XPU_ENABLE_XPU_GRAPH=1` runs decode under real stream capture; a kernel on its
own `sycl::queue` is never recorded and replays the warm-up buffer, silently, with no
error. `graphprobe/` demonstrates both halves of that.

## Status

Measured against synthetic 3-bit codes. Quantizing the real checkpoint and serving it
end-to-end behind the perplexity / GSM8K / greedy-equivalence gates in `verify/` is
in progress.

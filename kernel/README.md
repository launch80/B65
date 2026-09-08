# Can we write our own int4 GEMV? Measured: no.

Project 608 wanted a 3-bit weight format. Nothing on XPU consumes 3-bit
(`_xpu_C` has only `int4_gemm_w4a16`, `int4_gemm_w4a8`, `fp8_gemm`), and packing
3-bit values into the 4-bit container reads 4.125 bits/weight and gains nothing.
So 3-bit needs its own kernel.

Before writing one, we asked whether we could match Intel's existing int4 kernel
on the format we already have — same shapes, same data, one variable.

| implementation | GB/s | % of vendor |
|---|---|---|
| Triton, naive | 106 | 27% |
| Triton, tuned over BLOCK_N × warps × stages | 173 | 44% |
| SYCL, first cut | 67 | 17% |
| SYCL, no SLM staging + unrolled | 198 | 50% |
| SYCL, + vector `x` loads | **212** | **54%** |
| **Intel `int4_gemm_w4a16`** | **394** | 100% |
| memory wall (measured) | 587 | |

Both are numerically correct — 2.9e-04 (Triton) and 3.7e-04 (SYCL) against the
vendor op, which is fp16 rounding.

**The arithmetic that kills it.** 3-bit reads 76% of the bytes, so a 3-bit kernel
must sustain **≥298 GB/s just to break even** against the 4-bit kernel we already
have. At 212 GB/s a 3-bit GEMV takes 165 µs where the vendor's 4-bit takes 117 —
**1.41× slower than doing nothing**.

So the quantization result (3-bit at 5.13% output error, projected +25% throughput)
is real and unreachable through a kernel we write this way. The remaining gap is
almost certainly Intel's 2D block-load path (`has_subgroup_2d_block_io`) via ESIMD
or the matrix extensions — a deeper specialisation than plain SYCL pointer loads,
and one the vendor has already done.

**The productive move is to ask upstream for a w3a16 kernel rather than write one**,
and the evidence to motivate that request is in `docs/` — 3-bit costs 5.13% output
error where round-to-nearest costs 22.34%, and the body is 82% of every decode step.

## Files

- `sycl_w4a16_gemv.cpp` — working SYCL int4 GEMV, sub-group per output column,
  configurable sub-group count and unroll, self-verifying
- `Dockerfile.dpcpp` — adds the Intel DPC++ compiler to the vLLM XPU image, so the
  Level Zero runtime and driver are identical to where the vendor numbers were taken
- `triton_w4a16_gemv.py` — the Triton equivalent, with a config sweep

```sh
docker build -t p608-sycl -f Dockerfile.dpcpp .
docker run --rm --device /dev/dri/renderD128 -v $PWD:/w p608-sycl bash -lc \
  'source /opt/intel/oneapi/setvars.sh; cd /w && icpx -fsycl -O3 sycl_w4a16_gemv.cpp -o v && ./v'
```

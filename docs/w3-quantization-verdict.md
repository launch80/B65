# Why 3-bit does not ship on this model — and what the kernel result means anyway

## The number that started this was measured in-sample

Project 608 pursued 3-bit because `research/r2/gptq_study.py` reported **5.13% output
error at 3 bits** where round-to-nearest cost 22.34%. That study used **1024
activation rows for a K=5120 Hessian** and scored the same rows it fitted. At rank
1024 << 5120, GPTQ's error compensation can cancel almost perfectly inside the
sampled subspace, so the figure is close to a measure of its own fit.

Same weights, same 1024 rows, same GPTQ code — only the scoring changes:

| | `layers.9.mlp.gate_up_proj` | `layers.9.mlp.down_proj` |
|---|---|---|
| scored **in-sample** (reproduces the study) | **5.83%** | **2.97%** |
| scored **out-of-sample** | 20.96% | 18.66% |
| 32x the calibration data, out-of-sample | 20.35% | 22.39% |

The third row is the one that settles it: **more calibration data does not help.**
This is a property of 3-bit at group 128, not a data shortfall.

Across a depth sweep (layers 0, 1, 9, 10, 31, 32, 62, 63) the out-of-sample output
error is **median 18.8%, p95 23.0%** — flat with depth, so not a first/last-layer
artifact. The plan's gate was <=6% for 95% of modules. It misses by 3-4x.

The full-model run was stopped at 56 of 256 modules once this was established.

## What was built and is worth keeping

The pipeline itself is sound and re-runnable (`research/r9/`):

- `quantize_w3.py` — layer-chunked GPTQ requantization. Hooks vLLM's 256 merged body
  linears, accumulates Hessians for a layer range, quantizes, frees, moves on;
  99 GB of Hessians would be resident otherwise, against ~11 GB free.
- It **verifies the int4 dequantisation against the module's own forward**
  (2.7e-04) before quantizing anything, because a transposed layout produces weights
  with a perfectly plausible distribution and no error anywhere.
- It holds out every 64th activation row from the Hessian and scores on those. This
  is the change that exposed the problem; the original study had no held-out set.
- Calibration uses wikitext-2 validation, **not** `verify/ppl-corpus.json` or gsm8k,
  which are the quality gates. Calibrating on the gate would have flattered it.
- `w3_pack_torch.py` — GPU packer, validated against the numpy spec on random data,
  ordinal patterns, the independent unpacker, and plane layout.
- `patch_w3.py`, `serve-w3.sh`, `run_ppl_gate.sh` — the serving path, with a W3=0/1
  toggle so both arms of a comparison run identical code.

Two facts about the checkpoint worth recording: `desc_act: false` and **no `g_idx`
tensor at all**, so there is no act-order permutation to preserve; and vLLM **merges**
linears (`in_proj_qkvz`, `gate_up_proj`), so the 400 checkpoint tensors are 256
runtime modules with 4 distinct Hessians per layer.

## The kernel result is independent and stands

**1.17x faster than Intel's `int4_gemm_w4a16` at 3 bits** is a measured kernel fact.
It does not depend on what the quantizer achieves — it says the bytes can be read and
decoded that fast, not that this model's weights survive 3 bits. See
`docs/14-w3a16-kernel.md`.

## The 4-bit path is exhausted

The same column-blocking that took W3 from 1.02x to 1.17x was ported to the 4-bit
kernel. It helps far less:

| | reference shape | 400-linear mix |
|---|---|---|
| W4A16 NCOL=1 | 512.4 (0.97x) | 0.97x |
| **W4A16 NCOL=2** | **537.9 (1.02x)** | **1.00x** |
| W4A16 NCOL=4 | 517.5 (0.98x) | 0.97x |

At **91.6% of the memory wall** there is nothing meaningful left at 4 bits. W4 already
had perfect lane balance (NV = K/32 = 160 is a multiple of the sub-group) and is
closer to bandwidth-bound, so the lever that mattered at 3 bits barely moves it. This
independently reproduces what `docs/10` concluded: the incumbent GEMV is not the
bottleneck.

## Group size was tested and is not the lever

The obvious rescue was that the failure is at group 128, and that a finer group would
recover accuracy for a modest byte cost. Tested on layers 9-10, same calibration, same
held-out scoring; group 128 reproduced the earlier numbers exactly, so the three rows
are directly comparable:

| group | bits/wt | bytes vs int4 | kernel speedup | median out err | p95 |
|---|---|---|---|---|---|
| 128 | 3.125 | 0.758x | **1.17x** | 21.18% | 23.09% |
| 64 | 3.250 | 0.788x | **1.16x** | 19.93% | 21.13% |
| 32 | 3.500 | 0.848x | **1.11x** | 17.93% | 19.10% |
| *gate* | | | | *<=6%* | |

**Halving the group twice buys 15% relative error and costs 12% more bytes.** At
roughly 6% relative improvement per halving, closing the 3.5x gap to the 6% gate would
take on the order of twenty halvings. The error is dominated by having only 8 levels,
not by scale granularity: with the grid at s = amax/3, RMS quantization error is about
s/sqrt(12) ~ 0.096*amax regardless of how finely amax is estimated. Finer groups only
reduce how much a group's outlier inflates its own scale, which is a second-order
effect.

**3-bit is not viable for this model at any practical group size.**

Worth recording separately: the kernel side of this trade is nearly free. Group 64
costs 1% of the speedup (1.17x -> 1.16x) because the kernel is ALU-bound, the dequant
op count is identical, and the scale *load* count per block does not change - only
1.4 MB more data out of 36. The kernel supports groups 128/64/32 via its BPS
parameter and is correct at all three (rel err ~6.4e-04). If another model's weights
do survive 3 bits, the fast path is already there.

## Where a deployable win would have to come from

Not from the 4-bit kernel — it is at the wall. Two candidates remain:

Not from a finer 3-bit group either - that was the obvious rescue and it is measured
dead above. What remains:

1. **The MTP draft passes**, which `docs/11-current-state.md` identifies as ~20% of the
   byte budget and quality-free by construction, since the target verifies every token.
   Not kernel work, and the largest remaining lever.
2. **A representation with more than 8 levels per weight but fewer than 16 bytes'
   worth** - the failure here is level count, so anything that keeps 4-bit's grid while
   spending fewer bytes (a shared/low-rank correction, a sparse outlier channel on top of
   3-bit) attacks the actual constraint. Speculative; nothing measured.

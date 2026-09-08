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

## Where a deployable win would have to come from

Not from the 4-bit kernel — it is at the wall. Two candidates remain:

1. **3-bit at a smaller group.** The failure above is at group 128. Group 64 is
   3.25 bits/weight (21% fewer bytes than int4's 4.125) and group 32 is 3.5
   (15% fewer); both should cut quantization error substantially, since group size is
   the standard lever for exactly this failure mode. The kernel ties one scale to 4
   blocks; supporting 2 blocks or 1 is a small change. **Untested.**
2. **The MTP draft passes**, which `docs/11-current-state.md` identifies as ~20% of
   the byte budget and quality-free by construction, since the target verifies every
   token. Not kernel work.

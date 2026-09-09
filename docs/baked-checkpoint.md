# The baked checkpoint: R3 + M1 on disk, loadable by stock vLLM

## What it is

`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` with the two components that the
boot patches R3 and M1 requantized *in VRAM* now quantized **on disk**, in the same
INT4 g128 symmetric GPTQ layout as the 400 body linears (which are byte-identical):

| tensor | on disk before | on disk now |
|---|---|---|
| `lm_head` | fp16, 2.54 GB (R3: RTN->INT4 at boot) | GPTQ INT4, Hessian-compensated, 0.66 GB |
| `mtp.fc` + `mtp.layers.0.*` (7 linears) | BF16, 0.85 GB (M1: RTN->INT4 at boot) | GPTQ INT4, 0.22 GB |

`quantize_config.json` / `config.json:quantization_config` set `lm_head: true` and drop
the `-:.*mtp.*` exclusion. Stock vLLM 0.27.2-xpu then builds both quantized natively:
`gptq_utils` honours `lm_head_quantized`, and `qwen3_5_mtp.py`'s "force draft
unquantized" gate is conditional on that exclusion key. **What is in VRAM is what is
on disk.** Total 17.05 GB.

## How it was made (`research/bake/`)

- `bake_quant.py` — runs vLLM in-process with MTP on, no weight patches; hooks the six
  input-owning modules (target `lm_head` via `logits_processor` — `ParallelLMHead` is
  never called through `forward()`; `fc`, `qkv_proj`, `o_proj`, `gate_up_proj`,
  `down_proj` on the draft); accumulates Hessians over 256 x 128-token generations of
  wikitext-2-valid (disjoint from every gate); checkpoints them to `hessians.pt`.
- `bake_gptq_resume.py` — GPTQ + pack from the checkpoint, reading the dense source
  tensors straight from the shards. Merged vLLM modules split back into HF tensors by
  quantizing each HF tensor against the shared input Hessian. Packs to the body's
  exact layout: `qweight [K/8,N]` nibble `j` at bit `4j`, `scales [K/G,N]`, `qzeros`
  all-7 (the zero-minus-one convention the source uses), `g_idx = k//128`; verified
  by an independent unpack. lm_head chunked over N (5.1 GB fp32 does not fit beside
  the model).
- `assemble.py` — physically rebuilds every shard without the 9 dense tensors (vLLM's
  loader walks `*.safetensors`; a stale `lm_head.weight` in an old shard would load
  over the baked one), adds the bake shard, rewrites the index and both configs.
- `serve-baked.sh` — serves it with **no weight patches**: the three vLLM correctness
  fixes + R6 only. `B70_MTP_BF16_DRAFT` must NOT be set (the cookbook MTP patch forces
  a dense draft when it is 1, which discards the baked INT4 draft).

Layer-wise output error on held-out activations, GPTQ vs the RTN the patches used —
roughly halved everywhere:

| tensor | GPTQ (disk) | RTN (patch) |
|---|---|---|
| `lm_head` | 3.64% | 7.58% |
| `mtp.fc` | 5.68% | 11.36% |
| q / k / v | 2.20 / 5.26 / 4.24% | 4.47 / 10.73 / 8.40% |
| o_proj | 5.31% | 12.34% |
| gate / up / down | 3.02 / 4.72 / 5.67% | 6.51 / 9.78 / 11.23% |

## Gates (same corpus and scripts as the R3 gate)

| gate | baked | patched stock (RTN head) | fp16 head |
|---|---|---|---|
| perplexity, 13,027 tokens, spec off | **6.133** | 6.144 | 6.093 |
| GSM8K n=100 @768 tokens | 82 (18 trunc) | 90 (11 trunc) | 88 (14 trunc) |
| GSM8K n=100 @1536 tokens, paired | **91** (8 trunc) | 89 (10 trunc) | — |
| spec-on vs spec-off greedy, same weights | **8/8 identical** | — | — |

The @768 drop is entirely truncation: all 10 flips are `finish=length`, zero are wrong
answers, and on the 80 problems both finished it is 80/80 vs 80/80. At 1536 the
paired result is baked 91 / current 89 (baked-only right 2, current-only 0, McNemar
0.50). **The GPTQ head is slightly more verbose and at least as accurate.** Perplexity
is better than the stack it replaces.

Draft acceptance on the fixed 8-prompt workload: 39.3% vs 36.5% (an earlier 50.5%
sample was contaminated by a concurrent GSM8K run — the spec counters are global).

## The bug this surfaced in R6

R6's prefix builder read `head.weight`; on a GPTQ head that is `None`, so R6 silently
switched off and every draft pass read the full 0.66 GB head. That alone cost ~7.5 ms
per step (52.7 vs 45.2 ms on the fixed workload), the whole BetterBench regression
(69.2 vs 77.8). Fixed in `patch_r3_lmhead_int4.py`: when the head has `qweight`, take
`V` from `scales.shape[1]` (the loaded `qweight` is `[V, K/8]` row-major, **not** the
`[K/8, V]` view — the first wrong fix assumed the latter and returned `None` on the
`n >= V` guard), and slice the first `n` rows, whose transpose is the `[K/8, n]`
stride-`(1, K/8)` view the vendor op wants. Engaged and verified.

## Throughput

BetterBench, single stream, MTP k=6, both arms measured the **same day on the same
config** (vLLM 0.27.2-xpu, GDN mixed-split v5, XPU graph capture off), weighted by the
harness's category weights:

| | code | file_edit | json | math | prose | reasoning | summ. | **weighted** |
|---|---|---|---|---|---|---|---|---|
| current stack (patched, RTN head) | 62.7 | 86.3 | 97.7 | 87.9 | 56.8 | 61.9 | 87.5 | **71.8** |
| baked (GPTQ head + draft on disk) | 66.0 | 83.5 | 94.0 | 88.5 | 57.9 | 61.4 | 86.3 | **71.8** |

**1.00x.** The baked checkpoint is speed-neutral and quality-better, with no boot-time
weight patches. Fixed 8-prompt workload agrees: 71.9 vs 70.6 tok/s, acceptance 38.0%
vs 36.5%, step 45.6 vs 45.2 ms.

L80 page for the baked run: https://launch80.com/a/2c20c65e-f477-4621-98d9-7df1b16646fe
(self-reported rendering; free-tier pages expire - `results/bake-betterbench-*.json` is
the citation of record).

Two earlier numbers are superseded and should not be quoted against this: the
09-07 baseline of 77.8 was a different config (no GDN v5, capture on), and today's
current stack measures 71.8 on the same harness. What changed between those two
configs was not re-measured, on instruction; the like-for-like comparison above is
the one that stands. The first baked run at 69.2 / 67.9 was with R6 silently off.

Published: https://huggingface.co/Launch80/Qwen3.8-27B-GPTQ-Int4-baked

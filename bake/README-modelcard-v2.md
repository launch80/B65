---
license: apache-2.0
base_model: mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
base_model_relation: quantized
tags: [gptq, int4, mtp, speculative-decoding, intel, xpu, arc]
---

# Qwen3.8-27B GPTQ-Int4 baked, v2 — MikeCaldera's body, lm_head and MTP draft on disk

**What this is.** [mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
(MikeCaldera's fresh GPTQ INT4 g128 quant of Qwen3.8-27B, GPTQModel 7.3.2) with the two
components that stock checkpoints leave dense now quantized **on disk**, in the same INT4
g128 symmetric GPTQ format as the body:

| tensor | before | now |
|---|---|---|
| `lm_head` | fp16, 2.54 GB | GPTQ INT4 on disk, Hessian-compensated, 0.66 GB |
| `mtp.*` (the MTP draft layer + `fc`) | BF16, 0.85 GB | GPTQ INT4 on disk, 0.22 GB |

The 400 body linears are **byte-identical** to MikeCaldera's. `quantize_config.json` sets
`lm_head: true` and drops the `-:.*mtp.*` exclusion, so stock vLLM loads both natively.
**What is in VRAM is what is on disk.** 17.05 GB total.

**v1 vs v2.** [v1](https://huggingface.co/Launch80/Qwen3.8-27B-GPTQ-Int4-baked) is the
same bake on SergiioB's body. The bake pipeline, calibration text (wikitext-2 valid, 256
prompts × 128 generated tokens) and settings are identical; only the body differs. On an
Arc Pro B65 the two read the same bytes per decode step, so any throughput difference
between them is draft acceptance, i.e. how closely the body tracks the BF16 model the MTP
draft was trained against.

**Why.** Under MTP speculative decoding the `lm_head` is read once per verify and the draft
layer once per draft pass; on a read-bound card those are the largest movable bytes after
the body. As boot-time patches they were worth +37% and +10% on the B65, and on this
checkpoint the whole head-and-draft treatment measured 1.67× at 6K context. This ships the
same bytes as a checkpoint nobody needs a patch stack to load.

**Measurements** (Arc Pro B65 32 GB, vLLM 0.27.2-xpu, GDN mixed-split, capture off, MTP k=6, single stream, BetterBench 0.4.0 corpus v1.0, category-weighted decode):

| | v2 (this) | v1 (SergiioB body) |
|---|---|---|
| BetterBench decode, weighted | **73.5** | 71.8 (measured 2026-09-08) |

BetterBench run page (Launch80): https://launch80.com/a/0a807aa7-36a2-4ba8-83c0-12f06ef4031b — a self-reported rendering; free-tier pages
expire, so the results JSON in the GitHub repo is the citation of record.

**Quality gates: not yet run for v2.** v1 passed perplexity, paired GSM8K and a spec-on
vs spec-off greedy invariant; those results do **not** transfer to a different body. Treat
v2 as a throughput candidate until the same gates are published here.

Layer-wise output error on held-out activations (123k rows for `lm_head`, 212k for the
draft), GPTQ on disk vs round-to-nearest — GPTQ roughly halves it everywhere, and the
figures match v1 within 0.1 points:

| tensor | GPTQ (this) | RTN |
|---|---|---|
| `lm_head` | 3.60% | 7.55% |
| `mtp.fc` | 5.57% | 11.34% |
| draft q / k / v | 2.16 / 5.15 / 4.16% | 4.47 / 10.71 / 8.44% |
| draft o_proj | 5.31% | 12.37% |
| draft gate / up / down | 2.95 / 4.68 / 5.62% | 6.45 / 9.84 / 11.34% |

**Serving (Intel XPU).** `vllm serve <this> --quantization gptq --dtype float16
--kv-cache-dtype fp8 --max-num-seqs 32
--speculative-config '{"method":"mtp","num_speculative_tokens":6}'`. On vLLM 0.27.2-xpu
you still need three correctness patches from the
[B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) (MTP
nightly, MTP boundary, and `patch_gdn_mixed_split_v5.py` — without the last one the
engine dies under any concurrent load). None of them touch weights. Turn XPU graph
capture **off**. The optional draft-vocab prefix (`P608_DRAFT_VOCAB`, +6% on the B65)
needs `patches/p608_lmhead_int4.py` dated 2026-09-08 or later from the repo below.

**Credit.** Qwen3.8-27B by Qwen (Apache-2.0). INT4 body and MTP head by MikeCaldera
([repro](https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm)); the bake method
follows SergiioB's cookbook. The `lm_head`/draft quantization and measurements by Launch80:
https://github.com/launch80/B65. Every number above has a script and a raw JSON there.

---
license: apache-2.0
base_model: SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
base_model_relation: quantized
tags: [gptq, int4, mtp, speculative-decoding, intel, xpu, arc]
---

# Qwen3.8-27B GPTQ-Int4 — lm_head and MTP draft baked in

**What this is.** [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
with two components that were previously requantized *in GPU memory at boot* by patches
now quantized **on disk**, in the same INT4 g128 symmetric GPTQ format as the body:

| tensor | before | now |
|---|---|---|
| `lm_head` | fp16, 2.54 GB, RTN→INT4 at boot | GPTQ INT4 on disk, Hessian-compensated, 0.66 GB |
| `mtp.*` (the MTP draft layer + `fc`) | BF16, 0.85 GB, RTN→INT4 at boot | GPTQ INT4 on disk, 0.22 GB |

The 400 body linears are **byte-identical** to SergiioB's. `quantize_config.json` sets
`lm_head: true` and drops the `-:.*mtp.*` exclusion, so stock vLLM loads both natively.
**What is in VRAM is what is on disk.**

**Why.** Under MTP speculative decoding the `lm_head` is read once per verify and the draft
layer once per draft pass; on a read-bound card those two are the largest movable bytes
after the body. Quantizing them was worth +37% and +10% on an Arc Pro B65 — but as boot
patches, which nobody else could reproduce without the patch stack. This ships the same
bytes as a checkpoint.

**Quality gates** (vs the patched stock checkpoint, same server, same corpus):

| gate | this checkpoint | patched stock (RTN head) | fp16 head |
|---|---|---|---|
| perplexity, 13,027 held-out tokens, spec off | **6.133** | 6.144 | 6.093 |
| GSM8K n=100, 768-token budget | 82 (18 truncated) | 90 (11 truncated) | 88 (14 truncated) |
| GSM8K n=100, 1536-token budget, paired | **91** (8 truncated) | 89 (10 truncated) | — |
| spec-on vs spec-off greedy, same weights, 8 × 48 tok | **8/8 identical** | — | — |
| BetterBench decode, MTP k=6, weighted, same day/config | **71.8** | 71.8 | — |

Perplexity is **better** than the patched stock checkpoint it replaces (−0.19%) and +0.65%
over the fp16 head, against a +1.5% gate. The GSM8K drop at 768 tokens is entirely
truncation: all 10 flipped problems ran out of budget mid-reasoning, zero were wrong
answers, and on the 80 problems both heads finished they score 80/80 each. At 1536
tokens the paired result is baked 91 / stock 89 (McNemar 0.50, not significant). The GPTQ
head makes the model slightly more verbose; it does not make it less accurate. Draft
acceptance on a fixed workload: 38.0% vs 36.5% (Hessian-compensated draft vs round-to-nearest).

BetterBench run page (Launch80): https://launch80.com/a/2c20c65e-f477-4621-98d9-7df1b16646fe — a self-reported rendering; free-tier pages expire,
so the results JSON in the GitHub repo is the citation of record.

Layer-wise output error on held-out activations (118k rows for `lm_head`, 207k for the
draft), GPTQ on disk vs the RTN the boot patches used — GPTQ roughly halves it everywhere:

| tensor | GPTQ (this) | RTN (patches) |
|---|---|---|
| `lm_head` | 3.64% | 7.58% |
| `mtp.fc` | 5.68% | 11.36% |
| draft q / k / v | 2.20 / 5.26 / 4.24% | 4.47 / 10.73 / 8.40% |
| draft o_proj | 5.31% | 12.34% |
| draft gate / up / down | 3.02 / 4.72 / 5.67% | 6.51 / 9.78 / 11.23% |

**Serving (Intel XPU).** `vllm serve <this> --quantization gptq --dtype float16
--speculative-config '{"method":"mtp","num_speculative_tokens":6}'`. On vLLM 0.27.2-xpu
you still need three correctness patches from the
[B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) (MTP
nightly, MTP boundary, and `patch_gdn_mixed_split_v5.py` — without the last one the
engine dies under any concurrent load). None of them touch weights. Turn XPU graph
capture **off**; it is worth 0.0% on this image and slows the draft ~1%.

If you also use the draft-vocab prefix (`P608_DRAFT_VOCAB`, +6% on the B65), you need the
version of `patches/p608_lmhead_int4.py` in the repo above dated 2026-09-08 or later: on a
GPTQ-quantized head the older one found no `.weight`, silently disabled the prefix, and
every draft pass read the full 0.66 GB head (about −11% end to end).

**Credit.** Qwen3.8-27B by Qwen (Apache-2.0). INT4 body and MTP head by SergiioB. The
`lm_head`/draft quantization, quality gates and measurements by Launch80:
https://github.com/launch80/B65. Every number above has a script and a raw JSON there.

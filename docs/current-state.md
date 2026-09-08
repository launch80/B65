# Project 608 — current state, 2026-09-07

**73.1 tok/s decode**, from 24.7 at the start of the day. **2.96×.**
Gap to the B70 cookbook's 83.7 at MTP4: **1.145×**, from 4.3×.

## The ladder, and what each step actually bought

| step | t/s | gain | quality gate |
|---|---|---|---|
| baseline — vLLM 0.21.0-xpu, `--enforce-eager`, no spec | 24.7 | — | — |
| current image + `VLLM_XPU_ENABLE_XPU_GRAPH=1` | 27.9 | +13% | none needed |
| + native MTP4 speculative decoding | 48.4 | +73% | none needed |
| + **R3**: INT4 the *shared* `lm_head` | 66.2 | +36.8% | **PPL +0.840%, GSM8K χ²=0.250** |
| + **M1**: INT4 the MTP draft layer's linears | **73.1** | +10.4% | none needed — draft-only |

Two of the four gains are free by construction: graph capture changes no numerics, and
draft-side quantization is verified by the target. Only R3 touches the emitted
distribution, and it is the only one that carries gates.

## What is left, sized honestly

Per-token byte budget at MTP4 (the thing that sets the rate):

| | GB/step | share |
|---|---|---|
| verify: body | 12.703 | 74% |
| verify: `lm_head` INT4 | 0.656 | 4% |
| 4 draft passes: MTP layer INT4 + `lm_head` INT4 | 3.500 | 20% |
| GDN state + KV | 0.330 | 2% |
| **total** | **17.19** | |

- **The body is 74%** and R2 (sub-4-bit) is blocked on method — plain PQ gives 31% output
  error at 2 bits. Needs AQLM/QuIP#-class work that must beat GPTQ.
- **The draft passes are 20%** and are *quality-free ground*: the target verifies every
  proposed token, so approximation there costs acceptance, not correctness.
- Kernel work is closed: the GEMV runs at 93% of the 587 GB/s measured wall and is 75% of
  XPU time.

## What we cannot recover

**~1.18× is hardware.** The B70 runs a 230 W cap and reports 3400 MHz under load; the B65
is firmware-locked at 200 W and 2400 MHz, drawing 177 W with every throttle flag at zero.
Writing 230 W is accepted and reads back 200. Same image, same kernels — this part of the
gap is silicon and power.

## Five things measurement overturned

1. **R1 refuted** — `int4_gemm_w4a8` already ships and is 1.03× faster. The dequantize step
   was never the constraint.
2. **P2 closed** — the int4 GEMV already reaches 93% of the memory wall.
3. **R2 blocked** — plain PQ at 2 bits/weight gives 31% output error.
4. **R3 step 2 negative** — a cluster shortlist reads 26% fewer bytes for ~20× worse KL, and
   **top-1 agreement stays 95.7–98.8% the whole way**, which would have hidden it.
5. **385 GB/s was our own artifact** — eager-mode dispatch, not the card. It sustains 434.

## Two bugs found, neither ours

- **MTP emits garbage on prompts under ~8 tokens.** `"2 + 2 ="` → `'!!!!!!!!'`. Reproduced
  with the **stock fp16 head and no patches**. Longer prompts and the chat endpoint are
  fine, which is why a week of benchmarking never surfaced it.
- **`logprobs`/`prompt_logprobs` return NaN whenever MTP is on.** The rejection sampler is
  clean — instrumented at every stage — so it enters downstream in output processing.

**The lesson from the first one is the one worth keeping:** a throughput harness measures
tokens per second and does not care whether the tokens are words. Every number here was
collected without once checking the model still produced English. There is now a
greedy-equivalence check that compares against a no-speculation reference.

## Acceptance is workload-dependent to a factor of 1.8×

Same server, same config, only the prompt changed:

| workload | acceptance | t/s |
|---|---|---|
| repetitive filler | 100% | 78 |
| code continuation | 82% | 80 |
| step-by-step reasoning | 67% | 72 |
| dense real prose | 35% | 43 |

Any single tok/s figure for a speculative-decoding setup is a statement about the prompt as
much as the hardware. A benchmark built from a repeated sentence reports numbers no
deployment sees.

## Next target: 85 t/s with no quality cost

The draft passes are 20% of the byte budget and are quality-free by construction. That is
where the remaining headroom is.

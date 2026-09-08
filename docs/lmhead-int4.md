# R3 step 1 — INT4 the target `lm_head`

Started and measured 2026-09-07. Published as *B65 R3-01*.
Patch: [`../research/r3/patch_r3_lmhead_int4.py`](../research/r3/patch_r3_lmhead_int4.py).

## The opening

R3 was scheduled behind P0.5, but the draft-INT4 rung failing put it first — and the reason it
failed is the reason this works.

SergiioB's Phase S quantizes the **draft's** `lm_head` and keeps the target fp16, so
verification stays bit-exact. It relies on the draft owning a private copy
(`DraftModelProposer._maybe_share_lm_head` being a no-op). Our engine logs the opposite:

```
Detected MTP model. Sharing target model lm_head weights with the draft model.
```

**Their patch is blocked by the sharing. Ours is enabled by it.** One INT4 copy, cached by the
shared weight's storage pointer, serves the target verify pass *and* all four draft passes.

## What it does

`2.543 GB fp16 → 0.656 GB INT4 g128 sym — 3.88×, 1.887 GB saved per read.`

GPTQ INT4 g128 symmetric, in the exact layout `torch.ops._xpu_C.int4_gemm_w4a16` already
consumes; quantizer adapted from the cookbook's Phase S helper. Hooks `compute_logits` on both
`qwen3_5.py` (target) and `qwen3_5_mtp.py` (draft), lazily, env-gated on
`P608_R3_LMHEAD_INT4=1`, and falls through to the stock fp16 path on any failure or under TP>1.

## Result

| config | decode t/s | vs |
|---|---|---|
| baseline — old stack, eager, no spec | 24.7 | — |
| no-spec, current stack | 27.9 | +13% |
| **no-spec + R3** | **31.7** | **+13.5%** |
| MTP4 | 48.4 | +73% |
| **MTP4 + R3** | **66.2** | **+36.8%** |

**2.68× over the starting point.** Bar: B70 MTP4 = 83.7, +draft-INT4 = 106.7.

**The asymmetry is the proof of mechanism.** R3 is worth +13.5% with no speculation, where the
head is read once per token, and +36.8% under MTP4, where it is read five times per step. The
gain tracks the read count, exactly as a bytes argument predicts. This is also the roadmap
correction: R3 was costed assuming one read per token, and under speculation it is worth
roughly five times that.

## Quality — BOTH GATES PASSED, 2026-09-07

Published as *B65 R3-02*. R3 is now the default in the daily driver.

### Gate 1 — perplexity, gate < 1%

| | |
|---|---|
| corpus | 13,027 tokens, frozen, identical both runs |
| fp16 `lm_head` | **6.09315** |
| INT4 `lm_head` | **6.14434** |
| **delta** | **+0.840% — PASS** |

Computed from `prompt_logprobs`: the corpus is tokenized by the server's own `/tokenize`, then
the logprob of the *actual* next token is read at every position. Teacher forcing is inherent,
so both heads score identical inputs. Corpus is deliberately mixed and frozen in
[`../research/r3/ppl-corpus.json`](../research/r3/ppl-corpus.json) — every BetterBench prompt,
cookbook technical prose, real Python source, and the cookbook's JSON catalog.

**Where the loss lands.** Large-n buckets bracket the aggregate:

| source | n | fp16 → INT4 | Δ |
|---|---|---|---|
| prose/cookbook | 5499 | 9.3625 → 9.4579 | **+1.019%** |
| code/python | 3401 | 3.8173 → 3.8503 | +0.866% |
| json/catalog | 2440 | 3.4942 → 3.5019 | +0.219% |

Prose alone is **marginally over the 1% line**. Stated rather than hidden behind the aggregate.
Small-n categories swing more (chat +3.4% on 209 tokens) but that is noise at those sizes.

### Gate 2 — GSM8K, paired, n=100

| | fp16 | INT4 |
|---|---|---|
| accuracy | 88.0% | **90.0%** |
| 95% CI | [80.2, 93.0] | [82.6, 94.5] |
| truncated | 14 | 11 |

**The +2.0 pp is not an improvement and must not be reported as one.** Of 100 paired problems,
**96 got the same verdict**. Of the four that changed: 1 right→wrong, 3 wrong→right. McNemar
with continuity correction gives **χ² = 0.250** against a 3.841 threshold — **not significant**.

The honest claim: INT4 quantization of `lm_head` does not measurably change reasoning accuracy
at n=100. Detecting an effect smaller than roughly 8 points would need a larger subset.

Speculation was left **on** for GSM8K — MTP verification preserves the target distribution, so
it accelerates without changing which head is measured. It had to be **off** for perplexity,
because of the NaN bug below.

### Shipped

`serve-b65.sh` now defaults `R3=1`. Daily driver measured at **58.7 t/s wall-clock** on a
256-token completion against **41.1** with the fp16 head — same prompt, same server.

### A retraction

An earlier pass compared **free-running** generations position-by-position and reported 67.93%
top-1 agreement. **That number was wrong.** Once greedy output diverges the two runs score
different contexts, so it measures drift, not the head. Teacher forcing moved it to 94.31%, and
the perplexity/GSM8K gates above supersede it entirely.

## Bug found on the way

**`logprobs` and `prompt_logprobs` return NaN whenever MTP speculative decoding is enabled**,
so the request fails with `Out of range float values are not JSON compliant: nan`. Reproduced
with the **stock fp16 head**, so it is not caused by this patch. Confirmed absent with
`--speculative-config` removed.

Consequences: no distributional quality work is possible under MTP on this stack, which is why
the gate above was run with speculation off. This is upstream vLLM XPU material — **P5**.

## Next in R3

Step 1 took the head to 4 bits. The roadmap's full R3 — a two-stage hierarchical projection over
~2,048 vocabulary clusters, reading only selected rows — targets **2.543 → ~0.2 GB**, another
3× beyond this. Step 1's quality result argues for it: a shortlist head scored on KL against the
full head can be made *more* faithful than blanket 4-bit quantization, because it spends its bits
where the distribution actually has mass.

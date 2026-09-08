# B65 — making a 27B model read fewer bytes

**24.7 → 77.2 tok/s** on an Intel Arc Pro B65 32GB running Qwen3.8-27B GPTQ-Int4.
**3.13×**, without writing a single fast kernel.

Every gain here comes from the same idea: at batch 1 a language model is **not compute
bound, it is read bound**. Producing one token requires reading every weight, then doing
about **2 operations per byte read**, against a card that can do ~580. The matrix units
sit ~99.7% idle. So making the model *read less* is the only thing that helps.

| step | tok/s | what changed |
|---|---|---|
| baseline | 24.7 | vLLM 0.21.0-xpu, `--enforce-eager`, no speculation |
| current stack | 27.9 | current image + `VLLM_XPU_ENABLE_XPU_GRAPH=1` |
| + MTP4 | 48.4 | native speculative decoding |
| + **INT4 `lm_head`** | 66.2 | the vocabulary table, 2.543 → 0.656 GB |
| + **INT4 MTP linears** | 73.1 | the draft model's own weights, 0.849 → 0.219 GB |
| + **draft vocab prefix** | **77.2** | draft reads 32,768 of 248,320 rows |

Measured with [BetterBench](https://github.com/GGZ14/BetterBench), decode-only, greedy.
Raw runs in [`results/`](results/).

**Never seen this before?** There's a
[visual walkthrough of one decode step](https://claude.ai/code/artifact/3198c44b-1f6d-48bc-8831-a2be8657f47d)
— press play and watch both machines make the same trip to memory, and see how many words
each one comes back with.

## Quick start

```sh
MODEL=/path/to/your/model ./run.sh
```

Or three lines into an existing vLLM XPU container:

```
  -v /path/to/this/repo:/p608:ro \
  -e P608_R3_LMHEAD_INT4=1 -e P608_DRAFT_VOCAB=32768 \
  --entrypoint bash <image> -lc 'python /p608/patches/p608_lmhead_int4.py; exec vllm serve ...'
```

**Does it apply to you?** Look for this in your server log:

```
Detected MTP model. Sharing target model lm_head weights with the draft model.
```

If you see it, yes. If your build gives the draft its own head, use
[SergiioB's Phase S](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
instead — and note the two are mutually exclusive.

## The two ideas

**1 · Under speculative decoding the LM head is read 5× per step, not once.**

MTP4 runs 4 draft passes plus 1 verify, and every one reads the vocabulary table. At
2.543 GB that is 12.7 GB per step, ~98% of DRAM peak. It is not 16% of your traffic — it
is closer to half. Quantizing it to INT4: **+36.8%**.

This one touches the head the target *verifies* with, so it is gated:

| gate | fp16 | INT4 |
|---|---|---|
| perplexity, 13,027 held-out tokens | 6.09315 | 6.14434 (**+0.840%**) |
| GSM8K, 100 **paired** problems | 88.0% | 90.0% (McNemar χ²=0.250, **not significant**) |

The GSM8K number is **not** an improvement — 96/100 identical, 1 worse, 3 better.

**2 · The draft's vocabulary is a free parameter.**

The draft proposes tokens; the target verifies every one. So an approximation in the
draft costs **acceptance**, never **correctness**. Measured on real hidden states,
**95.9% of true argmaxes fall in the first 32,768 of 248,320 rows** — and because
`qweight` is `[K/8, V]` with stride `(1, K/8)`, those rows are a **contiguous memory
prefix**. A slice, not a gather. **7.6× cheaper draft head, +6%.**

Verified: 7/8 completions byte-identical to a no-speculation reference — the same 7/8 as
before the change.

## Verify it yourself

```sh
# speculation must be OFF: prompt_logprobs returns NaN under MTP on vLLM XPU
python verify/perplexity.py   --endpoint http://localhost:8000/v1 --model <id> --out ppl.json
python verify/gsm8k.py        --endpoint http://localhost:8000/v1 --model <id> --out gsm.json
python verify/greedy_equiv.py --endpoint http://localhost:8000/v1 --out ref.json   # no-spec
python verify/greedy_equiv.py --endpoint http://localhost:8000/v1 --ref ref.json   # candidate
```

`verify/ppl-corpus.json` is frozen so two runs are scored on identical text.

## Two bugs you should know about

**MTP emits garbage on prompts under ~8 tokens.** `"2 + 2 ="` → `'!!!!!!!!'`. Reproduced
with the **stock fp16 head and no patches** — not from anything here. Chat and
normal-length prompts are fine. A throughput benchmark counts tokens per second and does
not check they are words; that is why `verify/greedy_equiv.py` exists.

**`logprobs`/`prompt_logprobs` return NaN whenever MTP is enabled.** The rejection sampler
is clean — instrumented at every stage — so it enters downstream in output processing.
Details and a repro in [`docs/upstream-nan-bug.md`](docs/upstream-nan-bug.md).

## Measurements, including the negative ones

| | |
|---|---|
| [`bench/memory_wall.py`](bench/memory_wall.py) | STREAM-style. This card: **587 GB/s** read, 97% of datasheet |
| [`bench/int4_gemv.py`](bench/int4_gemv.py) | int4 GEMV at M=1. **547 GB/s amortised = 93% of the wall** — no kernel work left |
| [`bench/acceptance.py`](bench/acceptance.py) | acceptance per workload. Spans **35%→100%**, a 1.8× throughput swing |
| [`bench/vocab_coverage.py`](bench/vocab_coverage.py) | how much vocabulary the draft actually needs |
| [`bench/pq_feasibility.py`](bench/pq_feasibility.py) | product quantization at 2 bits: **31% output error**. Blocked |
| [`bench/shortlist_feasibility.py`](bench/shortlist_feasibility.py) | cluster shortlist head: 26% fewer bytes, **~20× worse KL** |

Four of five planned "clever ideas" were refuted by measurement, each in about twenty
minutes. `bench/` is mostly a record of things that did not work, which is the useful half.

**Speculation depth is not a fixed constant.** k=4 was optimal until the draft got 2.9×
cheaper; then k=6 won (77.2 vs 76.4), and k=8 regresses to 72.1. Published depth guidance
is only valid at the draft cost it was measured at.

## What is still blocked

At the shipped configuration the per-step budget is 15.5 GB, of which the **language body
is 12.703 GB — 81.8%**. Getting past ~85 tok/s on a mixed corpus needs that number to
fall, which means AQLM/QuIP#-class quantization that has to beat a strong GPTQ baseline.
Plain product quantization does not get close.

On code-like traffic this configuration already measures **111.7 tok/s**.

## Credit

The groundwork is **[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)** —
the pinned image, the MTP patches, the checkpoint (they quantized it), the benchmark
discipline, and the 5-reads-per-step insight, which comes from their Phase S patch header.
They remain ahead on the standard benchmark. Their `data/benchmarks.v1.json` is the numeric
authority, not this README.

Their patches are **not redistributed here** — get them from that repo.

Numbers are self-reported. MIT licensed.

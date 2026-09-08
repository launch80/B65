# B65 — making a 27B model read fewer bytes

**24.7 → 77.2 tok/s** on an Intel Arc Pro B65 32GB running Qwen3.8-27B GPTQ-Int4.
**3.13×**, without writing a single fast kernel.

Every gain comes from one idea: at batch 1 a language model is **not compute bound, it
is read bound**. Producing one token requires reading every weight, then doing about
**2 operations per byte read**, against a card that can do ~580. The matrix units sit
~99.7% idle. Making the model *read less* is the only thing that helps.

| step | tok/s | what changed |
|---|---|---|
| baseline | 24.7 | vLLM 0.21.0-xpu, `--enforce-eager`, no speculation |
| current stack | 27.9 | current image + `VLLM_XPU_ENABLE_XPU_GRAPH=1` |
| + MTP speculation | 48.4 | native multi-token prediction |
| + **INT4 `lm_head`** | 66.2 | vocabulary table, 2.543 → 0.656 GB |
| + **INT4 MTP linears** | 73.1 | draft model's own weights, 0.849 → 0.219 GB |
| + **draft vocab prefix** | **77.2** | draft reads 32,768 of 248,320 rows |

Measured with [BetterBench](https://github.com/GGZ14/BetterBench), decode-only, greedy.
Raw runs in [`results/`](results/).

**Never seen this before?** There's a
[visual walkthrough of one decode step](https://claude.ai/code/artifact/3198c44b-1f6d-48bc-8831-a2be8657f47d)
— press play and watch both machines make the same trip to memory.

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
instead — the two are mutually exclusive.

## The two ideas

**1 · Under speculative decoding the LM head is read 5× per step, not once.**

MTP4 runs 4 draft passes plus 1 verify, and every one reads the vocabulary table. At
2.543 GB that is 12.7 GB per step, ~98% of DRAM peak — not 16% of your traffic but
closer to half. Quantizing it to INT4: **+36.8%**.

This touches the head the target *verifies* with, so it is gated:

| gate | fp16 | INT4 |
|---|---|---|
| perplexity, 13,027 held-out tokens | 6.09315 | 6.14434 (**+0.840%**) |
| GSM8K, 100 **paired** problems | 88.0% | 90.0% (McNemar χ²=0.250, **not significant**) |

The GSM8K figure is **not** an improvement — 96/100 identical, 1 worse, 3 better.

**2 · The draft's vocabulary is a free parameter.**

The draft proposes; the target verifies every token. So approximation in the draft costs
**acceptance**, never **correctness**. On real hidden states **95.9% of true argmaxes
fall in the first 32,768 of 248,320 rows** — and since `qweight` is `[K/8, V]` with
stride `(1, K/8)`, those rows are a **contiguous memory prefix**. A slice, not a gather.
**7.6× cheaper draft head, +6%.** Verified 7/8 completions byte-identical to a
no-speculation reference.

## Reading the numbers honestly

**77.2 is decode-only, short prompts, a fixed corpus.** Through a coding agent at 21k
context the same server reports **47.6 end-to-end**. Nothing is wrong — different
measurements:

| | |
|---|---|
| benchmarks start the clock at the first token | a request also pays prompt processing — 2.7 s here, a third of a 400-token reply |
| benchmarks use short prompts | 100 t/s at 500 tokens of context, 76.9 at 12k, **69.8 at 21k**, 65.2 at 28k |
| acceptance is workload-dependent | 100% on repetitive text, 35% on dense prose — a 1.8× swing |

**Measuring trap:** under speculative decoding one streaming chunk carries a *burst* of
tokens. Counting chunks undercounts by exactly the acceptance factor — it briefly looked
like a 3× regression here. Use `stream_options: {"include_usage": true}`.
`bench/context_sweep.py` does it correctly.

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
with the **stock fp16 head and no patches**. Chat and normal-length prompts are fine. A
throughput benchmark counts tokens per second and does not check they are words; that is
why `verify/greedy_equiv.py` exists.

**`logprobs`/`prompt_logprobs` return NaN whenever MTP is enabled.** The rejection sampler
is clean — instrumented at every stage — so it enters downstream in output processing.
See [`docs/upstream-nan-bug.md`](docs/upstream-nan-bug.md).

## Measurements, including the negative ones

| | |
|---|---|
| [`bench/memory_wall.py`](bench/memory_wall.py) | STREAM-style. This card: **587 GB/s** read, 97% of datasheet |
| [`bench/int4_gemv.py`](bench/int4_gemv.py) | int4 GEMV at M=1. **547 GB/s amortised = 93% of the wall** |
| [`bench/acceptance.py`](bench/acceptance.py) | acceptance per workload. **35%→100%**, a 1.8× swing |
| [`bench/context_sweep.py`](bench/context_sweep.py) | throughput vs conversation length |
| [`bench/vocab_coverage.py`](bench/vocab_coverage.py) | how much vocabulary the draft actually needs |
| [`bench/pq_feasibility.py`](bench/pq_feasibility.py) | product quantization at 2 bits: **31% output error** |
| [`bench/aqlm_feasibility.py`](bench/aqlm_feasibility.py) | AQLM's ingredients — additive codebooks, activation weighting — **both fail** |
| [`bench/gptq_compensation.py`](bench/gptq_compensation.py) | error compensation: **22.34% → 5.13% at 3 bits** |
| [`bench/shortlist_feasibility.py`](bench/shortlist_feasibility.py) | cluster shortlist head: **~20× worse KL** |
| [`kernel/`](kernel/) | can we write our own int4 GEMV? **No — 54% of vendor** |

Most of `bench/` is a record of things that did not work, which is the useful half.
**Speculation depth is not a constant:** k=4 was optimal until the draft got 2.9× cheaper;
then k=6 won (77.2 vs 76.4), and k=8 regresses to 72.1.

## What is blocked, and why

The language body is **12.703 GB — 82% of every decode step**. Quantizing it to 3 bits
with error compensation costs 5.13% output error and would project ~96 tok/s. But nothing
on XPU consumes 3-bit, and packing 3-bit into the 4-bit container reads 4.125 bits/weight
and gains nothing — so it needs a purpose-built kernel.

We measured whether we could write one. Best of four implementations: **212 GB/s against
the vendor's 394**. A 3-bit kernel needs **298 just to break even**. See
[`kernel/README.md`](kernel/README.md). The productive move is to ask upstream for a
w3a16 kernel rather than write one, and the evidence for that request is in `bench/`.

## Credit

The groundwork is **[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)** —
the pinned image, the MTP patches, the checkpoint (they quantized it), the benchmark
discipline, and the 5-reads-per-step insight from their Phase S patch header. They remain
ahead on the standard benchmark. Their `data/benchmarks.v1.json` is the numeric authority,
not this README. **Their patches are not redistributed here** — get them from that repo.

Numbers are self-reported. MIT licensed.

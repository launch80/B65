# B65 — making a 27B model read fewer bytes

**24.7 → 77.2 tok/s** on an Intel Arc Pro B65 32GB running Qwen3.8-27B GPTQ-Int4.
**3.13×**, and nothing we wrote runs in the serving path — it is Intel's int4 kernel the
whole way.

Every gain comes from one idea: at batch 1 a language model is **not compute bound, it
is read bound**. Producing one token requires reading every weight, then doing about
**2 operations per byte read**, against a card that can do ~580. The matrix units sit
~99.7% idle. Making the model *read less* is the only thing that helps.

| step | tok/s | what changed | whose |
|---|---|---|---|
| baseline | 24.7 | vLLM 0.21.0-xpu, `--enforce-eager`, no speculation | — |
| current image | 27.9 | vLLM 0.27.2 with inductor compile on. *Not* graph capture — see below | upstream |
| + MTP speculation | 48.4 | native multi-token prediction, draft head ships in the checkpoint | cookbook |
| + **INT4 `lm_head`** | 66.2 | vocabulary table, 2.543 → 0.656 GB, read 5× per step | **ours** |
| + INT4 MTP linears | 73.1 | draft model's own weights, 0.849 → 0.219 GB | cookbook (Phase M1) |
| + **draft vocab prefix** | **77.2** | draft reads 32,768 of 248,320 rows, k=6 | **ours** |
| + capture off | ~79 | `VLLM_XPU_ENABLE_XPU_GRAPH=0`, +2% on a fixed workload, repeated | ours |

Measured with [BetterBench](https://github.com/GGZ14/BetterBench), decode-only, greedy.
Raw runs in [`results/`](results/). The last row is measured on the fixed workload in
[`bench/draft_accept.py`](bench/draft_accept.py), not yet on BetterBench, so it carries a
tilde.

**Never seen this before?** There's a
[visual walkthrough of one decode step](https://claude.ai/code/artifact/3198c44b-1f6d-48bc-8831-a2be8657f47d)
— press play and watch both machines make the same trip to memory.

## What is happening right now (2026-09-08)

**We are publishing a baked checkpoint.** Today the INT4 `lm_head` and the INT4 MTP draft
layer exist only *in VRAM*: a patch requantizes them round-to-nearest at boot. That works,
but nobody can reproduce it without the patch stack. The bake produces the same tensors
**on disk**, in the exact GPTQ INT4 g128 symmetric layout the body already uses, with
Hessian error compensation instead of round-to-nearest — so **stock vLLM loads them with no
weight patches at all.** What is in VRAM will be what is on disk.

| stage | status |
|---|---|
| GPTQ bake of `lm_head` + the 8 draft-layer tensors, calibrated on wikitext-2 | **running now** |
| assemble the checkpoint: rebuild shards, swap the 9 dense tensors, set `lm_head: true` and drop the `mtp.*` exclusion in `quantize_config.json` | next |
| gate 1 — greedy equivalence against the patched stock checkpoint | pending |
| gate 2 — perplexity, same 13,027 frozen tokens | pending |
| gate 3 — GSM8K, 100 paired problems, McNemar | pending |
| gate 4 — BetterBench decode at MTP k=6, must match 77.2 | pending |
| publish to Hugging Face, add the bake tooling and gate results here | after the gates |

The 400 body linears stay byte-identical to SergiioB's. The draft vocab prefix is a
runtime slice, not a weight, so it stays a patch. Three *correctness* patches from the
cookbook are still needed on vLLM 0.27.2-xpu (MTP nightly, MTP boundary, GDN mixed-split);
none of them touch weights.

## Quick start

```sh
MODEL=/path/to/your/model ./run.sh
```

Or three lines into an existing vLLM XPU container:

```
  -v /path/to/this/repo:/p608:ro \
  -e P608_R3_LMHEAD_INT4=1 -e P608_DRAFT_VOCAB=32768 -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  --entrypoint bash <image> -lc 'python /p608/patches/p608_lmhead_int4.py; exec vllm serve ...'
```

[`serving/serve-b65.sh`](serving/serve-b65.sh) is the daily driver on our box, with every
knob documented inline and the current defaults: k=6, capture off, GDN mixed-split on.

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

## What a draft pass actually costs

The byte model said a draft pass moves 0.306 GB, about 0.76 ms at this card's rate, and
predicted k=8 would beat k=6. Measured, k=8 is ~10% slower. So we measured step time
directly across k on one fixed workload ([`bench/draft_ksweep.sh`](bench/draft_ksweep.sh)):

```
step_ms = 32.5 + 2.53 * k        target pass 32.5 ms, each draft pass 2.53 ms
```

**A draft pass costs 3.3× its bytes.** One draft layer takes five times what a target
layer takes, because per-forward fixed cost — attention-metadata rebuild, launch chain,
argmax, host syncs in the proposer — is paid once by the target and k times by the draft.
At k=6 that is ~26% of the step and none of it is bytes. Details in
[`docs/draft-time-model.md`](docs/draft-time-model.md).

Three things follow, all measured:

- **k=6 is right** for this draft. k=4 only wins on low-acceptance workloads.
- **The current draft config is optimal.** Restoring the draft to full precision raises
  acceptance (36.5 → 41.4%) but the extra bytes cost more than the acceptance buys, in all
  four arms of the 2×2 ([`bench/draft_m1r6_sweep.sh`](bench/draft_m1r6_sweep.sh)). Bytes
  are additive to the fixed cost, not hidden under it.
- **XPU graph capture is worth 0.0%** on this image: target pass 32.26 ms on, 32.26 ms
  off ([`bench/draft_graphtest.sh`](bench/draft_graphtest.sh)). The +13% once credited to
  it was measured on vLLM 0.21 with `--enforce-eager`, where the compiler was off. Capture
  off is +2% end to end because the draft slope improves ~1%.

The remaining lever toward 85 tok/s is that 2.53 ms fixed cost per draft forward. That is
vLLM proposer internals, not configuration.

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

`verify/ppl-corpus.json` is frozen so two runs are scored on identical text. These are the
same four gates the baked checkpoint has to pass.

## Three bugs you should know about

**The engine dies under concurrent load with MTP on.** The fused `gdn_attention` binding
refuses a batch that mixes a prefilling request with speculating ones:

```
causal_conv1d does not support spec-decode and non-spec (prefill + decode) tokens in the same invocation
```

It is a hard `TORCH_CHECK`, it kills EngineCore, and every request after it gets a
connection error. It never fires single-stream, so no benchmark sees it; any agent harness
hits it within minutes. The fix is the cookbook's `patch_gdn_mixed_split_v5.py`, which
`serve-b65.sh` now applies whenever speculation is on. Every throughput number in this
repo was measured single-stream, so none is invalidated — but the benchmarked config and
the used config were not the same config until 2026-09-08.

**MTP emits garbage on prompts under ~8 tokens.** `"2 + 2 ="` → `'!!!!!!!!'`. Reproduced
with the **stock fp16 head and no patches**. Chat and normal-length prompts are fine. A
throughput benchmark counts tokens per second and does not check they are words; that is
why `verify/greedy_equiv.py` exists.

**`logprobs`/`prompt_logprobs` return NaN whenever MTP is enabled.** The rejection sampler
is clean — instrumented at every stage — so it enters downstream in output processing.
See [`docs/upstream-nan-bug.md`](docs/upstream-nan-bug.md).

## The kernel we built and cannot use

Nobody ships a fast 3-bit weight-only GEMV for Intel GPUs: Marlin and Machete reject
`bits != 4,8`, BitBLAS has no INT3 row, Intel's ARK lists INT4/INT8 only. So we wrote one.

**[`kernel/w3_l80.cpp`](kernel/) is 1.17× faster than Intel's `int4_gemm_w4a16`** across
all 400 body linears, at `rel err 6.4e-04`. 32 weights in exactly 3 int32 words, zero
waste, shaped so dequant is one masked `half2` reinterpret. Column-major beat k-major by
32%, `grf_size<256>` cost 15%, and column blocking that had "collapsed" turned out to be
the single largest win once the register pressure was understood. Three of four design
decisions went against prediction and were settled by measurement.
[`kernel/README.md`](kernel/README.md) has the format and the numbers.

**It does not ship for this model.** The 5.13% output error that motivated it was scored
*in-sample* — on the same 1,024 activation rows the Hessian was fitted to. Held out, the
same weights and code give **~21%**, 32× more calibration data does not move it, and
group 64 and 32 recover only 20% and 18%. The error is the 8-level grid, not the scale
granularity. Gate was ≤6%. The quantizer that established this is in
[`quantize/`](quantize/), with a held-out set, a verified dequant path, and calibration
text disjoint from the gates. See [`docs/w3-quantization-verdict.md`](docs/w3-quantization-verdict.md).

**And the 4-bit path is nearly at the wall.** Measured with incompressible data
([`kernel/ceiling.cpp`](kernel/ceiling.cpp)): this card reads at **579 GB/s, 95% of the
608 datasheet**, and the GEMV access pattern reaches that rate. Intel's kernel sits
1.10× above the pure-read floor, so that is the cap on beating it; our 4-bit kernel gets
1.02×. The remainder is dequant ALU, and the
obvious route to it — dp4a via `dot_acc` — is a portability shim that made it *slower*.
[`docs/bandwidth-ceiling.md`](docs/bandwidth-ceiling.md).

## Measurements, including the negative ones

| | |
|---|---|
| [`bench/memory_wall.py`](bench/memory_wall.py) | STREAM-style, torch-XPU: 587 GB/s read. Re-measured in SYCL with incompressible data: **579** |
| [`bench/int4_gemv.py`](bench/int4_gemv.py) | int4 GEMV at M=1, amortised over back-to-back calls |
| [`bench/draft_accept.py`](bench/draft_accept.py) | acceptance + tokens/step + observed tok/s on a fixed workload |
| [`bench/draft_ksweep.sh`](bench/draft_ksweep.sh) | k = 0…8 → `step_ms = 32.5 + 2.53k` |
| [`bench/draft_graphtest.sh`](bench/draft_graphtest.sh) | graph capture on/off: **0.0%** on the target pass |
| [`bench/draft_m1r6_sweep.sh`](bench/draft_m1r6_sweep.sh) | INT4 draft × vocab prefix, 2×2 at k=6: current config wins |
| [`bench/acceptance.py`](bench/acceptance.py) | acceptance per workload. **35%→100%**, a 1.8× swing |
| [`bench/context_sweep.py`](bench/context_sweep.py) | throughput vs conversation length |
| [`bench/vocab_coverage.py`](bench/vocab_coverage.py) | how much vocabulary the draft actually needs |
| [`bench/pq_feasibility.py`](bench/pq_feasibility.py) | product quantization at 2 bits: **31% output error** |
| [`bench/aqlm_feasibility.py`](bench/aqlm_feasibility.py) | AQLM's ingredients — additive codebooks, activation weighting — **both fail** |
| [`bench/gptq_compensation.py`](bench/gptq_compensation.py) | error compensation at 3 bits: 22.34% → 5.13% **in-sample**; ~21% held out |
| [`bench/shortlist_feasibility.py`](bench/shortlist_feasibility.py) | cluster shortlist head: **~20× worse KL** |
| [`kernel/`](kernel/) | W3A16 GEMV **1.17× vendor**; W4A16 1.02×; W4A8 via `dot_acc` 0.82× |
| [`quantize/`](quantize/) | layer-chunked GPTQ to 3 bits, with the held-out scoring that killed it |

Most of `bench/` is a record of things that did not work, which is the useful half.

## Things we were wrong about, and what killed each

- **"Graph capture, +13%."** 0.0% on this image. The compiler was the gain.
- **"The int4 kernel is exhausted at 91.6% of the wall."** The wall was mis-stated. Real
  cap on beating the vendor at 4 bits is 1.10×; the last 5% is DRAM overhead.
- **"1073 GB/s read."** 177% of datasheet. Arc does lossless memory compression and the
  buffer was constant-filled. Read benchmarks on this hardware must use random data.
- **"3-bit costs 5% error."** In-sample. Really ~21%.
- **"The draft is 20% of the bytes; shrink it."** Stale by one optimisation (11.8%), and
  bytes are not what it costs — the fixed per-forward overhead is.
- **"Hand-written SYCL reaches 54% of the vendor."** Measured per-call rather than in a
  burst, and on the wrong GPU: this box has two Battlemage cards and `--device /dev/dri`
  with `ZE_AFFINITY_MASK=0` silently selects the B580. Pass the render node by PCI address.

Each was falsified by a measurement that took 15–30 minutes.

## Layout

| | |
|---|---|
| [`patches/`](patches/) | the INT4 `lm_head` + draft-vocab patch, env-gated, falls through to stock on failure |
| [`run.sh`](run.sh) · [`serving/`](serving/) | one-command server; the daily driver with every knob documented |
| [`verify/`](verify/) | the quality gates and the frozen corpus |
| [`bench/`](bench/) | every measurement, positive and negative |
| [`kernel/`](kernel/) | the W3A16 / W4A16 / W4A8 kernels, benches, and the ceiling measurement |
| [`quantize/`](quantize/) | the 3-bit GPTQ pipeline and its serving path |
| [`docs/`](docs/) | the reasoning behind each number; `current-state.md` is a 2026-09-07 snapshot |
| [`results/`](results/) | raw BetterBench and counter JSON for every number above |

## Credit

The groundwork is **[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)** —
the pinned image, the MTP patches, the checkpoint (they quantized it), the draft-INT4
rung, the crash fix, the benchmark discipline, and the 5-reads-per-step insight from their
Phase S patch header. They remain ahead on a B70 at 230 W; about 1.18× of the gap is their
230 W / 3400 MHz against this card's firmware-locked 200 W / 2400 MHz, which no software
recovers. Their `data/benchmarks.v1.json` is the numeric authority, not this README.
**Their patches are not redistributed here** — get them from that repo.

Numbers are self-reported. MIT licensed.

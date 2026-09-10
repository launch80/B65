# B65 — making a 27B model read fewer bytes

**24.7 → 71.8 tok/s** decode on an Intel Arc Pro B65 32GB running Qwen3.8-27B GPTQ-Int4,
in the configuration we actually ship. **2.9×**, and nothing we wrote runs in the serving
path — it is Intel's int4 kernel the whole way. The result is a
**[published checkpoint](https://huggingface.co/Launch80/Qwen3.8-27B-GPTQ-Int4-baked)** that
stock vLLM loads with **no weight patches**.

Every gain comes from one idea: at batch 1 a language model is **not compute bound, it is
read bound**. Producing one token means reading every weight and doing about 2 operations
per byte read, on a card that can do ~580. The matrix units sit ~99.7% idle. Making the
model *read less* is the only thing that helps.

| step | tok/s | what changed | whose |
|---|---|---|---|
| baseline | 24.7 | vLLM 0.21.0-xpu, `--enforce-eager`, no speculation | — |
| current image | 27.9 | vLLM 0.27.2 with inductor compile on. *Not* graph capture | upstream |
| + MTP speculation | 48.4 | native multi-token prediction; the draft head ships in the checkpoint | cookbook |
| + **INT4 `lm_head`** | 66.2 | vocabulary table, 2.543 → 0.656 GB, read 5× per step | **ours** |
| + INT4 MTP linears | 73.1 | the draft model's own weights, 0.849 → 0.219 GB | cookbook (Phase M1) |
| + **draft vocab prefix** | 77.2 | draft reads 32,768 of 248,320 rows, k=6 | **ours** |
| shipped config | **71.8** | + GDN mixed-split fix (mandatory under load), capture off | — |

All rows are [BetterBench](https://github.com/GGZ14/BetterBench), decode-only, greedy,
single stream; raw runs in [`results/`](results/). The ladder was climbed on 2026-09-07. On
2026-09-08 the shipped configuration — the same patches plus the concurrency crash fix
below — re-measured at 71.8 on the same harness, and the baked checkpoint matched it
exactly. Which part of that config accounts for the difference from 77.2 has not been
isolated; the like-for-like pair is the number that stands.

## Where things stand (2026-09-09)

**The baked checkpoint is published and is what we serve.** The INT4 `lm_head` and the INT4
MTP draft layer used to exist only in VRAM, requantized round-to-nearest at boot by a
patch. The bake produces the same tensors **on disk**, in the exact GPTQ INT4 g128
symmetric layout the body uses, with Hessian error compensation instead of
round-to-nearest. The 400 body linears are byte-identical to SergiioB's. 17.05 GB total.

| gate | baked | patched stock (RTN at boot) | fp16 head |
|---|---|---|---|
| perplexity, 13,027 frozen tokens, spec off | **6.133** | 6.144 | 6.093 |
| GSM8K n=100 @1536 tokens, paired | **91** | 89 | — |
| greedy, speculation on vs off, same weights | **8/8 identical** | — | — |
| BetterBench decode, MTP k=6, weighted, same day, same config | **71.8** | 71.8 | — |

Speed-neutral, quality-better, zero boot-time weight patches. Layer error against the RTN
patches roughly halves everywhere (lm_head 3.64% vs 7.58%). Details, including the R6 bug
the bake surfaced: [`docs/baked-checkpoint.md`](docs/baked-checkpoint.md). Tooling:
[`bake/`](bake/).

**In progress:** the same bake applied to
[MikeCaldera's fresh GPTQ quant](https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
of the same base model. It has the same layout, so every patch and the bake pipeline apply
unchanged. Early single-stream indications suggest its body accepts more draft tokens;
nothing goes in this table until it has been through BetterBench and the gates above.

## Quick start

Stock vLLM XPU 0.27.2, no weight patches:

```sh
vllm serve Launch80/Qwen3.8-27B-GPTQ-Int4-baked --quantization gptq --dtype float16 \
  --kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":6}'
```

Three *correctness* patches from the cookbook are still needed on that image (MTP nightly,
MTP boundary, GDN mixed-split) — none touches weights; get them from the cookbook repo.
The draft vocab prefix is a runtime slice, not a weight, so it stays a patch
([`patches/`](patches/), env-gated, falls through to stock on failure).

For a stock-format checkpoint with the patches applied at boot instead:

```sh
MODEL=/path/to/your/model ./run.sh
```

[`serving/serve-b65.sh`](serving/serve-b65.sh) and [`bake/serve-baked.sh`](bake/serve-baked.sh)
are the two launchers on our box, every knob documented inline. Current defaults: k=6,
capture off, GDN mixed-split on, `--max-num-seqs 32` (hybrid GDN models fail engine start
at vLLM's default 256).

**Does it apply to you?** Look for this in your server log:

```
Detected MTP model. Sharing target model lm_head weights with the draft model.
```

If you see it, yes. If your build gives the draft its own head, use
[SergiioB's Phase S](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
instead; the two are mutually exclusive.

## The two ideas

**1 · Under speculative decoding the LM head is read 5× per step, not once.** MTP4 runs 4
draft passes plus 1 verify and each reads the vocabulary table: 12.7 GB per step at fp16,
closer to half the traffic than the 16% a static byte count suggests. INT4: **+36.8%**.
This touches the head the target verifies with, so it is gated: perplexity +0.840%, GSM8K
paired 90 vs 88 (McNemar χ²=0.250, not significant; 96/100 identical answers).

**2 · The draft's vocabulary is a free parameter.** The draft proposes; the target verifies
every token, so approximation in the draft costs acceptance, never correctness. On real
hidden states **95.9% of true argmaxes fall in the first 32,768 of 248,320 rows**, and
because `qweight` is `[K/8, V]` with stride `(1, K/8)` those rows are a contiguous memory
prefix — a slice, not a gather. **7.6× cheaper draft head, +6%.**

## What a draft pass actually costs

The byte model predicted k=8 beats k=6. Measured, k=8 is ~10% slower. Step time across k on
one fixed workload ([`bench/draft_ksweep.sh`](bench/draft_ksweep.sh)):

```
step_ms = 32.5 + 2.53 * k        target pass 32.5 ms, each draft pass 2.53 ms
```

**A draft pass costs 3.3× its bytes**: attention-metadata rebuild, launch chain, argmax and
host syncs are paid once by the target and k times by the draft. Consequences, all
measured: k=6 is right; the INT4 draft plus vocab prefix beats every other arm of the 2×2
even though full precision raises acceptance 36.5 → 41.4%; and XPU graph capture is worth
**0.0%** on this image (the +13% once credited to it was the compiler, measured on 0.21
with `--enforce-eager`). [`docs/draft-time-model.md`](docs/draft-time-model.md).

## Reading the numbers honestly

**71.8 is decode-only, short prompts, a fixed corpus.** Through a coding agent at 21k
context the same server reports ~48 end-to-end. Nothing is wrong: a request also pays
prompt processing (2.7 s here, a third of a 400-token reply); throughput falls with context
(100 t/s at 500 tokens, 69.8 at 21k, 65.2 at 28k); and acceptance is workload-dependent
(100% on repetitive text, 35% on dense prose, a 1.8× swing). Spec-on tok/s is acceptance
times bandwidth, so it is never quoted here as a bandwidth figure.

**Measuring trap:** under speculative decoding one streaming chunk carries a burst of
tokens. Counting chunks undercounts by exactly the acceptance factor. Use
`stream_options: {"include_usage": true}`; [`bench/context_sweep.py`](bench/context_sweep.py)
does it correctly.

## Verify it yourself

```sh
# speculation must be OFF: prompt_logprobs returns NaN under MTP on vLLM XPU
python verify/perplexity.py   --endpoint http://localhost:8000/v1 --model <id> --out ppl.json
python verify/gsm8k.py        --endpoint http://localhost:8000/v1 --model <id> --out gsm.json
python verify/greedy_equiv.py --endpoint http://localhost:8000/v1 --out ref.json   # no-spec
python verify/greedy_equiv.py --endpoint http://localhost:8000/v1 --ref ref.json   # candidate
```

`verify/ppl-corpus.json` is frozen so two runs score identical text. These are the gates
the baked checkpoint passed.

## Three bugs you should know about

- **The engine dies under concurrent load with MTP on.** The fused `gdn_attention` binding
  refuses a batch mixing a prefilling request with speculating ones
  (`causal_conv1d does not support spec-decode and non-spec ... in the same invocation`), a
  hard `TORCH_CHECK` that kills EngineCore. Never fires single-stream; any agent harness
  hits it in minutes. Fix: the cookbook's `patch_gdn_mixed_split_v5.py`, on by default here.
- **MTP emits garbage on prompts under ~8 tokens.** `"2 + 2 ="` → `'!!!!!!!!'`, reproduced
  with the stock fp16 head and no patches. Throughput benchmarks count tokens, not words;
  that is why `verify/greedy_equiv.py` exists.
- **`logprobs` return NaN whenever MTP is enabled.** The rejection sampler is clean; it
  enters downstream in output processing. [`docs/upstream-nan-bug.md`](docs/upstream-nan-bug.md).

## The kernel we built and cannot use

Nobody ships a fast 3-bit weight-only GEMV for Intel GPUs, so we wrote one.
[`kernel/w3_l80.cpp`](kernel/) is **1.17× faster than Intel's `int4_gemm_w4a16`** across all
400 body linears at `rel err 6.4e-04`. It does not ship for this model: the 5.13% output
error that motivated it was scored in-sample; held out it is **~21%**, and neither 32× more
calibration data nor smaller groups move it. The error is the 8-level grid. And the 4-bit
path is nearly at the wall: this card reads **579 GB/s, 95% of datasheet**, with
incompressible data, and Intel's kernel sits 1.10× above the pure-read floor.
[`docs/w3-quantization-verdict.md`](docs/w3-quantization-verdict.md),
[`docs/bandwidth-ceiling.md`](docs/bandwidth-ceiling.md), [`kernel/README.md`](kernel/README.md).

## Things we were wrong about

- **"Graph capture, +13%."** 0.0% on this image. The compiler was the gain.
- **"1073 GB/s read."** Arc does lossless memory compression and the buffer was
  constant-filled. Read benchmarks on this hardware must use random data.
- **"3-bit costs 5% error."** In-sample. Really ~21%.
- **"The draft is 20% of the bytes; shrink it."** Bytes are not what a draft pass costs.
- **"Hand-written SYCL reaches 54% of the vendor."** Measured per-call, on the wrong GPU:
  `--device /dev/dri` with `ZE_AFFINITY_MASK=0` silently selects the B580 on a two-card
  box. Pass the render node by PCI address.
- **"The benchmarked config is the served config."** It was not, until the GDN fix went in
  on 2026-09-08. Every number above was measured single-stream, so none is invalidated.

Each was falsified by a measurement that took 15–30 minutes. Most of [`bench/`](bench/) is
a record of things that did not work, which is the useful half.

## Layout

| | |
|---|---|
| [`bake/`](bake/) | the GPTQ bake of `lm_head` + draft, assembly, gates, launcher, model card |
| [`patches/`](patches/) | the INT4 `lm_head` + draft-vocab boot patch, env-gated |
| [`run.sh`](run.sh) · [`serving/`](serving/) | one-command server; the daily-driver launcher with every knob documented |
| [`verify/`](verify/) | the quality gates and the frozen corpus |
| [`bench/`](bench/) | every measurement, positive and negative |
| [`kernel/`](kernel/) · [`quantize/`](quantize/) | the W3A16 / W4A16 / W4A8 kernels and the 3-bit pipeline that killed them |
| [`docs/`](docs/) | the reasoning behind each number |
| [`results/`](results/) | raw BetterBench and counter JSON for every number above |

## Credit

The groundwork is **[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)**:
the pinned image, the MTP patches, the checkpoint the bake is built on, the draft-INT4
rung, the crash fix, the benchmark discipline, and the 5-reads-per-step insight. They
remain ahead on a B70 at 230 W; about 1.18× of the gap is their 230 W / 3400 MHz against
this card's firmware-locked 200 W / 2400 MHz, which no software recovers. Their
`data/benchmarks.v1.json` is the numeric authority, not this README. **Their patches are
not redistributed here.**

Numbers are self-reported. MIT licensed.

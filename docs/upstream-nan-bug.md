# DRAFT — upstream issue for vllm-project/vllm

**Not filed.** Prepared 2026-09-07, awaiting the user's go-ahead before posting.

---

**Title:** `[Bug][XPU] logprobs and prompt_logprobs return NaN when MTP speculative decoding is enabled`

### Summary

On the XPU backend, enabling MTP speculative decoding makes every request that asks for
`logprobs` or `prompt_logprobs` fail. The values come back as `NaN`, and FastAPI's JSON
encoder then rejects the response:

```
{"error":{"message":"Out of range float values are not JSON compliant: nan",
          "type":"BadRequestError","param":null,"code":400}}
```

Text generation itself is unaffected and output is coherent — only the logprob path is broken.
Removing `--speculative-config` fixes it immediately, on the same image, same model, same
weights, same request.

### Environment

| | |
|---|---|
| GPU | Intel Arc Pro B65 32 GB (Battlemage BMG-G31, 20 Xe cores, 160 XMX) |
| Image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b.xpu` |
| kernels | `vllm-xpu-kernels 0.1.12.3` |
| torch | `2.13.0+xpu` |
| driver | Level Zero 1.15.37833+4 |
| model | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (`Qwen3_5MTP`), GPTQ INT4, fp8 KV |

### Reproduction

Serve with MTP speculative decoding:

```bash
vllm serve <model> --quantization gptq --dtype float16 \
  --max-model-len 32768 --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 \
  --max-num-seqs 1 --no-enable-prefix-caching --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

Then:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "<model>", "prompt": "The capital of France is",
  "max_tokens": 4, "temperature": 0, "logprobs": 20 }'
```

**Result:** HTTP 400, `Out of range float values are not JSON compliant: nan`.
Same for `"prompt_logprobs": 20`.

**Control:** drop `--speculative-config`, restart, re-issue the identical request:

```json
{"choices":[{"text":" Paris.\nThe","logprobs":{
  "token_logprobs":[-0.58048015832901,-0.37162813544273376,
                    -1.2526633739471436,-1.1680587530136108],
  "tokens":[" Paris",".","\n","The"], ...
```

Correct values, no NaN.

### Scope

- Reproduces with `logprobs` on `/v1/completions` and with `prompt_logprobs`.
- Reproduces with an **unmodified** model. We separately run a local patch that quantizes
  `lm_head`; the bug appears with that patch absent, so it is not related.
- `num_speculative_tokens` 4 tested. Not yet swept over other values.
- Not yet checked whether CUDA is affected — this may be XPU-specific or general to the MTP
  proposer path.

### Impact

Any evaluation that needs a distribution rather than a sampled token is impossible while
speculative decoding is on. That includes perplexity, KL-based comparisons, and calibration
work — precisely the measurements needed to validate a quantization change on a stack where
speculation is the main throughput win. Practical workaround is to disable speculation for
evaluation runs, which doubles the work and prevents measuring the served configuration.

### Localization — measured, not guessed

Instrumented `RejectionSampler._get_logprobs_tensors` to count NaN/Inf at every stage, on
every call. **The sampler path is clean.** On a failing request:

```
logits(in)          shape=(5, 248320) fp16  nan=0 inf=0
target_logits       shape=(4, 248320) fp32  nan=0 inf=0
bonus_logits        shape=(1, 248320) fp32  nan=0 inf=0
final_logits        shape=(5, 248320) fp32  nan=0 inf=0
rows=5 covered=5 UNCOVERED=0
accepted_logits     shape=(5, 248320) fp32  nan=0 inf=0
accepted_logprobs   shape=(5, 248320) fp32  nan=0 inf=0
OUT.logprobs        shape=(5, 6)      fp32  nan=0 inf=0
```

Left instrumented across a full request: **zero NaN reports**, every call, while the request
still fails with the serialization error. Every row of `final_logits` is covered by
`target_logits_indices` ∪ `bonus_logits_indices`, so the zero-fill is never surfaced.

**Conclusion: the NaN is introduced downstream of the rejection sampler**, somewhere between
`parse_output` / `LogprobsTensors.filter` and the OpenAI serialization layer. We did not chase
it further — see below.

> **Retracting an earlier guess.** A previous draft of this report speculated that rejected
> draft positions were left uninitialized in the logprob buffer and surfaced as garbage. **The
> instrumentation above disproves that.** The buffer is fully written and clean. Anyone picking
> this up should start at the output processor, not the proposer.

### Why this is not blocking us (and why we stopped here)

The measurement this bug prevents — distributional comparison of a quantized `lm_head` — turns
out not to need it:

- head quality is a property of the head, so perplexity and KL are measured with
  `--speculative-config` omitted, which *isolates* the variable rather than confounding it;
- the effect on speculation is read from `vllm:spec_decode_*` Prometheus counters, which are
  unaffected.

So this is filed as a correctness report for others, not as a blocker for us. The localization
above is the useful part.

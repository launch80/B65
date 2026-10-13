# DFlash2 drafter on one Arc Pro B65: 80.5 → 96.7 tok/s on the same checkpoint

Target: [Launch80/Swift-1.5-Qwen3.8-27B-Uncensored-Int4-baked](https://huggingface.co/Launch80/Swift-1.5-Qwen3.8-27B-Uncensored-Int4-baked)
(GPTQ INT4 body, `lm_head` + MTP draft baked on disk). Drafter:
[Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter](https://huggingface.co/Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter)
(z-lab's DFlash 2 block-diffusion drafter with its five layers' `o_proj`/`gate`/`up`/`down` and `fc` GPTQ INT4).
vLLM 0.30.0 XPU has `method: "dflash"` built in. All numbers: one Arc Pro B65 32 GB, fp8 KV, single stream.

## Result (BetterBench 0.4.0 quick run, category-weighted median decode tok/s)

| configuration | weighted | tokens/step | step ms | context |
|---|--:|--:|--:|--:|
| built-in MTP k=6 | 80.5 | 3.35 | 42.0 | 262k |
| DFlash2 bf16 drafter, k=7 (8-prompt probe) | 84.6 | 4.29 | 50.0 | 32k probe |
| **DFlash2 INT4 drafter, k=7** | **96.7** | 4.23 | 44.5 | 128k |
| reference: Wrapzii Swift 1.5 bake (not abliterated), MTP k=6 | 88.6 | 3.68 | 42.1 | 262k |

Gates on the DFlash2 arm: GSM8K 94/100 (same as MTP), 0/50 refusals on XSTest-safe prompts, greedy spec-on == spec-off 16/16.
Page: https://launch80.com/a/602f86d7-b0d3-4d2b-9c94-1037fe1d3cd3 (free-tier pages expire; the JSON in `results/` is the record).

## What mattered, in order

1. **The drafter must run in bf16.** vLLM copies the target dtype onto the draft; on the fp16 XPU stack the bf16-trained
   drafter then accepts 0 of 2,681 drafts (the target's massive residual activations overflow its input projection).
   `patches/patch_dflash_mixed_dtype.py` keeps the target fp16 and the drafter bf16 (`P608_DRAFT_DTYPE=bfloat16`): draft
   ModelConfig dtype, proposer buffers, and casts where target-dtype tensors enter the drafter (shared embedding, `fc`,
   context-KV projection) and where its hidden states hit the shared GPTQ `lm_head`. Running everything in bf16 also works
   (no patch) at about −1 ms of step.
2. **Acceptance is not the lever, bytes are.** DFlash2 lifts acceptance 3.4 → 4.2 tokens/step, but the bf16 drafter's 3.2 GB
   of layer weights add ~8 ms to a 41 ms verify step (B65 bandwidth). Net +4%. Quantizing the drafter's linears to INT4
   (`dflash/bake_hessians_drafter.py` + `dflash/bake_drafter_int4.py`, Hessians captured in-engine from the drafter's own
   inputs, GPTQ sym g128, held-out error 0.9–5%) brings the step to 44.5 ms at −1.6% acceptance: net +20%.
3. **q/k/v stay dense.** vLLM's DFlash context-KV precompute slices `qkv_proj.weight` directly.
4. **Two config traps for any quantized draft model** (both written by the bake script): vLLM substring-matches
   `modules_in_block_to_quantize` against runtime names that carry the target's layer offset (`layers.64…`), so the names
   must be index-free; and the draft quant config gets no fused-name mapper, so `mlp.gate_up_proj` must be listed next to
   `mlp.gate_proj`/`mlp.up_proj`. Otherwise the drafter silently loads unquantized and then fails on the packed weights.
5. **Things that did not help.** XPU graph capture (−8%); the R6 draft-vocabulary prefix that helps MTP drafts costs DFlash2
   ~10% acceptance for ~1 ms (`patches/patch_dflash_draft_vocab.py`, kept, off by default).
6. **Memory.** Target + INT4 drafter load at 17.3 GiB (bf16 drafter: 19.4). A 262k request needs 11.5 GiB of KV with the
   drafter's layers and 7.9 are free, so the DFlash2 configuration serves 128k; MTP mode serves 262k. At concurrency 16
   the extra drafter pass per step outweighs acceptance (197 vs 232 aggregate tok/s): this is the single-user configuration.

## Serve

```bash
BAKED=/path/to/Swift-1.5-Qwen3.8-27B-Uncensored-Int4-baked SPEC=7 SPEC_METHOD=dflash \
DRAFT_MODEL=/path/to/Qwen3.8-27B-DFlash2-Int4-baked-drafter DRAFT_DTYPE=bfloat16 DRAFT_QUANT=gptq MAX_LEN=131072 \
bake/serve-baked.sh
```
Equivalent plain vLLM (bf16 everywhere, no patch, CUDA or XPU):
```bash
vllm serve Launch80/Swift-1.5-Qwen3.8-27B-Uncensored-Int4-baked --quantization gptq --dtype bfloat16 --kv-cache-dtype fp8 \
  --max-model-len 131072 --speculative-config '{"method":"dflash","model":"Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter","num_speculative_tokens":7,"quantization":"gptq"}'
```

`dflash/decode_probe.py` is the 8-prompt go/no-go probe used between arms (acceptance from the engine's `spec_decode` counters);
it tracked the quick BetterBench within ±1% on every arm.

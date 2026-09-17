# Embed-only INT8 derivative (post baked-v2)

## Why

`Launch80/Qwen3.8-27B-GPTQ-Int4-baked-v2` still keeps
`model.language_model.embed_tokens.weight` as **FP16**
`[248320, 5120]` (~2.37 GiB). Body / `lm_head` / MTP are already GPTQ INT4.

Quantizing **only** the input embedding reclaims ~1.15 GiB of resident weights
and returns that budget to FP8 KV — enough to move a graphs+MTP6 serve from a
~65k useful window to a **131k-class** `max_model_len` on a 24 GiB B60 without
touching body GPTQ, graphs, MTP depth, or KV dtype.

## Verdict (2026-09-13, Arc Pro B60 / vLLM XPU)

| | baked-v2 FP16 embed | **INT8 embed (ship)** | INT4 embed (reject) |
|---|---|---|---|
| Embed in VRAM | ~2.37 GiB | **~1.18 GiB** | ~0.59 GiB packed |
| Weights + non-torch | 15.72 GiB | **14.57 GiB** | 13.98 GiB |
| KV @ util 0.97 | 3.46 GiB → 65 536 | 5.79 GiB → **136 162** | 6.83 GiB → 160 340 |
| Short decode (graphs+MTP6) | ~120 tok/s | **118.8 (−1.0%)** | **94.4 (−21%)** |
| MTP mean acceptance | ~6.5 | **6.51** | **5.55** |

INT4 embed fails the ~5% decode gate via MTP acceptance collapse. Do not ship it.

## Tooling in this repo

| path | role |
|---|---|
| [`bake/bake_embed_quant.py`](../bake/bake_embed_quant.py) | chunked low-RAM bake; hardlinks body shards |
| [`bake/bake_embed_in_docker.sh`](../bake/bake_embed_in_docker.sh) | run the bake inside the vLLM XPU image |
| [`patches/patch_embed_int8.py`](../patches/patch_embed_int8.py) | boot disk-patch: materialize int8 embed after load |

### Bake

```bash
SRC=/path/to/Qwen3.8-27B-GPTQ-Int4-baked-v2 \
DST=/path/to/Qwen3.8-27B-GPTQ-Int4-baked-v2-embed-int8 \
bash bake/bake_embed_in_docker.sh 8
```

Default index layout (stock vLLM):

- `embed_tokens.weight` → original FP16 shard (loader succeeds)
- `weight_scale` **not** in the weight map
- quantized tensors live in `model-embed-int8.safetensors`
- `config.json` gains `embed_tokens_quant`

### Serve

Same grail flags as baked-v2 (graphs ON, MTP6, fp8 KV, prefix, `max_num_seqs=1`),
plus run the embed patch **before** `vllm serve`:

```bash
python patches/patch_embed_int8.py   # inside the container, against site-packages
vllm serve /model ... --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":6}'
```

The patch edits `UnquantizedEmbeddingMethod.process_weights_after_loading` /
`.embedding` and writes `embed_int8_runtime.py` beside vLLM's quantization
package. EngineCore is a subprocess, so an in-memory monkeypatch of the parent
API process is not enough — disk patch is required.

Look for:

```text
[embed-quant] materialized int8 embed (248320, 5120) on xpu:0 gib=1.184
```

## What this is not

- Not another body / `lm_head` / MTP bake (that is already baked-v2).
- Not a substitute for native low-bit **KV** attention (separate project).
- Not bit-identical to FP16 embed; gate on decode + small paired eval + MTP accept.

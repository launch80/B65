#!/usr/bin/env bash
# Serve the BAKED checkpoint with NO weight patches. The point of the bake is that
# R3 and M1 live on disk now, so this deliberately runs only the three vLLM
# correctness fixes (MTP nightly, MTP boundary, GDN mixed-split) plus the R6 vocab
# prefix, which is a runtime slice and cannot be a weight. If this serves at the same
# quality and speed as the patched stock checkpoint, the bake is correct.
#
#   BAKED=$HOME/bake-model ./serve-baked.sh
set -euo pipefail
BAKED="${BAKED:?path to the assembled baked checkpoint}"
# Optional: path inside $BAKED to the model dir. Lets BAKED be an HF-cache repo dir
# (models--Org--Name) whose snapshots/<rev> is all symlinks into ../../blobs.
BAKED_SUBDIR="${BAKED_SUBDIR:-}"
MODEL_PATH="/model${BAKED_SUBDIR:+/$BAKED_SUBDIR}"
IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d}"  # v0.30.0 (2026-10-01); 0.27.2 nightly was @sha256:f01e24f6c7ff…
# Space-separated: every name routes to the same weights. The HF canonical id and its
# lowercase form (the BetterBench / L80 label) are the same checkpoint, not an alias.
SERVED="${SERVED:-Launch80/Qwen3.8-27B-GPTQ-Int4-baked launch80/Qwen3.8-27B-GPTQ-Int4-baked}"
PORT="${PORT:-8002}"; NAME="${NAME:-vllm-baked}"; GPU_BDF="${GPU_BDF:-0000:86:00.0}"
MAX_LEN="${MAX_LEN:-262144}"; UTIL="${UTIL:-0.95}"; MAX_SEQS="${MAX_SEQS:-32}"
SPEC="${SPEC:-6}"; DRAFT_VOCAB="${DRAFT_VOCAB:-32768}"
# 2026-10-13 DFlash2: SPEC_METHOD=dflash DRAFT_MODEL=/host/path/to/drafter (mounted at /draft) swaps the
# built-in MTP head for a separate block-diffusion drafter (vLLM 0.30.0 has it built in: method "dflash").
# SPEC is then the draft block length minus one (z-lab: 7). The R6 draft-vocab prefix is MTP-only and is skipped.
SPEC_METHOD="${SPEC_METHOD:-mtp}"; DRAFT_MODEL="${DRAFT_MODEL:-}"
# DRAFT_DTYPE=bfloat16 keeps the drafter in bf16 while the target stays $DTYPE (research/dflash/patch_dflash_mixed_dtype.py)
DRAFT_DTYPE="${DRAFT_DTYPE:-}"; DFLASH_DIR="${DFLASH_DIR:-$HERE/../patches}"
# DFlash: the R6 prefix costs 10% acceptance for a 1 ms step saving (2026-10-13), so it is OFF unless DRAFT_VOCAB is set explicitly
[[ "$SPEC_METHOD" == "dflash" ]] && DRAFT_VOCAB="${DFLASH_DRAFT_VOCAB:-0}"
# DRAFT_QUANT=gptq serves a GPTQ-quantized drafter (research/dflash/bake_drafter_int4.py output)
DRAFT_QUANT="${DRAFT_QUANT:-}"
# XPU graph capture: off by default (measured 2026-09-08, off is +2%; see serving/serve-b65.sh).
# XPU_GRAPH=1 to A/B it.
XPU_GRAPH="${XPU_GRAPH:-0}"
# 2026-10-02 Wrapzii-stack knobs (github.com/Wrapzii/k8v4-xpu), all off by default:
#   EMBED_INT8=1   load the INT8 embedding side file via Wrapzii's TP-aware loader
#                  (research/swift/, patches vocab_parallel_embedding.py at boot)
#   DTYPE          float16 (ours) | bfloat16 (Wrapzii)
#   BATCHED_TOKENS --max-num-batched-tokens (Wrapzii: 4224); empty = vLLM default
#   GRAPH_MODE     e.g. FULL_DECODE_ONLY, with CAPTURE_SIZES like [7,14,21,28]
#                  (multiples of 1+SPEC); needs XPU_GRAPH=1. Empty = vLLM default.
EMBED_INT8="${EMBED_INT8:-0}"; DTYPE="${DTYPE:-float16}"; BATCHED_TOKENS="${BATCHED_TOKENS:-}"
GRAPH_MODE="${GRAPH_MODE:-}"; CAPTURE_SIZES="${CAPTURE_SIZES:-}"
SWIFT_DIR="${SWIFT_DIR:-$HOME/Documents/project608/research/swift}"   # Wrapzii INT8-embed loader (optional)
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="${PATCH_DIR:-$HERE/../patches}"
R3_DIR="${R3_DIR:-$HERE/../patches}"   # patch_r3_lmhead_int4.py (R3/R6)
BYPATH_DIR="$HOME/.cache/vllm-dri-by-path-baked"
RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render")"
rm -rf "$BYPATH_DIR"; mkdir -p "$BYPATH_DIR"; ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"
# No patch_draft_mtp_int4 (M1) and P608_R3_LMHEAD_INT4=0: both are on disk now.
# B70_MTP_BF16_DRAFT is deliberately NOT set: the cookbook MTP patch forces the draft
# to a dense build when it is 1, which would discard the baked INT4 draft on disk.
# patch_r3 is still loaded ONLY for its R6 draft-vocab prefix, gated by P608_DRAFT_VOCAB.
PATCH_CMDS='python /patches/patch_mtp_nightly.py; python /patches/patch_mtp_boundary.py'
[[ "$SPEC" != "0" ]] && PATCH_CMDS="$PATCH_CMDS; python /patches/patch_gdn_mixed_split_v5.py"
[[ "$DRAFT_VOCAB" != "0" ]] && PATCH_CMDS="$PATCH_CMDS; python /r3/p608_lmhead_int4.py"
EXTRA_MOUNTS=()
[[ -n "$DRAFT_MODEL" ]] && EXTRA_MOUNTS+=(-v "$DRAFT_MODEL:/draft:ro")
if [[ "$SPEC_METHOD" == "dflash" ]]; then
  EXTRA_MOUNTS+=(-v "$DFLASH_DIR:/dflash:ro")
  if [[ -n "$DRAFT_DTYPE" ]]; then
    EXTRA_MOUNTS+=(-e "P608_DRAFT_DTYPE=$DRAFT_DTYPE")
    PATCH_CMDS="$PATCH_CMDS; python /dflash/patch_dflash_mixed_dtype.py"
  fi
  [[ "$DRAFT_VOCAB" != "0" ]] && PATCH_CMDS="$PATCH_CMDS; python /dflash/patch_dflash_draft_vocab.py"
  for p in ${EXTRA_PATCHES:-}; do PATCH_CMDS="$PATCH_CMDS; python $p"; done   # e.g. EXTRA_PATCHES=/dflash/patch_dflash_debug_quant.py
fi
if [[ "$EMBED_INT8" == "1" ]]; then
  EXTRA_MOUNTS+=(-v "$SWIFT_DIR/embed_int8_runtime.py:/opt/venv/lib/python3.12/site-packages/swift_embed_int8_runtime.py:ro" -v "$SWIFT_DIR:/swift:ro")
  PATCH_CMDS="$PATCH_CMDS; python /swift/patch_embed_int8_v030.py"
fi
read -r -a SERVED_ARR <<< "$SERVED"
ARGS=(vllm serve "$MODEL_PATH" --host 0.0.0.0 --port 8000 --served-model-name "${SERVED_ARR[@]}"
  --quantization gptq --dtype "$DTYPE" --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$UTIL" --kv-cache-dtype fp8 --max-num-seqs "$MAX_SEQS"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3)
if [[ "$SPEC" != "0" && "$SPEC_METHOD" == "dflash" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${SPEC}${DRAFT_QUANT:+,\"quantization\":\"$DRAFT_QUANT\"}}")
elif [[ "$SPEC" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")
fi
[[ -n "$BATCHED_TOKENS" ]] && ARGS+=(--max-num-batched-tokens "$BATCHED_TOKENS")
if [[ -n "$GRAPH_MODE" ]]; then
  CC="{\"cudagraph_mode\":\"$GRAPH_MODE\""; [[ -n "$CAPTURE_SIZES" ]] && CC="$CC,\"cudagraph_capture_sizes\":$CAPTURE_SIZES"; CC="$CC}"
  ARGS+=(--compilation-config "$CC")
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" --device "$RENDER" --ipc=host \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" -v "$BAKED:/model:ro" \
  -v "$PATCH_DIR:/patches:ro" -v "$R3_DIR:/r3:ro" "${EXTRA_MOUNTS[@]}" -p "${PORT}:8000" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e "VLLM_XPU_ENABLE_XPU_GRAPH=${XPU_GRAPH:-0}" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e P608_R3_LMHEAD_INT4=0 -e "P608_DRAFT_VOCAB=$DRAFT_VOCAB" -e B70_DRAFT_MTP_INT4=0 \
  --entrypoint bash "$IMAGE" -lc "set -e; ${PATCH_CMDS}; exec $(printf '%q ' "${ARGS[@]}")"
